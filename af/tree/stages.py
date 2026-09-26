"""The tree stages as :mod:`af.pipeline` steps: build -> asr -> populate -> publish (task 5.12).

One pipeline for every subtype, rooted at ``<work>/trees/``. Its steps are named
``<subtype>.<stage>`` (``h3.build``), so the subtypes are independent and run side by side up to
``max_parallel``. Records live in ``<work>/trees/state/``, and each subtype writes under
``<work>/trees/<subtype>/``::

    build/tree.nwk             the finished tree (rooted, collapsed, ladderized)
    build/alignment.fasta      exactly the sequences in that tree
    build/build.json           the build's counts, and what the pre-build filter dropped
    asr/ancestral.parquet      one state per internal node, keyed by node id
    asr/asr.json               which backend, its version, how long it took
    populate/<purpose>/        the I6 files (af.tree.io.i6), ready to publish
    publish/published.json     the store ref the publish step wrote

**Jobs (task 5.6).** build, asr and populate each run as one job, ``python -m af.tree.stages --job
<stage> --subtype <s> <config>``, through the configured runner: locally a subprocess, under SLURM
one allocation with the stage's own resources (``[trees.subtypes.<s>.resources.<stage>]``). CMAPLE
runs inside the build job's allocation, so it gets the threads that job asked for. ASR is its own
job because its cost is unrelated to the build's and differs by backend. Publish and clades are
short writes to the store and run in the driver. The job finds the tree tools in the interpreter's
own ``bin/`` (the conda/micromamba env), which is prepended to its ``PATH``: a driver started as
``env/bin/python`` without activating the env would otherwise send jobs to nodes that cannot see
``cmaple``. Ctrl-C, SIGTERM and SIGHUP cancel the jobs in flight (af.pipeline, af.run).

**Why the step records survive a sync.** Every input is *named* (``inputs={"alignment": p}``),
and every output lies under the pipeline's ``root`` (``<work>/trees/``), so records are
keyed by role and by relative path, never by where the files happen to live. A work area synced to
the HPC or to ``o`` under another root is up to date there too; only a change of content re-runs.

**Why build does not read the leaf metadata.** CMAPLE is the expensive step. Leaf names, dates and
places change often (a corrected date, a new location alias) and none of them affects topology, so
they are an input of populate, not of build. The pre-build drops are recorded by leaf key and named
at populate, which does read the metadata.

The pipeline starts from export's files (:mod:`af.tree.export`, task 5.1): the aligned FASTA keyed
by leaf key (:func:`af.tree.populate.leaf_key`), the leaf records as Parquet (:func:`write_leaves`
defines the columns), and ``export.json``. Given ``export``, the build checks the other two are its
files and records the sequence-store version it names; that version is an input of the published
tree and is what the clades step's fallback labels from. Without it (hand-made inputs) the tree
records no sequence version and the clades step labels the tree's leaves only.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import logging
import os
import sys
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from af.pipeline import Pipeline, Step, StepContext, StepOutcome
from af.pipeline.config import RunnerSettings, make_runner
from af.run import Job, LocalRunner, Runner
from af.store import Store, StoreRef
from af.store.ref import ExternalInput
from af.store.store import Provenance
from af.store.work import PathsConfig, Work
from af.tree.asr import get_backend
from af.tree.asr.base import AncestralStates
from af.tree.build import build, prune_starting_tree
from af.tree.clock import apply_flags, find_outliers
from af.tree.config import JOB_STAGES, SubtypeSettings, TreeSettings
from af.tree.export import check_matches, read_export
from af.tree.io import i6, newick
from af.tree.io.fasta import read_alignment, write_alignment
from af.tree.model import Tree
from af.tree.populate import LeafRecord, af_clades_assigner, leaf_key, populate
from af.tree.prebuild import ExclusionPlan, ExclusionRule
from af.tree.prebuild import plan as prebuild_plan
from af.util.artefacts import Artefact
from af.util.config import load_config

if TYPE_CHECKING:
    from af.clades.nomenclature import CladeSet

log = logging.getLogger(__name__)

KIND = "trees"
BUILD, ASR, POPULATE, PUBLISH, CLADES = "build", "asr", "populate", "publish", "clades"
STAGES = (BUILD, ASR, POPULATE, PUBLISH, CLADES)
"""``clades`` runs only when the config names a clade set: it publishes ``clades/<subtype>`` from
the tree version (workstream 4's af.clades.from_tree), reading the clades populate assigned."""

STATE_DIR = "state"

CONTINENT_NOT_ASSIGNED = (
    "no country -> continent table yet (workstream 2, rules/locations/countries.tsv); every leaf's "
    "continent is null and draws grey"
)
"""Recorded in tree.json's counts, so a figure with no continent colours says why."""

TEST_PURPOSE = "test"
"""A test-only export (a stand-in outgroup) publishes only under a purpose starting with this:
weekly and report trees are read by people, and must never be rooted on a stand-in."""

CODE_VERSION = 1
"""Bump when a stage's code changes what it writes, so existing records stop counting."""


class StageError(ValueError):
    """A stage's input or configuration is not usable."""


# ---------------------------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class SubtypeInputs:
    """Where one subtype's inputs come from. All paths from config; none has a default."""

    alignment: Path
    """Aligned FASTA keyed by leaf key: export's output (task 5.1)."""
    leaves: Path
    """Leaf records, Parquet, as :func:`write_leaves` writes them: export's other output."""
    export: Path | None = None
    """``export.json`` from :mod:`af.tree.export`, which wrote ``alignment`` and ``leaves``. The
    build checks both are its files, and the sequence-store version it names goes into build.json,
    the tree's provenance and the clades step (whose fallback labels the sequences not on the
    tree from that same version). Absent: hand-made inputs, and the tree records no sequence
    version, which build.json says."""
    previous: Path | None = None
    """The last finished tree. The long-branch rule is measured on it (Sarah, 25 Sep: long
    branches leave before the build), and with ``incremental`` CMAPLE starts from it. Absent on a
    first build, and then nothing is dropped, which build.json says."""
    incremental: bool = False
    """Start CMAPLE from ``previous`` rather than from scratch. Off by default: the incremental
    topology is not yet tested against a from-scratch build (notes/trees/CUT-NODE.md §3b)."""
    clade_set: str | None = None
    """The nomenclature's subtype name (a key of af.clades.nomenclature.HA_REPOSITORIES)."""
    nomenclature_repository: str | None = None
    """The clone's directory name, when it is not the usual one for ``clade_set``."""
    clade_pin: str | None = None
    """The nomenclature commit to use. Checked against the clone, never assumed. Absent: the
    clone's current commit, read when the pipeline is set up."""
    purpose: str = "weekly"

    def __post_init__(self) -> None:
        if self.clade_set is None and (self.nomenclature_repository or self.clade_pin):
            raise StageError("nomenclature_repository and clade_pin need a clade_set")
        if self.incremental and self.previous is None:
            raise StageError("incremental = true needs a previous tree to start from")


@dataclass(frozen=True)
class TreeRunConfig:
    """``[paths]``, ``[trees]`` (af.tree.config.TreeSettings) and ``[inputs.<subtype>]``."""

    paths: PathsConfig
    trees: TreeSettings
    inputs: dict[str, SubtypeInputs]
    runner: RunnerSettings = field(default_factory=lambda: RunnerSettings(kind="local"))
    max_parallel: int = 1
    """How many steps may run at once. Steps of one subtype depend on each other, so in practice
    this is how many subtypes build side by side."""

    def __post_init__(self) -> None:
        if STATE_DIR in self.inputs:
            raise StageError(f"a subtype cannot be called {STATE_DIR!r}: the records live there")
        if self.max_parallel < 1:
            raise StageError("max_parallel must be at least 1")
        unknown = sorted(set(self.inputs) - set(self.trees.subtypes))
        if unknown:
            raise StageError(f"[inputs.X] for subtype(s) with no [trees.subtypes.X]: {unknown}")
        labelled = sorted(name for name, inputs in self.inputs.items() if inputs.clade_set)
        if labelled and self.paths.nomenclature is None:
            raise StageError(
                f"[paths] nomenclature (the nomenclature clones) is required: "
                f"subtype(s) {labelled} have a clade_set"
            )


def load_run_config(path: Path) -> TreeRunConfig:
    return load_config(Path(path), TreeRunConfig)


# ---------------------------------------------------------------------------------------------
# Hand-off files between stages

LEAF_SCHEMA = pa.schema(
    [
        ("leaf_id", pa.string()),
        ("epi_isl", pa.string()),
        ("accession", pa.string()),
        ("name", pa.string()),
        ("collection_date", pa.date32()),
        ("date_precision", pa.string()),
        ("collection_date_first", pa.date32()),
        ("collection_date_last", pa.date32()),
        ("country", pa.string()),
        ("region", pa.string()),
        ("embargoed", pa.bool_()),
    ]
)
"""The leaf records export hands to populate. The sequence is not here: it is in the alignment,
under the same key, and one copy of it is enough (design rule 6)."""

_LEAF_FIELDS = [name for name in LEAF_SCHEMA.names if name != "leaf_id"]


def write_leaves(records: Mapping[str, LeafRecord], path: Path) -> None:
    rows = sorted(records.values(), key=lambda record: record.key)
    columns: dict[str, list[Any]] = {"leaf_id": [record.key for record in rows]}
    for name in _LEAF_FIELDS:
        columns[name] = [getattr(record, name) for record in rows]
    pq.write_table(pa.table(columns, schema=LEAF_SCHEMA), path, compression="zstd")


def read_leaves(path: Path, alignment: Mapping[str, str]) -> dict[str, LeafRecord]:
    """Leaf records joined to their aligned sequences. A leaf with no sequence is an error."""
    table = pq.read_table(path, schema=LEAF_SCHEMA)
    records: dict[str, LeafRecord] = {}
    for row in table.to_pylist():
        key = row.pop("leaf_id")
        if key != leaf_key(row["epi_isl"], row["accession"]):
            raise StageError(f"{path}: leaf_id {key!r} does not match its EPI_ISL and accession")
        if key in records:
            raise StageError(f"{path}: leaf {key!r} appears twice")
        sequence = alignment.get(key)
        if sequence is None:
            raise StageError(f"{path}: leaf {key!r} has no sequence in the alignment")
        records[key] = LeafRecord(nucleotides=sequence, **row)
    return records


def write_states(states: AncestralStates, tree: Tree, directory: Path) -> None:
    """``ancestral.parquet`` (the I6 layout, one row per internal node) and ``asr.json``."""
    internal = [node for node in tree.preorder() if not node.is_leaf]
    missing = [node.id_hex for node in internal if node.node_id not in states.nucleotides]
    if missing:
        raise StageError(f"{states.backend} gave no state for {len(missing)} internal nodes")
    table = pa.table(
        {
            "node_id": [node.id_hex for node in internal],
            "nucleotides": [states.nucleotides[node.node_id] for node in internal],
        },
        schema=i6.ANCESTRAL_SCHEMA,
    )
    pq.write_table(table, directory / i6.ANCESTRAL_FILE, compression="zstd")
    meta = {
        "backend": states.backend,
        "backend_version": states.backend_version,
        "seconds": states.seconds,
        "parameters": dict(states.parameters),
    }
    (directory / ASR_META).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")


def read_states(directory: Path) -> AncestralStates:
    meta = json.loads((directory / ASR_META).read_text())
    table = pq.read_table(directory / i6.ANCESTRAL_FILE, schema=i6.ANCESTRAL_SCHEMA)
    nucleotides = {
        int(node_id, 16): sequence
        for node_id, sequence in zip(
            table["node_id"].to_pylist(), table["nucleotides"].to_pylist(), strict=True
        )
    }
    return AncestralStates(
        nucleotides=nucleotides,
        backend=meta["backend"],
        backend_version=meta["backend_version"],
        seconds=meta["seconds"],
        parameters=meta["parameters"],
    )


def load_finished_tree(path: Path) -> Tree:
    """A finished tree read back. Ids are recomputed from leaf keys, so they match the build's."""
    tree = newick.load(path)
    tree.assign_ids()
    return tree


def _parse_json(path: Path) -> object:
    return json.loads(path.read_text())


def _parse_parquet(path: Path) -> object:
    return pq.read_table(path)


def _parse_tree(path: Path) -> object:
    return load_finished_tree(path)


# ---------------------------------------------------------------------------------------------
# The steps

TREE_FILE = "tree.nwk"
ALIGNMENT_FILE = "alignment.fasta"
BUILD_META = "build.json"
ASR_META = "asr.json"
PUBLISHED_FILE = "published.json"


@dataclass(frozen=True)
class Layout:
    """Where one subtype's stages write, under its work directory (the pipeline root)."""

    root: Path
    purpose: str

    def stage(self, name: str) -> Path:
        return self.root / name

    @property
    def tree(self) -> Path:
        return self.stage(BUILD) / TREE_FILE

    @property
    def alignment(self) -> Path:
        return self.stage(BUILD) / ALIGNMENT_FILE

    @property
    def build_meta(self) -> Path:
        return self.stage(BUILD) / BUILD_META

    @property
    def ancestral(self) -> Path:
        return self.stage(ASR) / i6.ANCESTRAL_FILE

    @property
    def asr_meta(self) -> Path:
        return self.stage(ASR) / ASR_META

    @property
    def i6(self) -> Path:
        return self.stage(POPULATE) / self.purpose

    @property
    def published(self) -> Path:
        return self.stage(PUBLISH) / PUBLISHED_FILE


def layout_for(paths: PathsConfig, subtype: str, purpose: str) -> Layout:
    """The subtype's work directory. The work area must exist: a typo must not start afresh."""
    work = Work.open(paths.work).dataset(KIND, subtype)
    return Layout(root=work.root, purpose=purpose)


def tree_steps(
    subtype: str,
    settings: TreeSettings,
    inputs: SubtypeInputs,
    layout: Layout,
    store_root: Path,
    clones: Path | None = None,
    *,
    jobs_from: Path | None = None,
) -> list[Step]:
    """The steps for one subtype, in order, named ``<subtype>.<stage>``. Inputs named by role.

    ``jobs_from``: the config file. When given, build, asr and populate run as jobs through the
    pipeline's runner (:func:`_as_job`); when None, every step does its work in this process,
    which is what the job itself does.
    """
    sub = settings.for_subtype(subtype)
    clades = clade_source(inputs, clones)
    steps = [
        _build_step(subtype, settings, sub, inputs, layout),
        _asr_step(sub, settings.resources_for(subtype, ASR).threads, layout),
        _populate_step(subtype, sub, inputs, clades, layout),
        _publish_step(subtype, inputs, clades, layout, store_root),
    ]
    if clades is not None:
        steps.append(_clades_step(clades, layout, store_root))
    if jobs_from is not None:
        steps = [
            _as_job(step, subtype, settings, layout, jobs_from) if step.name in JOB_STAGES else step
            for step in steps
        ]
    return [
        dataclasses.replace(
            step,
            name=step_name(subtype, step.name),
            after=[step_name(subtype, name) for name in step.after],
        )
        for step in steps
    ]


def step_name(subtype: str, stage: str) -> str:
    return f"{subtype}.{stage}"


def _as_job(step: Step, subtype: str, settings: TreeSettings, layout: Layout, config: Path) -> Step:
    """The same step, its work done by ``python -m af.tree.stages --job`` through the runner.

    The job's outputs are the step's, so the runner checks them before the step can succeed, and
    the job runs with the interpreter (and so the af) that is running this driver.
    """
    stage = step.name

    def submit(context: StepContext) -> None:
        directory = layout.stage(stage)
        directory.mkdir(parents=True, exist_ok=True)
        job = Job.python_module(
            f"tree-{subtype}-{stage}",
            __name__,
            ["--job", stage, "--subtype", subtype, config],
            cwd=directory,
            log=directory / "job.log",
            outputs=step.outputs,
            resources=settings.resources_for(subtype, stage),
            env={"PATH": tool_path()},
        )
        context.runner.run(job)

    return dataclasses.replace(step, action=submit)


def tool_path() -> str:
    """``PATH`` for a job: this interpreter's ``bin/`` first, where the env keeps the tree tools."""
    here = str(Path(sys.executable).parent)
    inherited = os.environ.get("PATH", "")
    return here if not inherited else f"{here}{os.pathsep}{inherited}"


def _build_step(
    subtype: str,
    settings: TreeSettings,
    sub: SubtypeSettings,
    inputs: SubtypeInputs,
    layout: Layout,
) -> Step:
    named: dict[str, Path] = {"alignment": inputs.alignment}
    if inputs.export is not None:
        named["export"] = inputs.export
    if inputs.previous is not None:
        named["previous"] = inputs.previous
    cmaple = settings.cmaple_for(subtype)
    rule = sub.long_branch_rule() if sub.drop_long_branches else None
    parameters = {
        "code_version": CODE_VERSION,
        # Not the executable's path or the thread count: both differ between the laptop, the HPC
        # and o, and neither is a reason to rebuild. The version CMAPLE reports is in build.json.
        # After upgrading CMAPLE, re-run deliberately (--force build).
        "cmaple": {
            k: v
            for k, v in dataclasses.asdict(cmaple).items()
            if k not in ("executable", "threads")
        },
        "outgroup": sub.outgroup,
        "collapse_tolerance": sub.collapse_tolerance,
        "long_branch_threshold": None if rule is None else rule.threshold,
        "incremental": inputs.incremental,
    }

    def action(context: StepContext) -> None:
        out = layout.stage(BUILD)
        out.mkdir(parents=True, exist_ok=True)
        source = _checked_export(subtype, sub, inputs)
        exclude, not_dropped = _prebuild(sub, inputs, rule)
        starting = None
        if inputs.incremental:
            assert inputs.previous is not None  # SubtypeInputs checks this
            keep = sorted(set(read_alignment(inputs.alignment)) - exclude.keys())
            starting, _ = prune_starting_tree(
                newick.load(inputs.previous), keep, out / "starting.nwk"
            )
        result = build(
            inputs.alignment,
            sub.outgroup,
            out / "cmaple",
            settings=cmaple,
            starting_tree=starting,
            collapse_tolerance=sub.collapse_tolerance,
            runner=context.runner,
            exclude=exclude,
        )
        newick.dump(result.tree, layout.tree, precision=12)
        in_tree = {leaf.name or "" for leaf in result.tree.leaves()}
        sequences = read_alignment(inputs.alignment)
        write_alignment(layout.alignment, {key: sequences[key] for key in sorted(in_tree)})
        meta = {
            "subtype": subtype,
            "counts": result.counts,
            "prebuild": exclude.counts,
            "prebuild_not_applied": not_dropped,
            "excluded": exclude.records(),
            "source": source,
        }
        layout.build_meta.write_text(json.dumps(meta, indent=1, default=str) + "\n")

    return Step(
        name=BUILD,
        action=action,
        inputs=named,
        parameters=parameters,
        outputs=[
            Artefact(layout.tree, parse=_parse_tree),
            Artefact(layout.alignment, parse=read_alignment),
            Artefact(layout.build_meta, parse=_parse_json),
        ],
    )


def _checked_export(
    subtype: str, sub: SubtypeSettings, inputs: SubtypeInputs
) -> dict[str, Any] | None:
    """What build.json records about the export, after checking the inputs are its files."""
    if inputs.export is None:
        return None
    record = read_export(inputs.export)
    check_matches(record, inputs.alignment, inputs.leaves)
    if record.outgroup != sub.outgroup:
        raise StageError(
            f"{subtype}: the export's outgroup {record.outgroup!r} is not the tree config's "
            f"{sub.outgroup!r}; the tree would be rooted on a sequence the rules did not pin"
        )
    _refuse_test_only(subtype, {"test_only": record.test_only}, inputs.purpose)
    return {
        "sequences": record.sequences.to_json(),
        "leaves": record.leaves,
        "embargoed": record.embargoed,
        "test_only": record.test_only,
        "export": str(inputs.export),
    }


def _refuse_test_only(subtype: str, source: Mapping[str, Any] | None, purpose: str) -> None:
    """Checked at build, so a mislabelled run fails before CMAPLE, and again at publish."""
    if source is not None and source["test_only"] and not purpose.startswith(TEST_PURPOSE):
        raise StageError(
            f"{subtype}: a test-only export ({source['test_only']}) publishes only under a "
            f"purpose starting with {TEST_PURPOSE!r}, not {purpose!r}"
        )


def build_source(layout: Layout) -> dict[str, Any] | None:
    """build.json's record of the export (None for hand-made inputs)."""
    source: dict[str, Any] | None = json.loads(layout.build_meta.read_text()).get("source")
    return source


def _source_sequences(layout: Layout) -> StoreRef | None:
    source = build_source(layout)
    return None if source is None else StoreRef.from_json(source["sequences"])


def _prebuild(
    sub: SubtypeSettings, inputs: SubtypeInputs, rule: ExclusionRule | None
) -> tuple[ExclusionPlan, str | None]:
    """The pre-build drops, and why there are none when there are none (never silently)."""
    reason = sub.no_long_branch_reason()
    if reason is not None or rule is None:
        return ExclusionPlan(), reason or "no long-branch rule"
    if inputs.previous is None:
        return ExclusionPlan(), "no previous tree to measure long branches on (first build)"
    previous = newick.load(inputs.previous)
    return prebuild_plan(previous, rule, keep=[sub.outgroup], source=str(inputs.previous)), None


def _asr_step(sub: SubtypeSettings, threads: int, layout: Layout) -> Step:
    def action(_: StepContext) -> None:
        out = layout.stage(ASR)
        out.mkdir(parents=True, exist_ok=True)
        tree = load_finished_tree(layout.tree)
        backend = get_backend(sub.asr_backend)
        states = backend.reconstruct(tree, layout.alignment, out / "work", threads=threads)
        write_states(states, tree, out)

    return Step(
        name=ASR,
        action=action,
        inputs={"tree": layout.tree, "alignment": layout.alignment},
        # Threads are not a parameter: the laptop and o differ, and it is not a reason to re-run.
        parameters={"code_version": CODE_VERSION, "backend": sub.asr_backend},
        outputs=[
            Artefact(layout.ancestral, parse=_parse_parquet),
            Artefact(layout.asr_meta, parse=_parse_json),
        ],
    )


@dataclass(frozen=True)
class CladeSource:
    """The clade set a run uses, fixed to one commit when the pipeline is set up.

    The commit is a step *parameter* of populate, not only an input hash, because a pin can move
    without the subclade files changing, and the clades step refuses a tree whose clades were
    assigned at another version (af.clades.from_tree). Changing the pin therefore re-runs populate,
    publish and clades, in that order.
    """

    subtype: str
    clone: Path
    commit: str

    @property
    def subclades(self) -> Path:
        return self.clone / "subclades"

    @property
    def version(self) -> str:
        """As af.clades.nomenclature writes it, and as tree.json records it."""
        return f"{self.clone.name}@{self.commit}"

    def load(self) -> CladeSet:
        from af.clades.nomenclature import Pin, load_clade_set

        pin = Pin(subtype=self.subtype, repository=self.clone.name, commit=self.commit)
        return load_clade_set(self.subtype, self.clone.parent, pin, repository=self.clone.name)

    def provenance(self) -> ExternalInput:
        """The same record on the tree version and on the clades version: path, hash, commit."""
        return ExternalInput.of(self.subclades, version=self.commit)


def clade_source(inputs: SubtypeInputs, clones: Path | None) -> CladeSource | None:
    """``clones``: [paths] nomenclature, the directory holding the nomenclature clones."""
    if inputs.clade_set is None:
        return None
    if clones is None:
        raise StageError(f"clade_set {inputs.clade_set!r} needs [paths] nomenclature")
    from af.clades.nomenclature import HA_REPOSITORIES, head_commit

    repository = inputs.nomenclature_repository or HA_REPOSITORIES.get(inputs.clade_set)
    if repository is None:
        known = ", ".join(sorted(HA_REPOSITORIES))
        raise StageError(f"clade_set {inputs.clade_set!r} is not one of: {known}")
    clone = clones / repository
    commit = inputs.clade_pin or head_commit(clone)
    return CladeSource(subtype=inputs.clade_set, clone=clone, commit=commit)


def _populate_step(
    subtype: str,
    sub: SubtypeSettings,
    inputs: SubtypeInputs,
    clades: CladeSource | None,
    layout: Layout,
) -> Step:
    named = {
        "tree": layout.tree,
        "ancestral": layout.ancestral,
        "asr_meta": layout.asr_meta,
        "alignment": layout.alignment,
        "build_meta": layout.build_meta,
        "leaves": inputs.leaves,
    }
    if clades is not None:
        named["clades"] = clades.subclades
    clock = sub.clock_settings()
    parameters = {
        "code_version": CODE_VERSION,
        "purpose": inputs.purpose,
        "branch_scale": sub.branch_scale,
        "outgroup": sub.outgroup,
        "clade_set_version": None if clades is None else clades.version,
        "clock": _recordable(dataclasses.asdict(clock)),
    }

    def action(_: StepContext) -> None:
        out = layout.i6
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.iterdir():
            stale.unlink()
        tree = load_finished_tree(layout.tree)
        leaves = read_leaves(inputs.leaves, read_alignment(layout.alignment))
        names = {key: record.name for key, record in leaves.items()}
        excluded = json.loads(layout.build_meta.read_text())["excluded"]
        for row in excluded:
            row["name"] = names.get(row["leaf_id"])
        populated = populate(
            tree,
            subtype,
            leaves,
            read_states(layout.stage(ASR)),
            branch_scale=sub.scale,
            backend=get_backend(sub.asr_backend),
            assign_clades=None if clades is None else af_clades_assigner(clades.load()),
            # No continent until WS2's country -> continent table exists (rules/locations/
            # countries.tsv, Q55-Q57): GISAID's region is not the legend's vocabulary, and a
            # region -> legend map would still misplace Russia, the Middle East and Central America.
            continent_of=None,
            outgroup=sub.outgroup,
            excluded=excluded,
        )
        populated.counts["continent_not_assigned"] = CONTINENT_NOT_ASSIGNED
        apply_flags(populated, find_outliers(populated, clock))
        i6.write(populated, out, inputs.purpose, build_source(layout))

    return Step(
        name=POPULATE,
        action=action,
        inputs=named,
        parameters=parameters,
        outputs=[
            Artefact(layout.i6 / i6.TREE_FILE, parse=_parse_tree),
            Artefact(layout.i6 / i6.NODES_FILE, parse=_parse_parquet),
            Artefact(layout.i6 / i6.META_FILE, parse=_parse_json),
        ],
    )


def _recordable(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Settings as a step parameter: dates become ISO strings, which is how a record keeps them."""
    result: dict[str, Any] = json.loads(json.dumps(dict(settings), default=str))
    return result


def _publish_step(
    subtype: str,
    inputs: SubtypeInputs,
    clades: CladeSource | None,
    layout: Layout,
    store_root: Path,
) -> Step:
    def action(_: StepContext) -> None:
        started = datetime.datetime.now(datetime.UTC)
        store = Store.open(store_root)
        source = build_source(layout)
        _refuse_test_only(subtype, source, inputs.purpose)
        provenance_inputs: list[StoreRef | ExternalInput] = [
            ExternalInput.of(inputs.alignment),
            ExternalInput.of(inputs.leaves),
        ]
        sequences = _source_sequences(layout)
        if sequences is not None:
            provenance_inputs.append(sequences)
        if clades is not None:
            provenance_inputs.append(clades.provenance())
        with store.build(KIND, f"{subtype}/{inputs.purpose}") as builder:
            builder.copy(layout.i6, ".")  # from the work area: copy, never link (store.link)
            meta = i6.read_metadata(layout.i6)
            summary = {
                "leaves": meta["leaves"],
                "internal_nodes": meta["internal_nodes"],
                "branch_scale": meta["branch_scale"],
                "embargoed_leaves": meta["embargoed_leaves"],
                "test_only": None if source is None else source["test_only"],
            }
            provenance = Provenance(
                step=f"{KIND}.{PUBLISH}",
                inputs=tuple(provenance_inputs),
                parameters={"subtype": subtype, "purpose": inputs.purpose},
                started=started,
                finished=datetime.datetime.now(datetime.UTC),
            )
            ref = builder.publish(provenance, summary=summary)
        layout.published.parent.mkdir(parents=True, exist_ok=True)
        layout.published.write_text(json.dumps(ref.to_json(), indent=1, sort_keys=True) + "\n")

    return Step(
        name=PUBLISH,
        action=action,
        inputs={"i6": layout.i6, "build_meta": layout.build_meta},
        parameters={"code_version": CODE_VERSION, "purpose": inputs.purpose},
        # The I6 directory is not itself a declared output, so the driver cannot infer this.
        after=[POPULATE],
        outputs=[Artefact(layout.published, parse=_parse_json)],
    )


def _clades_step(clades: CladeSource, layout: Layout, store_root: Path) -> Step:
    written = layout.stage(CLADES) / PUBLISHED_FILE

    def action(_: StepContext) -> None:
        from af.clades.from_tree import publish_from_tree

        started = datetime.datetime.now(datetime.UTC)
        tree = StoreRef.from_json(json.loads(layout.published.read_text()))
        ref = publish_from_tree(
            Store.open(store_root),
            tree,
            clades.subtype,
            clades.load(),
            nomenclature=[clades.provenance()],
            started=started,
            sequences=_source_sequences(layout),
        )
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text(json.dumps(ref.to_json(), indent=1, sort_keys=True) + "\n")

    return Step(
        name=CLADES,
        action=action,
        inputs={
            "tree_ref": layout.published,
            "clades": clades.subclades,
            "build_meta": layout.build_meta,
        },
        parameters={"code_version": CODE_VERSION, "clade_set_version": clades.version},
        outputs=[Artefact(written, parse=_parse_json)],
    )


# ---------------------------------------------------------------------------------------------
# Running


def trees_root(paths: PathsConfig) -> Path:
    """``<work>/trees``. The work area must exist: a typo must not start afresh and re-run all."""
    return Work.open(paths.work).root / KIND


def pipeline_for(
    config_path: Path,
    config: TreeRunConfig,
    subtypes: Sequence[str],
    runner: Runner | None = None,
) -> Pipeline:
    """One pipeline over ``subtypes``; build, asr and populate run as jobs."""
    steps: list[Step] = []
    for subtype in subtypes:
        inputs = _inputs(config, subtype)
        layout = layout_for(config.paths, subtype, inputs.purpose)
        steps += tree_steps(
            subtype,
            config.trees,
            inputs,
            layout,
            config.paths.store,
            config.paths.nomenclature,
            jobs_from=config_path,
        )
    root = trees_root(config.paths)
    return Pipeline(
        steps,
        state_dir=root / STATE_DIR,
        runner=runner or make_runner(config.runner, config_path),
        root=root,
        max_parallel=config.max_parallel,
    )


def _inputs(config: TreeRunConfig, subtype: str) -> SubtypeInputs:
    try:
        return config.inputs[subtype]
    except KeyError:
        raise StageError(
            f"no [inputs.{subtype}] in the config; it names: {', '.join(sorted(config.inputs))}"
        ) from None


def run(
    config_path: Path,
    subtypes: Sequence[str] | None = None,
    *,
    until: str | None = None,
    force: Collection[str] = (),
    runner: Runner | None = None,
) -> dict[str, list[StepOutcome]]:
    """Run what is out of date, for every subtype at once. ``until`` and ``force`` name stages.

    The first failure stops the run; subtypes that finished keep their records.
    """
    config = load_run_config(config_path)
    chosen = list(subtypes) if subtypes else sorted(config.inputs)
    pipeline = pipeline_for(Path(config_path), config, chosen, runner)
    targets = None if until is None else [step_name(s, until) for s in chosen]
    forced = [step_name(s, stage) for s in chosen for stage in force]
    forced = [name for name in forced if name in pipeline.steps]
    outcomes: dict[str, list[StepOutcome]] = {subtype: [] for subtype in chosen}
    for outcome in pipeline.run(targets, force=forced):
        subtype, _, stage = outcome.name.partition(".")
        outcomes[subtype].append(dataclasses.replace(outcome, name=stage))
    return outcomes


def run_job(config_path: Path, subtype: str, stage: str) -> None:
    """One stage's work, in this process: what a job does on its node.

    Tools the stage calls (CMAPLE) run as local subprocesses, inside the job's allocation.
    """
    config = load_run_config(config_path)
    inputs = _inputs(config, subtype)
    layout = layout_for(config.paths, subtype, inputs.purpose)
    steps = tree_steps(
        subtype, config.trees, inputs, layout, config.paths.store, config.paths.nomenclature
    )
    wanted = step_name(subtype, stage)
    step = next((step for step in steps if step.name == wanted), None)
    if step is None or stage not in JOB_STAGES:
        raise StageError(f"{stage!r} is not a job stage; job stages: {JOB_STAGES}")
    step.action(StepContext(step=step, runner=LocalRunner()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--subtype", action="append", help="default: every [inputs.X]")
    parser.add_argument("--until", choices=STAGES, help="stop after this stage")
    parser.add_argument("--force", action="append", default=[], choices=STAGES)
    parser.add_argument("--job", choices=JOB_STAGES, help="internal: do one stage's work here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.job:
        if not args.subtype or len(args.subtype) != 1:
            parser.error("--job needs exactly one --subtype")
        run_job(args.config, args.subtype[0], args.job)
        return 0
    run(args.config, args.subtype, until=args.until, force=args.force)  # the driver logs outcomes
    return 0


if __name__ == "__main__":
    sys.exit(main())
