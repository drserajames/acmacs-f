"""The tree stages as pipeline steps: what re-runs, and what a moved work area does.

CMAPLE is not installed here, so a stub stands in for it: it writes the synthetic fixture tree
wherever CMAPLE would. ASR uses the in-process Fitch backend, so everything after the build runs
for real.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from af.run import Job, LocalRunner, Resources
from af.run.job import JobResult
from af.seq import processed
from af.seq import select as S
from af.seq.dates import parse as parse_date
from af.store import Store, StoreRef
from af.store.work import Work
from af.tree import export as E
from af.tree import stages
from af.tree.io import i6
from af.tree.io.fasta import write_alignment
from tests.clades.synthetic import build_clone, commit_command, write_clade
from tests.clades.test_fallback_store import raw_dataset
from tests.seq.test_store_build import aligned, record

from .tree_fixtures import KEYS, LEAF_SEQ, records

TREE_STEPS = ("build", "asr", "populate", "publish")

NEWICK = (
    f"({KEYS['o']}:0.1,(({KEYS['a']}:0.2,{KEYS['b']}:0.3):0.05,"
    f"({KEYS['c']}:0.1,{KEYS['d']}:0.1):0.01):0.02);"
)

STUB = f"""#!{sys.executable}
import sys
if "--help" in sys.argv:
    print("CMAPLE version 0.0-stub")
    sys.exit(0)
prefix = sys.argv[sys.argv.index("--prefix") + 1]
open(prefix + ".argv", "w").write(" ".join(sys.argv[1:]) + "\\n")
open(prefix + ".treefile", "w").write({NEWICK!r} + "\\n")
"""


def make_project(root: Path) -> Path:
    """A work area, a store, the export's two files, and a config naming them relatively."""
    root.mkdir(parents=True)
    stub = root / "cmaple-stub"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    Work.create(root / "work")
    Store.create(root / "store")
    write_alignment(root / "alignment.fasta", {KEYS[n]: LEAF_SEQ[n] for n in "oabcd"})
    stages.write_leaves(records(), root / "leaves.parquet")
    (root / "trees.toml").write_text(
        f"""
[paths]
store = "store"
work = "work"

[trees]
threads = 1

[trees.cmaple]
executable = "{stub}"

[trees.subtypes.h3]
outgroup = "{KEYS["o"]}"
asr_backend = "parsimony"

[inputs.h3]
alignment = "alignment.fasta"
leaves = "leaves.parquet"
"""
    )
    return root / "trees.toml"


OUTGROUP = S.Outgroup("EPI_ISL_900000", "EPI900000", "invented outgroup")


def fill(store: Store, *, ragged: bool = False, nextclade: StoreRef | None = None) -> None:
    """The five fixture leaves as a sequence-store version; ``c`` embargoed, ``b`` year-only."""
    recs, found = [], {}
    for index, name in enumerate("oabcd"):
        rec = record(
            index,
            epi_isl=f"EPI_ISL_90000{index}",
            accession=f"EPI90000{index}",
            collection_date=parse_date("2021" if name == "b" else f"202{index}-01-15"),
            embargoed_until="2031-01-01" if name == "c" else "",
        )
        recs.append(rec)
        sequence = LEAF_SEQ[name] + ("A" if ragged and name == "d" else "")
        found[rec.epi_isl] = aligned(rec, nucleotides=sequence, clade="P", subclade="P")
    found = {processed.seq_id(r): found[r.epi_isl] for r in recs}
    processed.publish_pull(
        store, "h3", "p1", processed.isolates_table(recs, "h3", "p1"),
        processed.sequences_table(recs, found, "p1"), inputs=[nextclade] if nextclade else [],
        parameters={}, started=datetime.datetime.now(datetime.UTC),
    )  # fmt: skip


def rules(outgroup: S.Outgroup = OUTGROUP) -> S.SubtypeRules:
    return S.SubtypeRules(
        "h3", outgroup, [S.Rule("host", "host", "human only", optional=True, allow=["Human"])]
    )


def exported_project(root: Path, *, test_only: bool = False, purpose: str | None = None) -> Path:
    """make_project's config, its inputs replaced by an export from its own store."""
    config = make_project(root)
    store = Store.open(root / "store")
    # The dataset the clades step's fallback reads the stored calls from (tests/clades/synthetic).
    fill(store, nextclade=raw_dataset(store, "synthetic", "P"))
    extra = {}
    if test_only:
        extra["test_only"] = E.TestOnly(OUTGROUP, "stand-in for the test")
    E.export(store, "h3", rules(), root / "export", **extra)
    text = config.read_text().replace(
        'alignment = "alignment.fasta"\nleaves = "leaves.parquet"\n',
        'alignment = "export/alignment.fasta"\nleaves = "export/leaves.parquet"\n'
        'export = "export/export.json"\n',
    )
    if purpose is not None:
        text += f'purpose = "{purpose}"\n'
    config.write_text(text)
    return config


def statuses(config_path: Path) -> dict[str, str]:
    return {o.name: o.status for o in stages.run(config_path)["h3"]}


def test_the_stages_run_in_order_and_publish_an_i6_version(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    assert statuses(config) == dict.fromkeys(TREE_STEPS, "ran")

    store = Store.open(tmp_path / "p" / "store")
    ref = store.current("trees", "h3/weekly")
    version = store.version_dir(ref)
    meta = i6.read_metadata(version)
    assert meta["leaves"] == 5
    assert meta["asr"]["backend"] == "parsimony"
    published = json.loads((tmp_path / "p/work/trees/h3/publish/published.json").read_text())
    assert published == ref.to_json()


def test_nothing_re_runs_when_nothing_changed(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    statuses(config)
    assert statuses(config) == dict.fromkeys(TREE_STEPS, "skipped")


def test_a_moved_work_area_is_still_up_to_date(tmp_path: Path) -> None:
    """The point of task 5.12: a store and work area synced elsewhere must not re-run CMAPLE."""
    config = make_project(tmp_path / "laptop")
    statuses(config)
    shutil.copytree(tmp_path / "laptop", tmp_path / "server")
    shutil.rmtree(tmp_path / "laptop")
    server = tmp_path / "server" / "trees.toml"
    # CMAPLE is installed somewhere else there too; that alone must not rebuild.
    moved = str(tmp_path / "server" / "cmaple-stub")
    server.write_text(server.read_text().replace(str(tmp_path / "laptop" / "cmaple-stub"), moved))
    assert moved in server.read_text()
    assert statuses(server) == dict.fromkeys(TREE_STEPS, "skipped")


def test_leaf_metadata_re_runs_populate_but_not_the_build(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    statuses(config)
    changed = records()
    key = KEYS["a"]
    changed[key] = dataclasses.replace(changed[key], country="ELSEWHERELAND")
    stages.write_leaves(changed, tmp_path / "p" / "leaves.parquet")
    assert statuses(config) == {
        "build": "skipped",
        "asr": "skipped",
        "populate": "ran",
        "publish": "ran",
    }


def test_a_changed_clock_threshold_re_runs_populate_not_asr(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    statuses(config)
    text = config.read_text().replace(
        'asr_backend = "parsimony"', 'asr_backend = "parsimony"\nclock_z_threshold = 5.0'
    )
    config.write_text(text)
    assert statuses(config) == {
        "build": "skipped",
        "asr": "skipped",
        "populate": "ran",
        "publish": "ran",
    }


def test_a_first_build_says_why_nothing_was_dropped(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    statuses(config)
    meta = json.loads((tmp_path / "p/work/trees/h3/build/build.json").read_text())
    assert meta["excluded"] == []
    assert meta["prebuild_not_applied"]


def test_leaves_round_trip_and_join_their_sequences(tmp_path: Path) -> None:
    path = tmp_path / "leaves.parquet"
    stages.write_leaves(records(), path)
    back = stages.read_leaves(path, {KEYS[n]: LEAF_SEQ[n] for n in "oabcd"})
    assert back == records()


def test_a_leaf_without_a_sequence_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "leaves.parquet"
    stages.write_leaves(records(), path)
    with pytest.raises(stages.StageError, match="no sequence"):
        stages.read_leaves(path, {KEYS["o"]: LEAF_SEQ["o"]})


def test_incremental_needs_a_previous_tree(tmp_path: Path) -> None:
    with pytest.raises(stages.StageError, match="previous"):
        stages.SubtypeInputs(
            alignment=tmp_path / "a.fasta", leaves=tmp_path / "l.parquet", incremental=True
        )


def test_a_clade_set_without_clones_fails(tmp_path: Path) -> None:
    inputs = stages.SubtypeInputs(
        alignment=tmp_path / "a.fasta", leaves=tmp_path / "l.parquet", clade_set="A(H3N2)"
    )
    with pytest.raises(stages.StageError, match=r"needs \[paths\] nomenclature"):
        stages.clade_source(inputs, None)


def test_inputs_for_an_unconfigured_subtype_are_refused(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    text = config.read_text() + '\n[inputs.h1]\nalignment = "alignment.fasta"\n'
    config.write_text(text + 'leaves = "leaves.parquet"\n')
    with pytest.raises(Exception, match="h1"):
        stages.load_run_config(config)


def with_clades(config: Path, pin: str | None = None) -> None:
    """Point the config at a synthetic nomenclature clone (tests/clades/synthetic.py)."""
    build_clone(config.parent / "clones")
    # The clones directory is the shared [paths] key; the subtype names its clade_set.
    text = config.read_text().replace(
        'work = "work"\n', 'work = "work"\nnomenclature = "clones"\n', 1
    )
    text += 'clade_set = "A(H3N2)"\nnomenclature_repository = "synthetic_HA"\n'
    if pin is not None:
        text += f'clade_pin = "{pin}"\n'
    config.write_text(text)


def test_the_clades_step_publishes_clades_from_the_tree_version(tmp_path: Path) -> None:
    # WS4 labels only a tree whose leaves it can find in a sequence version.
    config = exported_project(tmp_path / "p")
    with_clades(config)
    assert statuses(config) == dict.fromkeys((*TREE_STEPS, "clades"), "ran")
    store = Store.open(tmp_path / "p" / "store")
    tree_meta = i6.read_metadata(store.version_dir(store.current("trees", "h3/weekly")))
    assert tree_meta["clade_set_version"].startswith("synthetic_HA@")
    assert store.current("clades", "h3") is not None


def test_a_pin_move_re_runs_populate_publish_and_clades_in_order(tmp_path: Path) -> None:
    """WS4's guard refuses a tree labelled at another pin; the pipeline must never reach it.

    The new commit changes no subclade file, so only the pin (a parameter) says anything moved.
    """
    # WS4 labels only a tree whose leaves it can find in a sequence version.
    config = exported_project(tmp_path / "p")
    with_clades(config)
    statuses(config)
    clone = tmp_path / "p" / "clones" / "synthetic_HA"
    (clone / "README.md").write_text("a commit that changes no clade\n")
    subprocess.run(["git", "-C", str(clone), "add", "README.md"], check=True)
    subprocess.run(commit_command(clone, "readme"), check=True)
    outcomes = stages.run(config)["h3"]
    assert [(o.name, o.status) for o in outcomes] == [
        ("build", "skipped"),
        ("asr", "skipped"),
        ("populate", "ran"),
        ("publish", "ran"),
        ("clades", "ran"),
    ]
    assert "parameters changed: clade_set_version" in outcomes[2].reason


def test_a_pin_the_clone_is_not_at_is_refused(tmp_path: Path) -> None:
    # WS4 labels only a tree whose leaves it can find in a sequence version.
    config = exported_project(tmp_path / "p")
    with_clades(config, pin="0000000")
    with pytest.raises(Exception, match="not the pinned"):
        stages.run(config)


def test_a_new_clade_file_re_runs_populate(tmp_path: Path) -> None:
    # WS4 labels only a tree whose leaves it can find in a sequence version.
    config = exported_project(tmp_path / "p")
    with_clades(config)
    statuses(config)
    clone = tmp_path / "p" / "clones" / "synthetic_HA"
    write_clade(
        clone / "subclades",
        "P.9",
        "name: P.9\nparent: P\ndefining_mutations:\n- locus: HA1\n  position: 20\n  state: Y\n",
    )
    subprocess.run(["git", "-C", str(clone), "add", "."], check=True)
    subprocess.run(commit_command(clone, "P.9"), check=True)
    ran = {o.name for o in stages.run(config)["h3"] if o.status == "ran"}
    assert ran == {"populate", "publish", "clades"}


class RecordingRunner(LocalRunner):
    """Runs jobs locally, and keeps each one so a test can see what the scheduler would be asked."""

    def __init__(self) -> None:
        super().__init__()
        self.jobs: list[Job] = []

    def run(self, job: Job) -> JobResult:
        self.jobs.append(job)
        return super().run(job)


RESOURCES = """
[trees.subtypes.h3.resources.build]
threads = 3
memory_gb = 8
time_limit_minutes = 90

[trees.subtypes.h3.resources.asr]
threads = 1
time_limit_minutes = 30
"""


def with_resources(config: Path) -> None:
    text = config.read_text()
    head, _, inputs = text.partition("[inputs.h3]")
    config.write_text(head + RESOURCES + "\n[inputs.h3]" + inputs)


def test_build_asr_and_populate_run_as_jobs_with_their_own_resources(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    with_resources(config)
    runner = RecordingRunner()
    stages.run(config, runner=runner)
    by_stage = {
        job.name.rsplit("-", 1)[1]: job for job in runner.jobs if job.name.startswith("tree-")
    }
    assert sorted(by_stage) == ["asr", "build", "populate"]
    assert by_stage["build"].resources == Resources(threads=3, memory_gb=8, time_limit_minutes=90)
    assert by_stage["asr"].resources.time_limit_minutes == 30
    assert by_stage["populate"].resources == Resources(threads=1)  # [trees] threads = 1
    assert "--job" in by_stage["asr"].argv()
    assert by_stage["build"].env["PATH"].startswith(str(Path(sys.executable).parent))


def test_cmaple_gets_the_build_jobs_threads(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    with_resources(config)
    statuses(config)
    argv = (tmp_path / "p/work/trees/h3/build/cmaple/cmaple.argv").read_text()
    assert "-nt 3 " in argv


def test_a_different_thread_count_alone_re_runs_nothing(tmp_path: Path) -> None:
    """The laptop and o give CMAPLE and ASR different threads; that is not a reason to rebuild."""
    config = make_project(tmp_path / "p")
    statuses(config)
    with_resources(config)
    assert statuses(config) == dict.fromkeys(TREE_STEPS, "skipped")


def test_subtypes_build_side_by_side(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    text = "max_parallel = 2\n" + config.read_text()
    second = text[text.index("[trees.subtypes.h3]") : text.index("[inputs.h3]")]
    text += second.replace("h3", "h1") + text[text.index("[inputs.h3]") :].replace("h3", "h1")
    config.write_text(text)
    outcomes = stages.run(config)
    assert sorted(outcomes) == ["h1", "h3"]
    assert all(o.status == "ran" for results in outcomes.values() for o in results)
    store = Store.open(tmp_path / "p" / "store")
    assert store.current("trees", "h1/weekly") is not None


def test_resources_for_an_unknown_stage_are_refused(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    text = config.read_text().replace(
        "[inputs.h3]", "[trees.subtypes.h3.resources.publish]\nthreads = 2\n\n[inputs.h3]"
    )
    config.write_text(text)
    with pytest.raises(Exception, match="unknown stage"):
        stages.load_run_config(config)


def test_a_subtype_called_state_is_refused(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    config.write_text(config.read_text().replace("h3", "state"))
    with pytest.raises(Exception, match="cannot be called 'state'"):
        stages.load_run_config(config)


def test_a_clade_set_needs_the_shared_nomenclature_path(tmp_path: Path) -> None:
    """[paths] nomenclature is the one place the clones directory is configured."""
    config = make_project(tmp_path / "p")
    config.write_text(config.read_text() + 'clade_set = "A(H3N2)"\n')
    with pytest.raises(Exception, match=r"\[paths\] nomenclature .* required"):
        stages.load_run_config(config)


def test_the_old_per_subtype_key_is_rejected(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    config.write_text(config.read_text() + 'nomenclature = "clones"\n')
    with pytest.raises(Exception, match="nomenclature: unknown key"):
        stages.load_run_config(config)


def test_the_stage_assigns_no_continent_and_says_why(tmp_path: Path) -> None:
    config = make_project(tmp_path / "p")
    statuses(config)
    store = Store.open(tmp_path / "p" / "store")
    version = store.version_dir(store.current("trees", "h3/weekly"))
    assert i6.read_metadata(version)["counts"]["continent_not_assigned"].startswith("no country")
    nodes = i6.read_nodes(version, columns=["is_leaf", "continent"]).to_pylist()
    assert {n["continent"] for n in nodes if n["is_leaf"]} == {None}
