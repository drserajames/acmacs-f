"""Build the sequence store from GISAID pulls: import, then align and publish per pull.

Two commands, both driven by one TOML config (see :class:`SequencesConfig`)::

    python -m af.seq.build CONFIG import --source definitive    # extractor → raw/gisaid
    python -m af.seq.build CONFIG store --pull definitive-2024-0312-h3n2

``import`` copies every pull of an extractor set into ``raw/gisaid/<pull-id>``; a pull
already there publishes nothing new. ``store`` reads one raw pull, places each record in
its subtype dataset by the configured rules, aligns each group with Nextclade, and
publishes that pull's partitions of ``sequences/<subtype>`` (:mod:`af.seq.processed`).
Alignments are pipeline steps in the work area, so re-running an unchanged pull does not
re-run Nextclade.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.pipeline.config import RunnerSettings, make_runner
from af.pipeline.driver import Pipeline
from af.run import Runner
from af.seq import nextclade as nc
from af.seq import processed
from af.seq.gisaid import SequenceRecord, join, read_fasta, read_workbook
from af.seq.processed import PlacementRule
from af.seq.pulls import find_pulls, import_pull, open_pull
from af.store import DatasetWork, PathsConfig, Store, Work
from af.store.ref import StoreRef
from af.util.config import ConfigError, load_config

WORK_DATASET = "gisaid"


@dataclass(frozen=True)
class DatasetConfig:
    """One subtype's Nextclade dataset pin and the mature-HA length it must give."""

    path: str
    tag: str
    sha256: str
    mature_nt: int


@dataclass(frozen=True)
class NextcladeConfig:
    binary: Path
    version: str
    # R3's limits: align_step reports how many records each would fail. The store keeps
    # the facts, not the verdict; selection applies the limits again from its own rules.
    max_unknown_aa: int
    max_deleted_aa: int
    datasets: dict[str, DatasetConfig]
    threads: int = 4


@dataclass(frozen=True)
class PlacementConfig:
    gisaid_subtype: str
    lineage: str
    dataset: str
    reason: str


@dataclass(frozen=True)
class LineageCheckConfig:
    gisaid_subtype: str
    candidates: dict[str, str]  # GISAID lineage -> dataset
    min_margin: int
    reason: str


@dataclass(frozen=True)
class SequencesConfig:
    paths: PathsConfig
    runner: RunnerSettings
    nextclade: NextcladeConfig
    placement: list[PlacementConfig]
    #: Extractor set name → its directory, e.g. definitive = ".../out/definitive".
    sources: dict[str, Path]
    #: The extractor's subtype label → what the GISAID query asked for, in the name
    #: normaliser's terms ("A(H3N2)", "B"): a name that disagrees is flagged, not relabelled.
    source_subtypes: dict[str, str] = field(default_factory=dict)
    #: Subtypes placed by alignment rather than GISAID's label (processed.LineageCheck).
    lineage_check: list[LineageCheckConfig] = field(default_factory=list)


def import_source(config: SequencesConfig, source: str) -> list[StoreRef]:
    if source not in config.sources:
        raise ConfigError("<config>", [f"sources: no source {source!r}"])
    store = Store.open(config.paths.store)
    return [import_pull(store, pull) for pull in find_pulls(config.sources[source], source)]


def store_pull(config: SequencesConfig, pull_id: str, runner: Runner) -> dict[str, StoreRef]:
    """Align one raw pull and publish its partitions; returns the ref per dataset."""
    started = datetime.datetime.now(datetime.UTC)
    store = Store.open(config.paths.store)
    area = Work.open(config.paths.work).dataset("sequences", f"{WORK_DATASET}/{pull_id}")
    raw = open_pull(store, pull_id)
    label = pull_id.rsplit("-", 1)[-1]
    if label not in config.source_subtypes:
        raise ConfigError("<config>", [f"source_subtypes: no entry for pull label {label!r}"])

    workbook = read_workbook(raw.workbook)
    records, counts = join(
        read_fasta(raw.sequences), workbook.rows, subtype=config.source_subtypes[label]
    )
    rules = [
        PlacementRule(p.gisaid_subtype, p.lineage, p.dataset, p.reason) for p in config.placement
    ]
    checks = _lineage_checks(config)
    references = _References(config.nextclade, store)

    alignments: dict[str, dict[str, nc.Aligned]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    placed: dict[str, list[SequenceRecord]] = {}
    lineage_flags: dict[str, dict[str, int]] = {}
    for subtype, check in checks.items():
        checked = [r for r in records if r.subtype == subtype]
        if not checked:
            continue
        for dataset in sorted(set(check.candidates.values())):
            summaries[dataset], alignments[dataset] = _align(
                config.nextclade, references, dataset, checked, area, runner
            )
        groups, flags = processed.place_by_alignment(checked, rules, check, alignments)
        lineage_flags[subtype] = dict(sorted(flags.items()))
        _merge(placed, groups)
    unchecked = processed.place([r for r in records if r.subtype not in checks], rules)
    for dataset, group in sorted(unchecked.items()):
        if dataset in alignments:
            problem = f"placement: {dataset!r} is a lineage_check candidate and also receives"
            raise ConfigError("<config>", [f"{problem} unchecked records"])
        summaries[dataset], alignments[dataset] = _align(
            config.nextclade, references, dataset, group, area, runner
        )
    _merge(placed, unchecked)

    refs: dict[str, StoreRef] = {}
    for dataset, group in sorted(placed.items()):
        refs[dataset] = processed.publish_pull(
            store,
            dataset,
            pull_id,
            processed.isolates_table(group, dataset, pull_id),
            processed.sequences_table(group, alignments[dataset], pull_id),
            inputs=[raw.ref, *references.refs_for(dataset, checks)],
            parameters={
                "pull": pull_id,
                "read": counts.to_json(),
                "workbook_line_breaks": dict(workbook.line_breaks),
                "placement": _placement_used(config.placement, group, dataset),
                "lineage_check": lineage_flags,
                "alignment": summaries[dataset],
            },
            started=started,
        )
    return refs


def _lineage_checks(config: SequencesConfig) -> dict[str, processed.LineageCheck]:
    checks = {}
    for item in config.lineage_check:
        if item.gisaid_subtype in checks:
            raise ConfigError("<config>", [f"lineage_check: {item.gisaid_subtype!r} twice"])
        missing = sorted(set(item.candidates.values()) - set(config.nextclade.datasets))
        if missing:
            raise ConfigError("<config>", [f"lineage_check: no Nextclade dataset for {missing}"])
        checks[item.gisaid_subtype] = processed.LineageCheck(
            item.gisaid_subtype, dict(item.candidates), item.min_margin, item.reason
        )
    for rule in config.placement:
        check = checks.get(rule.gisaid_subtype)
        if check is not None and rule.dataset not in check.candidates.values():
            row = f"{rule.gisaid_subtype!r}/{rule.lineage!r} -> {rule.dataset!r}"
            raise ConfigError("<config>", [f"placement: {row} is not a lineage_check candidate"])
    return checks


class _References:
    """Each dataset's pinned Nextclade reference, fetched into the store once per run."""

    def __init__(self, settings: NextcladeConfig, store: Store) -> None:
        self.settings = settings
        self.store = store
        self._refs: dict[str, StoreRef] = {}

    def get(self, dataset: str) -> tuple[DatasetConfig, StoreRef, Path]:
        pin = self.settings.datasets.get(dataset)
        if pin is None:
            raise ConfigError("<config>", [f"nextclade.datasets: no dataset {dataset!r}"])
        if dataset not in self._refs:
            nc_pin = nc.DatasetPin(pin.path, pin.tag, pin.sha256)
            self._refs[dataset] = nc.fetch_dataset(nc_pin, self.store)
        ref = self._refs[dataset]
        return pin, ref, self.store.resolve(ref) / nc.DATASET_DIR

    def refs_for(
        self, dataset: str, checks: Mapping[str, processed.LineageCheck]
    ) -> list[StoreRef]:
        """The dataset's own reference, and every reference its records were placed against."""
        wanted = {dataset}
        for check in checks.values():
            if dataset in check.candidates.values():
                wanted |= set(check.candidates.values())
        return [self._refs[name] for name in sorted(wanted) if name in self._refs]


def _merge(
    into: dict[str, list[SequenceRecord]], groups: Mapping[str, list[SequenceRecord]]
) -> None:
    for dataset, group in groups.items():
        into.setdefault(dataset, []).extend(group)


def _align(
    settings: NextcladeConfig,
    references: _References,
    dataset: str,
    group: Sequence[SequenceRecord],
    area: DatasetWork,
    runner: Runner,
) -> tuple[dict[str, Any], dict[str, nc.Aligned]]:
    """Align ``group`` against ``dataset``'s reference, as a resumable pipeline step."""
    pin, _, dataset_dir = references.get(dataset)
    out = area.tmp / dataset
    out.mkdir(parents=True, exist_ok=True)
    fasta = out / "input.fasta"
    nc.write_input(((processed.seq_id(r), r.nucleotides) for r in processed.by_key(group)), fasta)
    result = out / "nextclade"
    step = nc.align_step(
        "align",
        {
            "binary": settings.binary,
            "nextclade_version": settings.version,
            "fasta": fasta,
            "dataset_dir": dataset_dir,
            "out_dir": result,
            "expected_mature_nt": pin.mature_nt,
            "max_unknown_aa": settings.max_unknown_aa,
            "max_deleted_aa": settings.max_deleted_aa,
            "threads": settings.threads,
        },
    )
    Pipeline([step], state_dir=area.state / dataset, runner=runner).run()
    reference = nc.read_reference(dataset_dir, pin.mature_nt)
    aligned = {record.seq_id: record for record in nc.read_alignment(result, reference)}
    return json.loads((result / nc.SUMMARY).read_text()), aligned


def _placement_used(
    rules: Sequence[PlacementConfig], group: Sequence[SequenceRecord], dataset: str
) -> list[dict[str, Any]]:
    """Each rule into this dataset with how many records it placed; 0 is reported too."""
    used = []
    for rule in rules:
        if rule.dataset == dataset:
            n = sum(
                1 for r in group if (r.subtype, r.lineage) == (rule.gisaid_subtype, rule.lineage)
            )
            used.append({"subtype": rule.gisaid_subtype, "lineage": rule.lineage,
                         "reason": rule.reason, "records": n})  # fmt: skip
    return used


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("import").add_argument("--source", required=True)
    commands.add_parser("store").add_argument("--pull", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config, SequencesConfig)
    if args.command == "import":
        refs = import_source(config, args.source)
        _print({ref.dataset: ref.version for ref in refs})
    else:
        runner = make_runner(config.runner, args.config)
        refs_by = store_pull(config, args.pull, runner)
        _print({ref.dataset: ref.version for ref in refs_by.values()})
    return 0


def _print(data: Mapping[str, str]) -> None:
    for key, value in data.items():
        print(f"{key}\t{value}")


if __name__ == "__main__":
    sys.exit(main())
