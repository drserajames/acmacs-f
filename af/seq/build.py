"""Build the sequence store from GISAID pulls: import, then align and publish per pull.

Two commands, both driven by one TOML config (see :class:`SequencesConfig`)::

    python -m af.seq.build CONFIG import --source definitive    # extractor → raw/gisaid
    python -m af.seq.build CONFIG store --pull definitive-2024-0312-h3n2
    python -m af.seq.build CONFIG store --pulls-file sweep.txt --batch seq-sweep

``import`` copies every pull of an extractor set into ``raw/gisaid/<pull-id>``; a pull
already there publishes nothing new. ``store`` reads each raw pull named, places each record
in its subtype dataset by the configured rules, aligns each group with Nextclade, and
publishes that pull's partitions of ``sequences/<subtype>`` (:mod:`af.seq.processed`).
Alignments are pipeline steps in the work area, so re-running an unchanged pull does not
re-run Nextclade.

``store`` holds one batch marker (:meth:`af.store.Store.batch`) for the whole run, naming
every dataset its pulls can write, so a map or report never starts against a half re-stored
store. Pulls run in the order given, and the first failure stops the run.

With a ``[locations]`` section, every ``store`` also checks the pull's name-locations against
af's places table (:mod:`af.seq.newplaces`, LOCATIONS-PROPOSAL §6c): the counts go in each
published version's provenance, and the proposed rows and the review list go in the pull's work
area under ``new-locations/<dataset>/``. Without the section the provenance says the check did
not run, so a new location never passes unnoticed.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.pipeline.config import RunnerSettings, make_runner
from af.pipeline.driver import Pipeline
from af.run import Runner
from af.run.job import run_main
from af.seq import newplaces, processed
from af.seq import nextclade as nc
from af.seq.gisaid import SequenceRecord, join, read_fasta, read_workbook
from af.seq.locations import LocationTables, name_location
from af.seq.processed import PlacementRule
from af.seq.pulls import find_pulls, import_pull, open_pull
from af.store import DatasetWork, PathsConfig, Store, Work
from af.store.ref import StoreRef
from af.util.config import ConfigError, load_config

WORK_DATASET = "gisaid"
NEW_LOCATIONS = "new-locations"  # under a pull's work area: one directory per dataset
#: The batch marker a ``store`` run holds unless given another name (af.store.busy).
DEFAULT_BATCH = "seq-store"


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
class ReferenceCheckConfig:
    """One ``[[reference_check]]`` row (:class:`processed.ReferenceCheck`)."""

    dataset: str
    max_substitutions: int
    expect_lineages: list[str]
    reason: str


@dataclass(frozen=True)
class LocationsConfig:
    """What the new-locations check reads (LOCATIONS-PROPOSAL §6c)."""

    tables: Path  # acmacs-f-data rules/locations: countries.tsv, regions.tsv, places.tsv
    geonames: Path  # a GeoNames snapshot: cities500.txt, admin1CodesASCII.txt, countryInfo.txt


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
    #: Datasets kept to their reference's lineage (processed.ReferenceCheck).
    reference_check: list[ReferenceCheckConfig] = field(default_factory=list)
    #: The new-locations check; absent, the provenance records that it did not run.
    locations: LocationsConfig | None = None


def import_source(config: SequencesConfig, source: str) -> list[StoreRef]:
    if source not in config.sources:
        raise ConfigError("<config>", [f"sources: no source {source!r}"])
    store = Store.open(config.paths.store)
    return [import_pull(store, pull) for pull in find_pulls(config.sources[source], source)]


def pull_datasets(config: SequencesConfig, pull_id: str) -> list[str]:
    """Every sequences dataset this pull can publish to, for the batch marker.

    Read from the config, not from a run: a B pull writes both lineages' datasets, and the
    marker must name them before the first record is placed.
    """
    label = pull_id.rsplit("-", 1)[-1]
    if label not in config.source_subtypes:
        raise ConfigError("<config>", [f"source_subtypes: no entry for pull label {label!r}"])
    subtype = config.source_subtypes[label]
    names = {
        rule.dataset for rule in config.placement if name_terms(rule.gisaid_subtype) == subtype
    }
    for check in config.lineage_check:
        if name_terms(check.gisaid_subtype) == subtype:
            names.update(check.candidates.values())
    if not names:
        raise ConfigError("<config>", [f"placement: no dataset receives {subtype!r} records"])
    return sorted(names)


def name_terms(gisaid_subtype: str) -> str:
    """GISAID's subtype ("A / H3N2", "B") in the name normaliser's terms ("A(H3N2)", "B").

    ``source_subtypes`` is written in the normaliser's terms and placement in GISAID's, so
    comparing them as written matches only B, and every A pull would name no dataset.
    """
    flu_type, sep, rest = gisaid_subtype.partition("/")
    return f"{flu_type.strip()}({rest.strip()})" if sep else gisaid_subtype.strip()


def store_pulls(
    config: SequencesConfig,
    pull_ids: Sequence[str],
    runner: Runner,
    *,
    batch: str = DEFAULT_BATCH,
) -> dict[str, dict[str, StoreRef]]:
    """Store several pulls under one batch marker; returns each pull's refs, in order.

    One marker for the whole run, not one per pull: a sweep publishes each dataset once per
    pull, and a reader between two of those publishes would see a dataset half re-stored.
    """
    if not pull_ids:
        raise ConfigError("<command line>", ["store: no pull named"])
    if repeated := sorted({p for p in pull_ids if list(pull_ids).count(p) > 1}):
        raise ConfigError("<command line>", [f"store: pull named twice: {', '.join(repeated)}"])
    datasets = sorted({f"{processed.KIND}/{d}" for p in pull_ids for d in pull_datasets(config, p)})
    store = Store.open(config.paths.store)
    with store.batch(batch, datasets):
        return {pull_id: store_pull(config, pull_id, runner) for pull_id in pull_ids}


def read_pulls_file(path: Path) -> list[str]:
    """One pull id per line; blank lines and ``#`` comments are skipped."""
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line]


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
    held_counts = _hold_far_records(config, placed, alignments)
    new_locations = _check_new_locations(
        config.locations, placed, area.root / NEW_LOCATIONS, started
    )

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
                "reference_check": held_counts.get(dataset, {}),
                "alignment": summaries[dataset],
                "new_locations": new_locations[dataset],
            },
            started=started,
        )
    return refs


def _hold_far_records(
    config: SequencesConfig,
    placed: dict[str, list[SequenceRecord]],
    alignments: Mapping[str, Mapping[str, nc.Aligned]],
) -> dict[str, dict[str, int]]:
    """Apply every ``reference_check`` to ``placed`` in place; returns the flag counts.

    Held records are removed from the dataset and so are never published: they stay in the
    raw pull only. A dataset left with nothing is fatal rather than a silent no-op, so a
    pull of another lineage stops the run instead of looking stored.
    """
    seen = Counter(item.dataset for item in config.reference_check)
    if twice := sorted(d for d, n in seen.items() if n > 1):
        raise ConfigError("<config>", [f"reference_check: {twice} named twice"])
    counts: dict[str, dict[str, int]] = {}
    for item in config.reference_check:
        if item.dataset not in placed:
            continue
        check = processed.ReferenceCheck(
            item.dataset, item.max_substitutions, tuple(item.expect_lineages), item.reason
        )
        kept, held, flags = processed.hold_far_from_reference(
            placed[item.dataset], alignments[item.dataset], check
        )
        counts[item.dataset] = {
            "max_substitutions": item.max_substitutions,
            "held": len(held),
            **dict(sorted(flags.items())),
        }
        if not kept:
            raise processed.StoreBuildError(
                f"reference_check held every record of {item.dataset!r}"
                f" ({len(held)} over {item.max_substitutions} substitutions):"
                " exclude the pull or widen the rule, deliberately"
            )
        placed[item.dataset] = kept
    return counts


def _check_new_locations(
    locations: LocationsConfig | None,
    placed: Mapping[str, list[SequenceRecord]],
    directory: Path,
    started: datetime.datetime,
) -> dict[str, dict[str, Any]]:
    """Per dataset: this pull's name-locations that places.tsv has no row for, and what became of
    each (proposed from GeoNames, or on the review list with the reason).

    GeoNames is read only when there is something to look up. The review list is written even
    when empty, so its absence always means the check did not run.
    """
    if locations is None:
        return {d: {"checked": False, "reason": "no [locations] in the sequences config"}
                for d in placed}  # fmt: skip
    tables = LocationTables.read(locations.tables)
    geo: newplaces.GeoNames | None = None
    report: dict[str, dict[str, Any]] = {}
    for dataset, group in sorted(placed.items()):
        stated = newplaces.new_locations(
            newplaces.stated_by_location(_stated_locations(group)), tables.places.by_location
        )
        outcomes: list[newplaces.Outcome] = []
        if stated:
            geo = geo or newplaces.GeoNames.read(locations.geonames)
            outcomes = [
                newplaces.resolve(location, s, tables.countries, geo)
                for location, s in sorted(stated.items())
            ]
        target = directory / dataset
        counts = newplaces.write(outcomes, target, "af.seq.build", started.date().isoformat())
        report[dataset] = {
            "checked": True,
            "new_locations": len(outcomes),
            "outcomes": dict(sorted(counts.items())),
            "places_sha256": hashlib.sha256(
                (locations.tables / "places.tsv").read_bytes()
            ).hexdigest(),
            "geonames": str(locations.geonames),
            "review": str(target / "review.tsv"),
        }
        print(f"{NEW_LOCATIONS}\t{dataset}\t{len(outcomes)}\t{json.dumps(counts)}", file=sys.stderr)
    return report


def _stated_locations(
    records: Iterable[SequenceRecord],
) -> Iterator[tuple[str, str | None, str | None]]:
    """(name-location, GISAID country, GISAID place) for each record whose name parsed cleanly."""
    for record in records:
        if any(problem.startswith("name.") for problem in record.problems):
            continue
        location = name_location(record.name)
        if location is not None:
            _, country, place = processed.split_location(record.location)
            yield location, country, place


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
    store = commands.add_parser("store")
    store.add_argument("--pull", action="append", default=[], help="a raw pull id; repeatable")
    store.add_argument("--pulls-file", type=Path, help="pull ids, one per line")
    store.add_argument("--batch", default=DEFAULT_BATCH, help="the batch marker's name")
    args = parser.parse_args(argv)
    config = load_config(args.config, SequencesConfig)
    if args.command == "import":
        refs = import_source(config, args.source)
        _print({ref.dataset: ref.version for ref in refs})
    else:
        pull_ids = args.pull + (read_pulls_file(args.pulls_file) if args.pulls_file else [])
        runner = make_runner(config.runner, args.config)
        for refs_by in store_pulls(config, pull_ids, runner, batch=args.batch).values():
            _print({ref.dataset: ref.version for ref in refs_by.values()})
    return 0


def _print(data: Mapping[str, str]) -> None:
    for key, value in data.items():
        print(f"{key}\t{value}")


if __name__ == "__main__":
    run_main(main)  # Ctrl-C / SIGTERM stop it at once
