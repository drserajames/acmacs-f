"""The report tree as its own dataset, derived from a published tree version without rebuilding.

    python -m af.tree.report_stage <store> <source dataset@version> <export dir> <purpose>
        --cutoff 2021-01-01 --tables-through 2026-09-21 --table-subtype "A(H1N1)"
        --locations <acmacs-f-data>/rules/locations --placement-limits <.../trees/placement.tsv>
        [--table-lineage VICTORIA]

Today's pipeline makes the report tree from the weekly tree in its W8 ``pdf`` step: leaves
collected before 2021 are dropped unless a WHO CC table holds them (af.tree.report_filter, task
5.5). This step does the same to a *published* version, named by its id, never by CURRENT, so a
round's report tree says exactly which weekly tree it came from, and publishes the result as
``trees/<subtype>/<purpose>``.

- The leaves' sequences come from the export the source was built from; the export must be the
  one the source records (same sequence-store version, same leaf count), and its files the ones it
  wrote (af.tree.export.check_matches).
- Titrated means held by a table in the tables store dated on or before ``--tables-through`` (the
  round's tables, not later ones), matched by the lab's EPI_ISL or by location/isolate/year
  (af.seq.matching's key, the one editable copy).
- Continent comes from acmacs-f-data's location tables (country spelling -> ISO3 -> scheme
  ``continent``). A leaf with none is kept and counted by reason, never guessed.
- Clock direction and placement are re-measured on the pruned tree and refused as in the build:
  pruning keeps every kept leaf's path, but spliced branches carry their net change, so the
  numbers are measured again rather than inherited.
- The source version is an input in PROVENANCE.json and in HISTORY.jsonl's publish event; the
  store's ``parent`` field is the previous version of this same dataset.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from af.seq.locations import LocationTables
from af.seq.matching import location_forms_key, name_key
from af.seq.places import PlacesError
from af.store import Provenance, Store, StoreRef
from af.store.ref import ExternalInput
from af.store.store import MANIFEST, Manifest
from af.tables.model import Table
from af.tables.store import current_tables, read_table
from af.tree.clock import InvertedClockError, check_clock_direction
from af.tree.export import check_matches, read_export
from af.tree.io import i6
from af.tree.io.fasta import read_alignment
from af.tree.placement import PlacementError, check_placement, limit_for, load_placement_limits
from af.tree.populate import CONTINENTS, LeafRecord, PopulatedTree
from af.tree.report_filter import TitratedIndex, report_tree
from af.tree.stages import build_record

KIND = "trees"
STEP = "trees.report"
CONTINENT_SCHEME = "continent"
COUNTRY_SOURCE = "gisaid"  # leaf countries are GISAID's spellings (af.seq store)


class ReportStageError(ValueError):
    """The source, its export or the derived tree is not usable."""


def pinned(store: Store, kind: str, spec: str) -> StoreRef:
    """``dataset@version`` as a verified ref. A bare dataset (CURRENT) is refused."""
    dataset, at, version = spec.partition("@")
    if not at or not version:
        raise ReportStageError(f"{spec!r}: name the version (dataset@version), never CURRENT")
    manifest = store.dataset_dir(kind, dataset) / "versions" / version / MANIFEST
    if not manifest.is_file():
        raise ReportStageError(f"{kind}/{dataset}@{version}: no such version")
    ref = StoreRef(kind, dataset, version, Manifest.from_bytes(manifest.read_bytes()).sha256())
    store.verify(ref)
    return ref


def report_name_key(name: str) -> str:
    """location/isolate/year with the location's spacing and punctuation dropped.

    A name of another shape keys as itself, so it can only match the identical name.
    """
    key = name_key(name)
    return f"unparsed:{name}" if key is None else "/".join(location_forms_key(key))


def round_tables(
    tables: Iterable[Table], subtype: str, lineage: str, through: datetime.date
) -> tuple[list[Table], dict[str, int]]:
    """One subtype's tables dated on or before ``through``; B antigens of other lineages out."""
    kept: list[Table] = []
    counts: Counter[str] = Counter()
    for table in tables:
        if table.subtype != subtype:
            continue
        if datetime.date.fromisoformat(table.date) > through:
            counts["tables_after_through"] += 1
            continue
        if lineage:
            antigens = [a for a in table.antigens if (a.lineage or table.lineage) == lineage]
            counts["antigens_other_lineage"] += len(table.antigens) - len(antigens)
            table = dataclasses.replace(table, antigens=antigens)
        kept.append(table)
    counts["tables_used"] = len(kept)
    if not kept:
        raise ReportStageError(f"no {subtype} {lineage} tables dated on or before {through}")
    return kept, dict(counts)


def continent_lookup(
    tables: LocationTables,
) -> tuple[Callable[[LeafRecord], str | None], Counter[str]]:
    """country -> continent, and the counter its misses are recorded in (by reason)."""
    missing: Counter[str] = Counter()

    def continent_of(record: LeafRecord) -> str | None:
        if not record.country:
            missing["no_country"] += 1
            return None
        try:
            code = tables.countries.code(record.country, COUNTRY_SOURCE)
        except PlacesError:
            missing["unknown_country_spelling"] += 1
            return None
        group = tables.regions.assigned(CONTINENT_SCHEME, code)
        if group is None or group not in CONTINENTS:
            missing["no_continent_for_country"] += 1
            return None
        return group

    return continent_of, missing


def checked_alignment(meta: dict[str, Any], export_dir: Path) -> tuple[dict[str, str], Path]:
    """The export's alignment, after checking it is the export the source tree was built from."""
    record = read_export(export_dir)
    check_matches(record, export_dir / "alignment.fasta", export_dir / "leaves.parquet")
    source = meta.get("source")
    if source is None:
        raise ReportStageError(
            "the source tree records no export, so its leaves' sequences are unknown"
        )
    if source["sequences"] != record.sequences.to_json() or source["leaves"] != record.leaves:
        raise ReportStageError(
            f"{export_dir}: not the source's export (sequences {record.sequences.version}, "
            f"{record.leaves} leaves; the tree records {source['sequences']['version']}, "
            f"{source['leaves']})"
        )
    return read_alignment(export_dir / "alignment.fasta"), export_dir / "alignment.fasta"


def derive(
    populated: PopulatedTree,
    cutoff: datetime.date,
    titrated: TitratedIndex,
    continent_of: Callable[[LeafRecord], str | None],
    missing: Counter[str],
    placement_limits: Path,
) -> PopulatedTree:
    """The report tree, re-checked. Refuses as the build would."""
    outgroup = populated.counts.get("outgroup")
    if not outgroup:
        raise ReportStageError("the source tree records no outgroup")
    result = report_tree(populated, cutoff, titrated, continent_of=continent_of)
    for reason, count in sorted(missing.items()):
        result.counts[f"continent_unassigned_{reason}"] = count
    result.counts.pop("continent_not_assigned", None)
    try:
        result.counts["clock_direction"] = check_clock_direction(result).to_json()
    except InvertedClockError as error:
        raise ReportStageError(str(error)) from error
    limit = limit_for(load_placement_limits(placement_limits), result.subtype)
    try:
        measured = check_placement(result, outgroup, limit)
    except PlacementError as error:
        raise ReportStageError(str(error)) from error
    result.counts["placement"] = {**measured.to_json(), "limit": limit.to_json()}
    return result


def publish(
    store: Store,
    result: PopulatedTree,
    purpose: str,
    source_ref: StoreRef,
    source_meta: dict[str, Any],
    inputs: Sequence[StoreRef | ExternalInput],
    parameters: dict[str, Any],
    started: datetime.datetime,
) -> StoreRef:
    if purpose == "weekly" or purpose == source_ref.dataset.split("/", 1)[1]:
        raise ReportStageError(f"purpose {purpose!r}: a report tree never replaces its source")
    source = {**(source_meta.get("source") or {}), "derived_from": source_ref.to_json()}
    with store.build(KIND, f"{result.subtype}/{purpose}") as builder:
        i6.write(result, builder.path, purpose, source)
        leaves, internal = result.tree.count()
        summary = {
            "leaves": leaves,
            "internal_nodes": internal,
            "branch_scale": result.branch_scale,
            "derived_from": source_ref.to_json(),
        }
        provenance = Provenance(
            step=STEP,
            inputs=(source_ref, *inputs),
            parameters=parameters,
            started=started,
            finished=datetime.datetime.now(datetime.UTC),
        )
        return builder.publish(provenance, summary=summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m af.tree.report_stage", description=__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument(
        "source", help="the tree to derive from, dataset@version (e.g. h1/weekly@...)"
    )
    parser.add_argument("export", type=Path, help="the export the source was built from")
    parser.add_argument("purpose", help="the new dataset's purpose, e.g. report-round-af-2026-0921")
    parser.add_argument("--cutoff", required=True, type=datetime.date.fromisoformat)
    parser.add_argument("--tables-through", required=True, type=datetime.date.fromisoformat)
    parser.add_argument(
        "--table-subtype", required=True, help='Table.subtype: "A(H1N1)", "A(H3N2)", "B"'
    )
    parser.add_argument("--table-lineage", default="", help="B only: VICTORIA or YAMAGATA")
    parser.add_argument(
        "--locations", required=True, type=Path, help="acmacs-f-data rules/locations"
    )
    parser.add_argument("--placement-limits", required=True, type=Path)
    parser.add_argument(
        "--build-record",
        type=Path,
        help="the source's build.json, for a source published before tree.json recorded its build;"
        " fills tree.json 'build' and is recorded as a hashed input",
    )
    parser.add_argument("--dry-run", action="store_true", help="derive and check; do not publish")
    args = parser.parse_args(argv)

    started = datetime.datetime.now(datetime.UTC)
    store = Store.open(args.store)
    source_ref = pinned(store, KIND, args.source)
    version = store.version_dir(source_ref)
    meta = i6.read_metadata(version)
    alignment, alignment_path = checked_alignment(meta, args.export)
    populated = i6.read(version, alignment)

    held = current_tables(store)
    tables, table_counts = round_tables(
        (read_table(path) for _, path in held.values()),
        args.table_subtype,
        args.table_lineage,
        args.tables_through,
    )
    table_refs = sorted({ref for ref, _ in held.values()}, key=lambda r: r.dataset)
    titrated = TitratedIndex.from_tables(tables, report_name_key)
    continent_of, missing = continent_lookup(LocationTables.read(args.locations))
    result = derive(populated, args.cutoff, titrated, continent_of, missing, args.placement_limits)
    result.counts.update({f"report_{k}": v for k, v in table_counts.items()})
    extra_inputs: list[StoreRef | ExternalInput] = []
    if args.build_record is not None:
        if populated.build is not None:
            raise ReportStageError(
                f"{source_ref} already records its build; --build-record would replace a fact"
            )
        result.build = build_record(json.loads(args.build_record.read_text()))
        extra_inputs.append(ExternalInput.of(args.build_record))
    print(f"  build                        {result.build}")

    parameters = {
        "source": source_ref.to_json(),
        "cutoff": args.cutoff.isoformat(),
        "tables_through": args.tables_through.isoformat(),
        "table_subtype": args.table_subtype,
        "table_lineage": args.table_lineage,
        "name_key": "af.seq.matching location/isolate/year, location forms",
        "continent": f"{args.locations} scheme {CONTINENT_SCHEME}, country source {COUNTRY_SOURCE}",
    }
    leaves, internal = result.tree.count()
    report = {
        k: result.counts.get(k)
        for k in sorted(result.counts)
        if not k.startswith("titrated_leaves_")
    }
    print(
        f"{source_ref}: {populated.tree.count()[0]} leaves -> {leaves} leaves, {internal} internal"
    )
    for key in (
        "removed_before_cutoff",
        "kept_titrated_before_cutoff",
        "kept_undated",
        "titrated_leaves",
        "titrated_by_epi_isl",
        "titrated_by_name",
        "titre_index_tables",
        "titre_index_antigens",
    ):
        print(f"  {key:<28} {report.get(key)}")
    print(f"  continents                   {result.counts.get('continents')}")
    for key, value in report.items():
        if key.startswith("continent_unassigned_") or key.startswith("report_"):
            print(f"  {key:<28} {value}")
    print(f"  clock_direction              {result.counts['clock_direction']}")
    placement = result.counts["placement"]
    correlation, limit = placement.get("correlation"), placement["limit"]
    print(f"  placement correlation        {correlation} (limit {limit})")
    if args.dry_run:
        print("dry run: not published")
        return 0
    ref = publish(
        store,
        result,
        args.purpose,
        source_ref,
        meta,
        [
            ExternalInput.of(alignment_path),
            ExternalInput.of(args.locations / "countries.tsv"),
            ExternalInput.of(args.locations / "regions.tsv"),
            *extra_inputs,
            *table_refs,
        ],
        parameters,
        started,
    )
    print(f"published {ref} (manifest_sha256 {ref.manifest_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
