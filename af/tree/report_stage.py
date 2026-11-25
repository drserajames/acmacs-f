"""The report tree as its own dataset, derived from a published tree version without rebuilding.

    python -m af.tree.report_stage <store> <source dataset@version> <export dir> <purpose>
        --locations <acmacs-f-data>/rules/locations --placement-limits <.../trees/placement.tsv>
        [--trim <acmacs-f-data>/trees/trim.tsv]

The report tree is derived from a *published* tree version, named by its id, never by CURRENT, so
a round's report tree says exactly which weekly tree it came from, and the result is published as
``trees/<subtype>/<purpose>``. It holds every leaf of its source unless ``--trim`` cuts it at a node
(af.tree.report_filter). There is **no date rule and no titre table** (Sarah, 25 Nov 2026: a report
tree must not drop sequences that were not titrated, and "the tables shoud not affect the trees"):
the shipped pipeline's ``--remove-leaves-isolated-before 2021 --important <titrated>`` is not
reproduced, no tables dataset is read, and none is an input of the published version.

- The leaves' sequences come from the export the source was built from; the export must be the
  one the source records (same sequence-store version, same leaf count), and its files the ones it
  wrote (af.tree.export.check_matches).
- Continent comes from acmacs-f-data's location tables (country spelling -> ISO3 -> scheme
  ``continent``). A leaf with none is kept and counted by reason, never guessed.
- Clock direction, placement and the clock's own leaf flags are re-measured on the pruned tree
  and refused as in the build: pruning keeps every kept leaf's path, but spliced branches carry
  their net change, so the numbers are measured again rather than inherited. The thresholds the
  flags use are the ones the SOURCE version recorded for its populate step, read back from its
  provenance, so a report tree flags on its source's rule and never on one typed in here.
- ``--trim`` applies the shipped 223V / 127T cut (af.tree.trim), found on the source tree. The
  trimmed tree has no outgroup, and placement is measured on the path to the outgroup, so it is
  measured on the *untrimmed* report tree and recorded as such. So are the clock's leaf flags,
  carried onto the leaves the trim keeps: one fit, one meaning of off-clock (00, 24 Oct). Refitted
  on the cut branch alone, the clock flagged 1,004 B/Vic leaves instead of 286, nearly all of them
  the current fast-moving clade, an artefact of the cut. The clock direction is checked on the
  trimmed tree. A trim file with no row for the subtype is an error. Without ``--trim`` nothing
  changes. Which node is cut is that file's row for the subtype: config, never code.
- The source version is an input in PROVENANCE.json and in HISTORY.jsonl's publish event; the
  store's ``parent`` field is the previous version of this same dataset.
"""

from __future__ import annotations

import argparse
import datetime
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from af.seq.locations import LocationTables
from af.seq.places import PlacesError
from af.store import Provenance, Store, StoreRef
from af.store.ref import ExternalInput
from af.store.store import MANIFEST, Manifest
from af.tree.clock import (
    ClockSettings,
    InvertedClockError,
    apply_flags,
    check_clock_direction,
    find_outliers,
)
from af.tree.export import check_matches, read_export
from af.tree.io import i6
from af.tree.io.fasta import read_alignment
from af.tree.placement import PlacementError, check_placement, limit_for, load_placement_limits
from af.tree.populate import CONTINENTS, LeafRecord, PopulatedTree
from af.tree.report_filter import report_tree
from af.tree.stages import build_record
from af.tree.trim import TrimError, TrimMatch, find_trim_node, load_trim_rules, rule_for

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


def source_clock_settings(version: Path) -> ClockSettings:
    """The clock thresholds the source version recorded for its own populate step.

    The flags on a report tree have to mean what the weekly tree's flags mean, so the thresholds
    are not defaulted here and not passed on the command line: they are read back from the step
    record the source wrote (``<file>.provenance.json``, step ``<subtype>.populate``), where
    af.tree.stages records ``dataclasses.asdict(clock)``. A source that recorded none is refused,
    because the alternative is to invent a threshold and call the result a measurement.
    """
    records = sorted(version.glob("*.provenance.json"))
    for record in records:
        try:
            written = json.loads(record.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ReportStageError(f"{record}: unreadable step record ({error})") from error
        if not str(written.get("step", "")).endswith(".populate"):
            continue
        clock = (written.get("parameters") or {}).get("clock")
        if not isinstance(clock, dict):
            continue
        try:
            return ClockSettings(
                z_threshold=float(clock["z_threshold"]),
                branch_threshold=(
                    None
                    if clock.get("branch_threshold") is None
                    else float(clock["branch_threshold"])
                ),
                earliest=datetime.date.fromisoformat(str(clock["earliest"])),
                min_leaves=int(clock["min_leaves"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ReportStageError(
                f"{record}: its populate step records clock settings this code cannot read "
                f"({error}); the report tree would have to invent the thresholds it flags on"
            ) from error
    raise ReportStageError(
        f"{version}: no populate step record carries clock settings "
        f"(looked in {len(records)} *.provenance.json), so there is no recorded threshold to "
        f"flag the report tree's leaves on"
    )


def derive(
    populated: PopulatedTree,
    continent_of: Callable[[LeafRecord], str | None],
    missing: Counter[str],
    placement_limits: Path,
    clock: ClockSettings,
    trim: TrimMatch | None = None,
) -> PopulatedTree:
    """The report tree, re-checked. Refuses as the build would."""
    outgroup = populated.counts.get("outgroup")
    if not outgroup:
        raise ReportStageError("the source tree records no outgroup")
    # Placement needs the outgroup, so with a trim it is measured on the untrimmed report tree.
    # Continents are not looked up there: the lookup counts its misses, and this tree is not kept.
    untrimmed = (
        report_tree(populated)
        if trim is not None
        else report_tree(populated, continent_of=continent_of)
    )
    result = (
        untrimmed if trim is None else report_tree(populated, continent_of=continent_of, trim=trim)
    )
    for reason, count in sorted(missing.items()):
        result.counts[f"continent_unassigned_{reason}"] = count
    result.counts.pop("continent_not_assigned", None)
    try:
        result.counts["clock_direction"] = check_clock_direction(result).to_json()
    except InvertedClockError as error:
        raise ReportStageError(str(error)) from error
    limit = limit_for(load_placement_limits(placement_limits), result.subtype)
    try:
        measured = check_placement(untrimmed, outgroup, limit)
    except PlacementError as error:
        raise ReportStageError(str(error)) from error
    result.counts["placement"] = {**measured.to_json(), "limit": limit.to_json()}
    if trim is not None:
        result.counts["placement"]["measured_on"] = "the untrimmed report tree (no outgroup after)"
    # With a trim, the flags come from the untrimmed tree's fit (module docstring); apply_flags
    # matches by leaf key, so the flags of leaves outside the cut are dropped.
    outliers = find_outliers(untrimmed, clock)
    apply_flags(result, outliers)
    if trim is not None:
        kept = {leaf.name for leaf in result.tree.leaves()}
        result.counts["clock_flags_measured_on"] = "the untrimmed report tree"
        result.counts["clock_flags_on_kept_leaves"] = dict(
            sorted(
                Counter(
                    reason
                    for key, reasons in outliers.flags.items()
                    if key in kept
                    for reason in reasons
                ).items()
            )
        )
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
    parser.add_argument(
        "--trim",
        type=Path,
        help="acmacs-f-data trees/trim.tsv: cut to the subtype's one 223V / 127T branch "
        "(af.tree.trim); a file with no row for the subtype is an error",
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

    continent_of, missing = continent_lookup(LocationTables.read(args.locations))
    trim: TrimMatch | None = None
    if args.trim is not None:
        try:
            trim = find_trim_node(
                populated, rule_for(load_trim_rules(args.trim), populated.subtype)
            )
        except TrimError as error:
            raise ReportStageError(str(error)) from error
    result = derive(
        populated,
        continent_of,
        missing,
        args.placement_limits,
        source_clock_settings(version),
        trim,
    )
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
        # Said in the record, because report trees made before 25 Nov 2026 carry a date rule.
        "date_rule": "none: no leaf is dropped for its date or for having no titres; "
        "only a node cut (trim) makes the tree smaller (Sarah, 25 Nov 2026)",
        "continent": f"{args.locations} scheme {CONTINENT_SCHEME}, country source {COUNTRY_SOURCE}",
    }
    if trim is not None:
        parameters["trim"] = {"file": str(args.trim), **trim.to_json()}
        extra_inputs.append(ExternalInput.of(args.trim))
    leaves, internal = result.tree.count()
    report = {k: result.counts.get(k) for k in sorted(result.counts)}
    print(
        f"{source_ref}: {populated.tree.count()[0]} leaves -> {leaves} leaves, {internal} internal"
    )
    for key in (
        "clock_flagged",
        "clock_long_branches",
        "clock_clock_outliers",
        "clock_impossible_date",
        "clock_branch_threshold",
        "clock_z_threshold",
        *(
            ()
            if trim is None
            else (
                "trim",
                "removed_by_trim",
                "removed_by_trim_by_year",
            )
        ),
    ):
        print(f"  {key:<28} {report.get(key)}")
    print(f"  continents                   {result.counts.get('continents')}")
    for key, value in report.items():
        if key.startswith("continent_unassigned_"):
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
        ],
        parameters,
        started,
    )
    print(f"published {ref} (manifest_sha256 {ref.manifest_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
