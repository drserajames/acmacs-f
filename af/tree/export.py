"""Export: a tree's input files from the sequence store, by workstream 2's rules (task 5.1).

Which sequences go into a tree is decided by :func:`af.seq.select.select` alone (workstream 2
owns the rules and their config); this module only writes what the build and populate stages
read, and says where it came from. One directory per export::

    alignment.fasta    nuc_aligned per leaf key (EPI_ISL|accession), outgroup first
    leaves.parquet     leaf records as af.tree.stages.write_leaves writes them, embargo included
    export.json        the sequence-store version, every rule's count, the file hashes

**Why the store version is written down.** The clades step labels the sequences that are not on
the tree from the *same* sequence version (workstream 4's ``publish_from_tree(sequences=…)``),
and refuses a tree whose leaves that version does not hold. So the version must be the one the
tree was built from, not whatever is CURRENT when the clades step runs; ``export.json`` carries
it, and the build stage checks the alignment it is given is the one exported with it.

**Embargo.** Embargoed sequences may be used in WHO reports, not in publications. Nothing here
drops them; the flag travels with each leaf into the tree store, and the count is in
``export.json``, so every consumer can tell.

**Test-only exports.** Until the outgroup a subtype's rules pin is in the store, ``select``
refuses to run (rightly: a tree cannot be rooted without it). A stand-in outgroup lets the rest
of the chain be exercised on real data, but the result is labelled ``test_only`` with its reason
here, in the tree's build record, and in the tree store, and the publish stage refuses it for
any purpose but a ``test…`` one.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from af.seq.select import Key, Outgroup, Selection, SubtypeRules, parse_rules, select
from af.store import Store, StoreRef
from af.tree.io.fasta import write_alignment
from af.tree.populate import LeafRecord, leaf_key
from af.util.artefacts import sha256_path

ALIGNMENT_FILE = "alignment.fasta"
LEAVES_FILE = "leaves.parquet"
EXPORT_FILE = "export.json"
FORMAT = 1


class ExportError(RuntimeError):
    """The selection cannot be written as a tree's input."""


@dataclass(frozen=True)
class TestOnly:
    """A stand-in outgroup, and why. Never for a report."""

    __test__ = False  # not a pytest class, despite the name

    outgroup: Outgroup
    reason: str


@dataclass(frozen=True)
class ExportRecord:
    """``export.json``, read back: what a tree built from this export must carry."""

    subtype: str
    sequences: StoreRef
    outgroup: str  # leaf key
    leaves: int
    embargoed: int
    alignment_sha256: str
    leaves_sha256: str
    test_only: str | None  # the reason, or None for a real export
    counts: list[dict[str, Any]]


def export(
    store: Store,
    subtype: str,
    rules: SubtypeRules,
    directory: Path,
    *,
    test_only: TestOnly | None = None,
) -> ExportRecord:
    """Select, then write the three files into ``directory`` (created; must be empty)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ExportError(f"{directory}: not empty; an export never mixes with another")
    configured = rules.outgroup
    if test_only is not None:
        if not test_only.reason.strip():
            raise ExportError("a test-only export needs its reason")
        rules = dataclasses.replace(rules, outgroup=test_only.outgroup)
    selection = select(store, rules)
    sequences, leaves = _load(store, selection)

    from af.tree.stages import write_leaves  # stages imports this module's reader

    alignment_path = directory / ALIGNMENT_FILE
    leaves_path = directory / LEAVES_FILE
    write_alignment(alignment_path, sequences)
    write_leaves(leaves, leaves_path)
    record = ExportRecord(
        subtype=subtype,
        sequences=selection.store,
        outgroup=leaf_key(*selection.outgroup),
        leaves=len(sequences),
        embargoed=sum(1 for leaf in leaves.values() if leaf.embargoed),
        alignment_sha256=sha256_path(alignment_path),
        leaves_sha256=sha256_path(leaves_path),
        test_only=None if test_only is None else test_only.reason,
        counts=[dataclasses.asdict(count) for count in selection.counts],
    )
    meta = _to_json(record, selection)
    if test_only is not None:
        meta["test_only"] = {
            "reason": test_only.reason,
            "stand_in_outgroup": leaf_key(*test_only.outgroup.key),
            "configured_outgroup": leaf_key(*configured.key),
        }
    (directory / EXPORT_FILE).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return record


def read_export(path: Path) -> ExportRecord:
    """``export.json`` (or the export directory holding it)."""
    path = Path(path)
    if path.is_dir():
        path = path / EXPORT_FILE
    data = json.loads(path.read_text())
    if data.get("format") != FORMAT:
        raise ExportError(f"{path}: format {data.get('format')!r}, this reader expects {FORMAT}")
    test_only = data.get("test_only")
    return ExportRecord(
        subtype=data["subtype"],
        sequences=StoreRef.from_json(data["sequences"]),
        outgroup=data["outgroup"],
        leaves=data["leaves"],
        embargoed=data["embargoed"],
        alignment_sha256=data["files"][ALIGNMENT_FILE],
        leaves_sha256=data["files"][LEAVES_FILE],
        test_only=None if test_only is None else test_only["reason"],
        counts=data["counts"],
    )


def check_matches(record: ExportRecord, alignment: Path, leaves: Path) -> None:
    """The files a build is given are the ones this export wrote, byte for byte.

    Otherwise the sequence version it records would describe some other set of leaves, and the
    clades step would label the wrong sequences as "not on the tree".
    """
    for name, path, expected in (
        (ALIGNMENT_FILE, alignment, record.alignment_sha256),
        (LEAVES_FILE, leaves, record.leaves_sha256),
    ):
        actual = sha256_path(Path(path))
        if actual != expected:
            raise ExportError(
                f"{path}: sha256 {actual[:12]} is not the {name} the export recorded "
                f"({expected[:12]}); give the export's own files"
            )


def _to_json(record: ExportRecord, selection: Selection) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "subtype": record.subtype,
        "sequences": record.sequences.to_json(),
        "store_records": selection.total,
        "outgroup": record.outgroup,
        "leaves": record.leaves,
        "embargoed": record.embargoed,
        "counts": record.counts,
        "files": {ALIGNMENT_FILE: record.alignment_sha256, LEAVES_FILE: record.leaves_sha256},
    }


def _load(store: Store, selection: Selection) -> tuple[dict[str, str], dict[str, LeafRecord]]:
    """Aligned sequences and leaf records for exactly the selected keys, in selection order."""
    version = store.resolve(selection.store)
    rows = duckdb.execute(
        "select i.epi_isl, i.accession, i.name, i.date_precision, i.collection_date_first,"
        " i.collection_date_last, i.country, i.region, i.embargoed, s.nuc_aligned"
        " from read_parquet(?) i join read_parquet(?) s using (epi_isl, accession)",
        [str(version / "isolates" / "*" / "*.parquet"),
         str(version / "sequences" / "*" / "*.parquet")],
    ).fetchall()  # fmt: skip
    by_key: dict[Key, tuple[Any, ...]] = {(row[0], row[1]): row for row in rows}
    sequences: dict[str, str] = {}
    leaves: dict[str, LeafRecord] = {}
    for key in selection.keys:
        row = by_key.get(key)
        if row is None:  # select read the same version, so this is a store fault
            raise ExportError(f"{selection.store}: selected {key} is not in the store version")
        epi, accession, name, precision, first, last, country, region, embargoed, nuc = row
        if not nuc:
            raise ExportError(f"{selection.store}: selected {key} has no aligned sequence")
        record = LeafRecord(
            epi_isl=epi,
            accession=accession,
            name=name,
            nucleotides=nuc,
            # I6 "date": the interval's first day, read with its precision; a year-only date is
            # 2021-01-01 *with* precision "year", never a 1 January measurement.
            collection_date=first,
            date_precision=precision,
            collection_date_first=first,
            collection_date_last=last,
            country=country or None,
            region=region or None,
            embargoed=bool(embargoed),
        )
        sequences[record.key] = nuc
        leaves[record.key] = record
    lengths = {len(sequence) for sequence in sequences.values()}
    if len(lengths) != 1:
        raise ExportError(
            f"{selection.store}: aligned lengths differ ({sorted(lengths)[:5]}); "
            "a tree alignment must be rectangular"
        )
    return sequences, leaves


# ---------------------------------------------------------------------------------------------
# Command line


def load_rules(config: Path, subtype: str) -> SubtypeRules:
    """One subtype's table of a selection config (acmacs-f-data ``config/selection.toml``)."""
    data = tomllib.loads(Path(config).read_text())
    if subtype not in data:
        raise ExportError(f"{config}: no [{subtype}] table; has {sorted(data)}")
    return parse_rules(data[subtype], Path(config).parent)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m af.tree.export", description=__doc__)
    parser.add_argument("selection", type=Path, help="selection config (config/selection.toml)")
    parser.add_argument("subtype", help="its table: h3, h1, bvic, byam")
    parser.add_argument("store", type=Path, help="the store root")
    parser.add_argument("out", type=Path, help="export directory (created; must be empty)")
    parser.add_argument(
        "--test-only-outgroup",
        metavar="EPI_ISL|ACCESSION",
        help="root on this stand-in, not the configured outgroup; labels the export test-only",
    )
    parser.add_argument("--test-only-reason", default="", help="required with the stand-in")
    args = parser.parse_args(argv)

    test_only = None
    if args.test_only_outgroup:
        epi, _, accession = args.test_only_outgroup.partition("|")
        if not accession:
            parser.error("--test-only-outgroup is EPI_ISL|ACCESSION")
        test_only = TestOnly(Outgroup(epi, accession, "stand-in"), args.test_only_reason)
    started = datetime.datetime.now(datetime.UTC)
    record = export(
        Store.open(args.store),
        args.subtype,
        load_rules(args.selection, args.subtype),
        args.out,
        test_only=test_only,
    )
    seconds = (datetime.datetime.now(datetime.UTC) - started).total_seconds()
    for count in record.counts:
        change = f"+{count['added']}" if count["added"] else f"-{count['removed']}"
        print(f"  {count['rule']:<24} {change:>9}  -> {count['remaining']}")
    label = f"  TEST-ONLY: {record.test_only}" if record.test_only else ""
    print(
        f"{record.subtype}: {record.leaves} leaves (outgroup {record.outgroup}), "
        f"{record.embargoed} embargoed, from {record.sequences}, {seconds:.1f} s{label}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
