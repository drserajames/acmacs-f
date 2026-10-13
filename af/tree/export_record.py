"""A record tying a tree's export files to the sequences version they came from.

A tree's provenance names the export it was built from by file hash, not by sequences version:
``alignment.fasta`` and ``leaves.parquet`` with their sha256. A report that walks from a figure to
a tree therefore cannot tell whether the tree read the same content as the sequences version the
figure itself reads -- the tree cites the version it was exported from, which may be an earlier one.
Comparing versions by label then refuses figures that are in fact built on identical content.

This module writes the missing fact: **(sequences version, export) -> the two file hashes**, so a
report check can say "the tree's recorded alignment hash is what exporting the version the figures
read produces" (same content) or refuse naming both hashes (content behind).

Layout, agreed with workstream 11 (11-reports-v2, 11 Oct 2026), who reads these:

    <root>/exports/<sequences dataset>/<alignment sha256, first 16 hex>/
        <sequences version>.export.json

Keying the directory by the **alignment hash** is what makes the reader's job one direct read: a
tree records that hash, so the reader goes straight to the directory and looks for the version the
figures read. Listing the directory answers the other question -- which sequences versions produce
this same alignment. The tree's own provenance cannot help here: its ``parameters`` are only
``{purpose, subtype}``, with no export parameters and no key to match on.

The record embeds ``export.json`` **verbatim** rather than restating its fields, so there is one
editable copy of each fact (design rule 6) and the two cannot drift. ``files`` is lifted out
because the reader compares it directly against the tree's external inputs.

``root`` is always passed in. It is deliberately not defaulted here: the records live outside the
store, at a site-specific path, and this package reads no config (design rule 10).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from af.store import Store
from af.util.artefacts import sha256_path

RECORDS_DIR = "exports"
RECORD_SUFFIX = ".export.json"
HASH_DIR_LENGTH = 16
SHA256 = re.compile(r"^[0-9a-f]{64}$")
AT_EXPORT = "at export"
BACK_FILLED = "back-filled"
HOW = (AT_EXPORT, BACK_FILLED)


class RecordError(RuntimeError):
    """An export record cannot be written, or disagrees with one already written."""


def _checked(export_json: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    """(dataset, version, files) out of an ``export.json``, refusing anything unusable."""
    sequences = export_json.get("sequences")
    if not isinstance(sequences, dict):
        raise RecordError("export.json has no 'sequences' StoreRef; cannot key a record by version")
    dataset, version = sequences.get("dataset"), sequences.get("version")
    if not dataset or not version:
        raise RecordError(f"export.json's sequences ref lacks dataset or version: {sequences}")
    files = export_json.get("files")
    if not isinstance(files, dict) or not files:
        raise RecordError("export.json has no 'files' hashes; the record would vouch for nothing")
    bad = {name: value for name, value in files.items() if not SHA256.match(str(value))}
    if bad:
        raise RecordError(f"export.json's files are not all sha256: {bad}")
    return str(dataset), str(version), {str(k): str(v) for k, v in files.items()}


def record_path(
    root: Path, export_json: dict[str, Any], *, alignment: str = "alignment.fasta"
) -> Path:
    """Where this export's record belongs. Derived from the export, never passed separately."""
    dataset, version, files = _checked(export_json)
    if alignment not in files:
        raise RecordError(f"export.json's files has no {alignment!r}: {sorted(files)}")
    return (
        Path(root)
        / RECORDS_DIR
        / dataset
        / files[alignment][:HASH_DIR_LENGTH]
        / f"{version}{RECORD_SUFFIX}"
    )


def content_hash(path: Path) -> str:
    """A hash of what a file *means*, stable across writer versions.

    **Why this exists.** ``leaves.parquet`` is not byte-reproducible: Parquet embeds the writer's
    version, so the same selection written by pyarrow 25.0.1 and 26.0.0 gives different sha256 over
    identical data (measured on bvic@dbe4b3bd: file hashes ``75414a16…`` and ``e7255143…``, schemas
    and tables equal). A check that only compared file bytes would refuse same-content pairs
    whenever the record was written with a different pyarrow from the build -- the very false
    refusal the record is meant to remove. ``alignment.fasta`` is text and needs no such treatment,
    so for anything but parquet this is the file hash.
    """
    path = Path(path)
    if path.suffix != ".parquet":
        return sha256_path(path)
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    digest = hashlib.sha256()
    digest.update("\0".join(table.column_names).encode() + b"\n")
    for batch in table.to_batches(max_chunksize=10_000):
        for row in batch.to_pylist():
            digest.update(json.dumps(row, sort_keys=True, default=str).encode() + b"\n")
    return digest.hexdigest()


def _writer() -> dict[str, str]:
    """The versions that decide the file bytes, so a byte mismatch is explainable."""
    import pyarrow

    return {"pyarrow": pyarrow.__version__}


def build_record(
    export_json: dict[str, Any],
    *,
    how: str,
    selection_sha256: str | None = None,
    af_release: str | None = None,
    written: datetime.datetime | None = None,
    directory: Path | None = None,
) -> dict[str, Any]:
    """The record's content. ``written`` carries a zone; a naive datetime is refused.

    With ``directory`` (the export's files, still on disk) the record also carries ``content``:
    writer-independent hashes, which are what lets a reader prove same-content when the file bytes
    differ only because of the writer's version. Without it, ``content`` is absent rather than
    guessed, and a reader falls back to byte comparison alone.
    """
    if how not in HOW:
        raise RecordError(f"how must be one of {HOW}, not {how!r}")
    _, _, files = _checked(export_json)
    when = written or datetime.datetime.now(datetime.UTC)
    if when.tzinfo is None:
        raise RecordError("'written' needs a time zone; a bare local time is not a record")
    record: dict[str, Any] = {
        "export": export_json,  # verbatim: one editable copy of each fact
        "files": files,
        "selection_sha256": selection_sha256,
        "af_release": af_release,
        "written": when.isoformat(),
        "how": how,
        "writer": _writer(),
    }
    if directory is not None:
        missing = [name for name in files if not (Path(directory) / name).is_file()]
        if missing:
            raise RecordError(f"{directory}: export files missing, cannot hash content: {missing}")
        record["content"] = {name: content_hash(Path(directory) / name) for name in sorted(files)}
    return record


def write_record(
    root: Path,
    export_json: dict[str, Any],
    *,
    how: str,
    selection_sha256: str | None = None,
    af_release: str | None = None,
    directory: Path | None = None,
) -> Path:
    """Write the record for one export. Idempotent; refuses to change an existing one silently.

    A record already there with the same ``export`` and ``files`` is left alone (only ``written``
    and ``how`` would differ, and the first writing is the truthful one). One that disagrees is an
    error: two exports of the same version cannot both be right about the hashes, and overwriting
    would destroy the evidence of whichever is wrong.
    """
    path = record_path(root, export_json)
    record = build_record(
        export_json,
        how=how,
        selection_sha256=selection_sha256,
        af_release=af_release,
        directory=directory,
    )
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("files") == record["files"] and existing.get("export") == record["export"]:
            return path
        raise RecordError(
            f"{path} already records different content for this version "
            f"(existing files {existing.get('files')}, new {record['files']}). One of the two is "
            "wrong; neither is overwritten. Compare them before deciding."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return path


def read_record(path: Path) -> dict[str, Any]:
    """A record, checked the way workstream 11's reader checks it."""
    record = json.loads(Path(path).read_text())
    if not isinstance(record, dict):
        raise RecordError(f"{path}: not a JSON object")
    if record.get("how") not in HOW:
        raise RecordError(f"{path}: how is {record.get('how')!r}, not one of {HOW}")
    _checked(record.get("export") or {})
    files = record.get("files")
    if not isinstance(files, dict) or any(not SHA256.match(str(v)) for v in files.values()):
        raise RecordError(f"{path}: files are not all sha256: {files}")
    return record


def back_fill(
    store: Store,
    subtype: str,
    selection: Path,
    root: Path,
    *,
    af_release: str | None = None,
    work: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Export the store's current version of ``subtype`` into scratch, record it, discard the files.

    The export is deterministic given (sequences version, selection rules), so a record can be made
    for an existing version at any time without a republish and without keeping the files: on B/Vic
    this takes about 1.5 s. The scratch directory is always removed, including on failure.
    """
    from af.tree.export import export, load_rules

    scratch = Path(tempfile.mkdtemp(dir=work, prefix=f"export-record-{subtype}-"))
    try:
        out = scratch / "export"
        export(store, subtype, load_rules(selection, subtype), out)
        export_json = json.loads((out / "export.json").read_text())
        path = write_record(
            root,
            export_json,
            how=BACK_FILLED,
            selection_sha256=sha256_path(selection),
            af_release=af_release,
            directory=out,
        )
        return path, export_json
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m af.tree.export_record", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("selection", type=Path, help="selection config (config/selection.toml)")
    parser.add_argument("subtype", help="its table: h3, h1, bvic, byam")
    parser.add_argument("store", type=Path, help="the store root")
    parser.add_argument(
        "root", type=Path, help="the records root (records go under <root>/exports)"
    )
    parser.add_argument("--af-release", help="the release this was exported with, for the record")
    parser.add_argument(
        "--expect-alignment",
        help="refuse unless the export's alignment.fasta has this sha256 -- use the hash a tree "
        "recorded, so a back-fill cannot quietly record a different export",
    )
    args = parser.parse_args(argv)

    path, export_json = back_fill(
        Store.open(args.store), args.subtype, args.selection, args.root, af_release=args.af_release
    )
    files = export_json["files"]
    if args.expect_alignment and files.get("alignment.fasta") != args.expect_alignment:
        print(
            f"MISMATCH: this export's alignment.fasta is {files.get('alignment.fasta')}, "
            f"not the expected {args.expect_alignment}. The record was written at {path} and is "
            "correct for what the store holds now, but it does NOT match the tree you named: that "
            "tree is content-behind this sequences version.",
            file=sys.stderr,
        )
        return 1
    print(f"{path}")
    print(
        f"  sequences {export_json['sequences']['dataset']}@{export_json['sequences']['version']}"
    )
    print(f"  leaves {export_json['leaves']}, embargoed {export_json['embargoed']}")
    for name, value in sorted(files.items()):
        print(f"  {name} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
