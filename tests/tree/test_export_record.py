"""The export record: (sequences version, export) -> the two file hashes.

Why these tests and not others: the record exists so a report can tell "the tree read the same
content" from "the tree is behind". Both answers are load-bearing, so each has a test, and the
record must refuse rather than guess whenever it cannot tell.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from af.store import Store
from af.tree import export as E
from af.tree import export_record as R

from .test_stages import OUTGROUP, fill, rules

# The same single optional host rule as the `rules()` fixture, as a config file, because back_fill
# loads the selection from disk (and hashes it into the record) rather than taking SubtypeRules.
SELECTION = f"""
[h3]
dataset = "h3"
[h3.outgroup]
epi_isl = "{OUTGROUP.epi_isl}"
accession = "{OUTGROUP.accession}"
reason = "{OUTGROUP.reason}"
[[h3.rule]]
id = "host"
kind = "host"
reason = "human only"
optional = true
allow = ["Human"]
"""


@pytest.fixture
def selection(tmp_path: Path) -> Path:
    path = tmp_path / "selection.toml"
    path.write_text(SELECTION)
    return path


@pytest.fixture
def store(tmp_path: Path) -> Store:
    st = Store.create(tmp_path / "store")
    fill(st)
    return st


def exported(store: Store, out: Path) -> dict[str, object]:
    E.export(store, "h3", rules(), out)
    return json.loads((out / E.EXPORT_FILE).read_text())


def test_the_record_is_keyed_by_the_alignment_hash_a_tree_records(
    store: Store, tmp_path: Path
) -> None:
    """A tree records the alignment hash, so that hash must lead straight to the record."""
    export_json = exported(store, tmp_path / "out")
    path = R.write_record(tmp_path / "records", export_json, how=R.AT_EXPORT)
    alignment = export_json["files"][E.ALIGNMENT_FILE]  # type: ignore[index]
    version = export_json["sequences"]["version"]  # type: ignore[index]
    assert path == (
        tmp_path / "records" / R.RECORDS_DIR / "h3" / alignment[:16] / f"{version}.export.json"
    )
    assert R.read_record(path)["files"][E.ALIGNMENT_FILE] == alignment


def test_the_record_embeds_export_json_verbatim(store: Store, tmp_path: Path) -> None:
    """Restating export.json's fields would let the two drift (design rule 6)."""
    export_json = exported(store, tmp_path / "out")
    path = R.write_record(tmp_path / "records", export_json, how=R.AT_EXPORT)
    assert R.read_record(path)["export"] == export_json


def test_writing_the_same_record_twice_is_idempotent(store: Store, tmp_path: Path) -> None:
    export_json = exported(store, tmp_path / "out")
    first = R.write_record(tmp_path / "records", export_json, how=R.AT_EXPORT)
    before = first.read_text()
    again = R.write_record(tmp_path / "records", export_json, how=R.BACK_FILLED)
    assert again == first
    assert (
        first.read_text() == before
    )  # the first writing is the truthful one; how is not rewritten


def test_a_record_disagreeing_about_the_hashes_is_refused_not_overwritten(
    store: Store, tmp_path: Path
) -> None:
    """Two exports of one version cannot both be right; overwriting would destroy the evidence."""
    export_json = exported(store, tmp_path / "out")
    root = tmp_path / "records"
    path = R.write_record(root, export_json, how=R.AT_EXPORT)
    other = json.loads(json.dumps(export_json))
    other["files"][E.LEAVES_FILE] = "0" * 64
    with pytest.raises(R.RecordError, match="already records different content"):
        R.write_record(root, other, how=R.BACK_FILLED)
    assert R.read_record(path)["export"] == export_json  # untouched


def test_an_export_without_a_sequences_ref_or_hashes_is_refused(tmp_path: Path) -> None:
    """A record that vouches for nothing is worse than no record: the reader would trust it."""
    with pytest.raises(R.RecordError, match="no 'sequences' StoreRef"):
        R.write_record(tmp_path / "r", {"files": {"alignment.fasta": "a" * 64}}, how=R.AT_EXPORT)
    with pytest.raises(R.RecordError, match="no 'files' hashes"):
        R.write_record(
            tmp_path / "r", {"sequences": {"dataset": "h3", "version": "v1"}}, how=R.AT_EXPORT
        )
    with pytest.raises(R.RecordError, match="not all sha256"):
        R.write_record(
            tmp_path / "r",
            {"sequences": {"dataset": "h3", "version": "v1"}, "files": {"alignment.fasta": "nope"}},
            how=R.AT_EXPORT,
        )


def test_how_must_say_whether_it_was_written_at_export_or_back_filled(tmp_path: Path) -> None:
    export_json = {
        "sequences": {"dataset": "h3", "version": "v1"},
        "files": {"alignment.fasta": "a" * 64},
    }
    with pytest.raises(R.RecordError, match="how must be one of"):
        R.build_record(export_json, how="whenever")


def test_written_must_carry_a_zone(tmp_path: Path) -> None:
    """A bare local time cannot be compared across machines, so it is not a record."""
    export_json = {
        "sequences": {"dataset": "h3", "version": "v1"},
        "files": {"alignment.fasta": "a" * 64},
    }
    with pytest.raises(R.RecordError, match="needs a time zone"):
        R.build_record(export_json, how=R.AT_EXPORT, written=datetime.datetime(2026, 10, 12))


def test_back_fill_records_an_existing_version_and_leaves_no_scratch(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    """The point of a back-fill: a record for a version published before records existed."""
    root = tmp_path / "records"
    work = tmp_path / "work"
    work.mkdir()
    path, export_json = R.back_fill(store, "h3", selection, root, work=work)
    assert path.exists()
    assert R.read_record(path)["how"] == R.BACK_FILLED
    assert export_json["sequences"] == store.current("sequences", "h3").to_json()
    assert list(work.iterdir()) == []  # the export files are not kept


def test_back_fill_reproduces_the_hashes_a_direct_export_gives(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    """Back-filling is only sound because the export is deterministic for a fixed version."""
    direct = exported(store, tmp_path / "out")
    _, from_back_fill = R.back_fill(store, "h3", selection, tmp_path / "records")
    assert from_back_fill["files"] == direct["files"]


def test_read_record_refuses_a_record_it_cannot_check(tmp_path: Path) -> None:
    path = tmp_path / "bad.export.json"
    path.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(R.RecordError, match="not a JSON object"):
        R.read_record(path)
    path.write_text(json.dumps({"export": {}, "files": {}, "how": R.AT_EXPORT}))
    with pytest.raises(R.RecordError):
        R.read_record(path)


def test_content_hashes_are_recorded_and_ignore_the_parquet_writer_version(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    """leaves.parquet is NOT byte-reproducible: Parquet embeds the writer's version.

    Measured on bvic@dbe4b3bd, 12 Oct 2026: pyarrow 25.0.1 wrote leaves.parquet 75414a16... (what
    trees/bvic/weekly@c1ee2d47 recorded) and 26.0.0 wrote e7255143..., with schemas and tables
    equal. So a check on file bytes alone refuses same-content pairs whenever the record's writer
    differs from the build's. The record therefore carries writer-independent content hashes, and
    this test is what stops them being removed as redundant.
    """
    path, _ = R.back_fill(store, "h3", selection, tmp_path / "records")
    record = R.read_record(path)
    assert set(record["content"]) == {E.ALIGNMENT_FILE, E.LEAVES_FILE}
    assert record["writer"]["pyarrow"]
    # the text file's content hash IS its file hash; the parquet's is deliberately not
    assert record["content"][E.ALIGNMENT_FILE] == record["files"][E.ALIGNMENT_FILE]
    assert record["content"][E.LEAVES_FILE] != record["files"][E.LEAVES_FILE]


def test_the_content_hash_of_a_rewritten_parquet_is_unchanged(tmp_path: Path) -> None:
    """Rewriting the same table must not change the content hash, though the bytes may."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"leaf_id": ["a", "b"], "embargoed": [False, True]})
    one, two = tmp_path / "one.parquet", tmp_path / "two.parquet"
    pq.write_table(table, one)
    pq.write_table(table, two, compression="gzip")  # different bytes, same data
    assert one.read_bytes() != two.read_bytes()
    assert R.content_hash(one) == R.content_hash(two)


def test_a_record_cannot_claim_content_it_could_not_hash(tmp_path: Path) -> None:
    export_json = {
        "sequences": {"dataset": "h3", "version": "v1"},
        "files": {"alignment.fasta": "a" * 64},
    }
    with pytest.raises(R.RecordError, match="cannot hash content"):
        R.build_record(export_json, how=R.AT_EXPORT, directory=tmp_path)
