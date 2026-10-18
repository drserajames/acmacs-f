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

from af.store import Store, StoreRef
from af.tree import export as E
from af.tree import export_record as R

from .test_stages import OUTGROUP, export_with, fill, rules

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


def current(store: Store) -> StoreRef:
    return store.current("sequences", "h3")


def exported(store: Store, out: Path) -> dict[str, object]:
    export_with(store, rules(), out)
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
    path, export_json = R.back_fill(
        store, "h3", selection, root, sequences=current(store), work=work
    )
    assert path.exists()
    assert R.read_record(path)["how"] == R.BACK_FILLED
    assert export_json["sequences"] == store.current("sequences", "h3").to_json()
    assert list(work.iterdir()) == []  # the export files are not kept


def test_back_fill_reproduces_the_hashes_a_direct_export_gives(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    """Back-filling is only sound because the export is deterministic for a fixed version."""
    direct = exported(store, tmp_path / "out")
    _, from_back_fill = R.back_fill(
        store, "h3", selection, tmp_path / "records", sequences=current(store)
    )
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
    path, _ = R.back_fill(store, "h3", selection, tmp_path / "records", sequences=current(store))
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


def test_a_back_fill_can_record_which_env_wrote_the_bytes(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    """The env that wrote a tree's own files is often not the one available at back-fill time.

    The first real record needed exactly this: the tree's leaves.parquet came from o's release
    env, which no laptop release reproduces, so the note says where the bytes came from and why
    pyarrow was pinned. Without it the record looks as though a release wrote it.
    """
    path, _ = R.back_fill(
        store,
        "h3",
        selection,
        tmp_path / "records",
        sequences=current(store),
        note="pyarrow pinned to X",
    )
    assert R.read_record(path)["note"] == "pyarrow pinned to X"
    assert (
        R.read_record(path)["af_release"] is None
    )  # not written by a release, and does not claim to


# ---------------------------------------------------------------------------------------------
# The record written at export, by the exporting process itself


def cli(store: Store, selection: Path, out: Path, *extra: str) -> int:
    return E.main([str(selection), "h3", str(store.root), str(out), *extra])


def test_an_export_writes_its_own_record_when_given_the_records_root(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    """The env that writes the bytes writes the record, so the two match by construction."""
    out, root = tmp_path / "out", tmp_path / "records"
    assert cli(store, selection, out, "--records", str(root)) == 0
    export_json = json.loads((out / E.EXPORT_FILE).read_text())
    record = R.read_record(R.record_path(root, export_json))
    assert record["how"] == R.AT_EXPORT
    assert record["files"] == export_json["files"]
    assert record["export"] == export_json
    assert record["selection_sha256"] == R.sha256_path(selection)
    assert sorted(record["content"]) == sorted(export_json["files"])


def test_an_export_must_say_where_its_record_goes_or_why_there_is_none(
    store: Store, tmp_path: Path, selection: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No silent default: an unrecorded export is only found out at report time."""
    with pytest.raises(SystemExit) as refused:
        cli(store, selection, tmp_path / "out")
    assert refused.value.code == 2
    assert "--records" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()  # refused before anything was written


def test_an_export_can_decline_a_record_only_with_a_reason(
    store: Store, tmp_path: Path, selection: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        cli(store, selection, tmp_path / "blank", "--no-record", "  ")
    assert cli(store, selection, tmp_path / "out", "--no-record", "placement experiment") == 0
    assert "no export record: placement experiment" in capsys.readouterr().out
    assert not list(tmp_path.rglob(f"*{R.RECORD_SUFFIX}"))


def test_a_record_is_refused_for_files_the_export_did_not_write(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    """A record vouches for bytes: a directory changed since the export gets no record."""
    out = tmp_path / "out"
    exported(store, out)
    with (out / E.ALIGNMENT_FILE).open("a") as handle:
        handle.write(">another\nACGT\n")
    with pytest.raises(R.RecordError, match="not the files this export wrote"):
        R.record_export(out, selection, tmp_path / "records")
    assert not list(tmp_path.rglob(f"*{R.RECORD_SUFFIX}"))


def test_recording_an_export_twice_changes_nothing(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    out, root = tmp_path / "out", tmp_path / "records"
    exported(store, out)
    first = R.record_export(out, selection, root)
    before = first.read_text()
    assert R.record_export(out, selection, root) == first
    assert first.read_text() == before


# ---------------------------------------------------------------------------------------------
# A back-fill names the version it re-exported, and what it was checked against


def back_fill_cli(store: Store, selection: Path, root: Path, *extra: str) -> int:
    return R.main([str(selection), "h3", str(store.root), str(root), *extra])


def test_a_back_fill_records_the_version_asked_for_not_current(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    """On 13 Oct a back-fill of CURRENT recorded a version seven after the one the tree read."""
    read = current(store)
    fill(store, names="oab", pull="p2")  # CURRENT moves on
    assert current(store) != read
    path, export_json = R.back_fill(store, "h3", selection, tmp_path / "records", sequences=read)
    assert export_json["sequences"] == read.to_json()
    assert path.name == f"{read.version}{R.RECORD_SUFFIX}"


def test_a_back_fill_checked_against_a_tree_keeps_the_hash_it_was_checked_with(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    alignment = exported(store, tmp_path / "out")["files"][E.ALIGNMENT_FILE]  # type: ignore[index]
    path, _ = R.back_fill(
        store,
        "h3",
        selection,
        tmp_path / "records",
        sequences=current(store),
        expect_alignment=alignment,
    )
    assert R.read_record(path)["checked_against"] == {E.ALIGNMENT_FILE: alignment}


def test_a_back_fill_that_does_not_reproduce_the_tree_writes_nothing(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    """A record made to vouch for a tree it does not match would be the wrong evidence."""
    with pytest.raises(R.RecordError, match="content-behind"):
        R.back_fill(
            store,
            "h3",
            selection,
            tmp_path / "records",
            sequences=current(store),
            expect_alignment="0" * 64,
        )
    assert not list(tmp_path.rglob(f"*{R.RECORD_SUFFIX}"))


def test_the_back_fill_command_needs_the_version(
    store: Store, tmp_path: Path, selection: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "records"
    with pytest.raises(SystemExit):
        back_fill_cli(store, selection, root)
    assert "--sequences" in capsys.readouterr().err
    assert back_fill_cli(store, selection, root, "--sequences", "0" * 16) == 1  # no such version
    assert "no such version" in capsys.readouterr().err
    version = current(store).version
    assert back_fill_cli(store, selection, root, "--sequences", version) == 0
    refused = back_fill_cli(
        store, selection, root, "--sequences", version, "--expect-alignment", "0" * 64
    )
    assert refused == 1
    assert "REFUSED" in capsys.readouterr().err


def test_a_format_3_record_keeps_the_byte_hashes_and_the_selection_inputs(
    store: Store, tmp_path: Path, selection: Path
) -> None:
    """``files`` (bytes) stays what a report compares; the selection inputs ride in the
    embedded export.json, so the record's shape for workstream 11's reader is unchanged."""
    path, export_json = R.back_fill(
        store, "h3", selection, tmp_path / "records", sequences=current(store)
    )
    record = R.read_record(path)
    assert record["files"] == export_json["files"]
    assert record["export"]["format"] == 3
    assert record["export"]["selection"]["config"]["sha256"] == R.sha256_path(selection)
