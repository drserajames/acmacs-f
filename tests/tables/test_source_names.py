"""A folder entry with ``names`` reads a lab's original workbooks under source_names names."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from af.store import PathsConfig, Store, Work
from af.tables import archive, update
from af.tables.rules import Rules

from .synthetic_rules import RULES, write_rules
from .test_vidrl import workbook

SET = "labv-own-2030"


def originals(root: Path) -> Path:
    """The lab's own folders and names: the date the run needs is not in the first name."""
    (root / "2030").mkdir(parents=True)
    (root / "2029").mkdir()
    workbook(root / "2030" / "LabV first test A.xlsx")
    workbook(root / "2030" / "20300105 LabV.xlsx")  # arrived, no row yet
    workbook(root / "2029" / "20291201 LabV.xlsx")  # before the entry's start
    return root


def store(tmp_path: Path) -> Store:
    path = tmp_path / "store"
    return Store.open(path) if path.exists() else Store.create(path)


def settings(rules_dir: Path, root: Path, names: str = "") -> update.TablesSettings:
    entry = update.VIDRLInputs(
        lab="LABV", dir=root, start="2030-01-01", subtype="B", lineage="VICTORIA", names=names
    )
    return update.TablesSettings(rules=rules_dir, run="t", vidrl=[entry])


def test_originals_are_read_under_their_read_names(tmp_path, rules_dir, rules):
    root = originals(tmp_path / "originals")
    given = settings(rules_dir, root, names=SET)
    staged, report, _ = update._stage_sources(
        given, rules, tmp_path / "work" / "sources", store(tmp_path)
    )
    tables, _, errors = update._read_all(staged, rules)
    assert errors == []
    (table,) = tables
    assert (table.date, table.meta["file"]) == ("2030-01-02", "labv-20300102.xlsx")
    assert rules.source_names.rules[0].hits == 1
    assert report[0] == (
        f"LABV {SET}: 1 workbooks read under their source_names names;"
        f" 0 from raw/tables/labv/{SET}, 1 from {root}"
    )
    assert report[1:] == [
        f"LABV workbooks in {root} not named by source_names: 2, 1 dated on or after 2030-01-01",
        "  2030/20300105 LabV.xlsx",
    ]
    # provenance names the original, not the link the run read it through
    assert update._input_files(given, rules) == [root / "2030" / "LabV first test A.xlsx"]


def test_the_staging_folder_is_rebuilt_every_run(tmp_path, rules_dir, rules):
    root = originals(tmp_path / "originals")
    sources = tmp_path / "work" / "sources"
    (sources / SET).mkdir(parents=True)
    (sources / SET / "labv-20291231.xlsx").write_bytes(b"")  # a row since removed
    staged, _, _ = update._stage_sources(
        settings(rules_dir, root, names=SET), rules, sources, store(tmp_path)
    )
    assert [p.name for p in (sources / SET).iterdir()] == ["labv-20300102.xlsx"]
    assert staged.vidrl[0].dir == sources / SET


def test_a_missing_original_or_a_group_with_no_rows_is_fatal(tmp_path, rules_dir, rules):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="originals not found, nor archived"):
        update._stage_sources(
            settings(rules_dir, empty, names=SET), rules, tmp_path / "s", store(tmp_path)
        )
    with pytest.raises(ValueError, match="no source_names rows for LABV no-such-set"):
        update._stage_sources(
            settings(rules_dir, empty, names="no-such-set"), rules, tmp_path / "s", store(tmp_path)
        )


def test_an_entry_without_names_is_not_staged(tmp_path, rules_dir, rules):
    given = settings(rules_dir, tmp_path)
    staged, report, sets = update._stage_sources(given, rules, tmp_path / "s", store(tmp_path))
    assert (staged, report, sets) == (given, [], [])
    assert not (tmp_path / "s").exists()


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (f"LABV\t{SET}\t/abs/x.xlsx\tlabv-20300102.xlsx", "must be inside the root"),
        (f"LABV\t{SET}\t../x.xlsx\tlabv-20300102.xlsx", "must be inside the root"),
        (f"LABV\t{SET}\tx.xlsx\tlabv-first.xlsx", "is not a dated name"),
        (f"LABV\t{SET}\tx.xlsx\tsub/labv-20300102.xlsx", "is not a dated name"),
        ("LABV\tNot A Set\tx.xlsx\tlabv-20300102.xlsx", "is not a set name"),
        (f"LABV\t{SET}\ty.xlsx\tlabv-20300102.xlsx", "repeats source_names.tsv:2"),
    ],
)
def test_source_names_rows_are_checked_on_load(tmp_path, row, message):
    directory = write_rules(tmp_path / "rules")
    lines = RULES["source_names"]
    meta = "\tinvented for tests\ttest\t2030-01-01\tyes"
    (directory / "source_names.tsv").write_text(
        "\n".join([*lines, row + meta]) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        Rules(directory)


def test_two_sets_reading_one_name_are_an_error(tmp_path):
    """A delivery and the lab's archive both holding a test must not read as two tests."""
    directory = write_rules(tmp_path / "rules")
    meta = "\tinvented for tests\ttest\t2030-01-01\tyes"
    rows = [
        *RULES["source_names"],
        "LABV\tlabv-delivery\tLabV first test A.xlsx\tlabv-20300102.xlsx" + meta,
    ]
    (directory / "source_names.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    rules = Rules(directory)
    archive = originals(tmp_path / "archive")
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    workbook(delivery / "LabV first test A.xlsx")
    entries = [
        update.VIDRLInputs("LABV", archive, "2030-01-01", "B", "VICTORIA", names=SET),
        update.VIDRLInputs("LABV", delivery, "2030-01-01", "B", "VICTORIA", names="labv-delivery"),
    ]
    given = update.TablesSettings(rules=directory, run="t", vidrl=entries)
    with pytest.raises(ValueError, match="read from sets labv-own-2030 and labv-delivery"):
        update._stage_sources(given, rules, tmp_path / "s", store(tmp_path))


def test_a_copy_named_as_a_read_name_is_accounted_for(tmp_path):
    """A root of curated copies: the copy of a test now read from its original is not new."""
    directory = write_rules(tmp_path / "rules")
    meta = "\tinvented for tests\ttest\t2030-01-01\tyes"
    rows = [
        *RULES["source_names"],
        "LABV\tlabv-copies\tlabv-20291230.xlsx\tlabv-20291230.xlsx" + meta,
    ]
    (directory / "source_names.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    root = originals(tmp_path / "originals")
    copies = tmp_path / "copies"
    copies.mkdir()
    workbook(copies / "labv-20291230.xlsx")  # read from this root
    workbook(copies / "labv-20300102.xlsx")  # the read name of the original in `root`
    workbook(copies / "labv-20300109.xlsx")  # a copy nothing accounts for
    entries = [
        update.VIDRLInputs("LABV", root, "2030-01-01", "B", "VICTORIA", names=SET),
        update.VIDRLInputs("LABV", copies, "2029-12-01", "B", "VICTORIA", names="labv-copies"),
    ]
    given = update.TablesSettings(rules=directory, run="t", vidrl=entries)
    _, report, _ = update._stage_sources(given, Rules(directory), tmp_path / "s", store(tmp_path))
    i = report.index(
        f"LABV workbooks in {copies} not named by source_names: 1, 1 dated on or after 2029-12-01"
    )
    assert report[i + 1] == "  labv-20300109.xlsx"


def run(tmp_path: Path, root: Path, *, dry_run: bool = False) -> tuple[int, list[str], Store]:
    """A full update of one names entry into a store of its own."""
    paths = PathsConfig(store=tmp_path / "store", work=tmp_path / "work")
    if not paths.store.exists():
        Store.create(paths.store)
        Work.create(paths.work)
    given = settings(write_rules(tmp_path / "rules", all_optional=True), root, names=SET)
    status, report = update.update(given, paths, dry_run=dry_run)
    return status, report, Store.open(paths.store)


def test_originals_are_archived_by_a_publishing_run_and_read_from_there(tmp_path):
    root = originals(tmp_path / "originals")
    status, report, st = run(tmp_path, root, dry_run=True)
    assert status == 0 and f"would archive 1 originals into raw/tables/labv/{SET}" in report
    assert not (st.root / "raw").exists()  # a dry run writes nothing to the store

    status, report, st = run(tmp_path, root)
    assert status == 0, report
    ref = st.current("raw", archive.dataset("LABV", SET))
    copy = st.version_dir(ref) / "2030" / "LabV first test A.xlsx"
    assert copy.is_file() and not copy.is_symlink()
    assert copy.read_bytes() == (root / "2030" / "LabV first test A.xlsx").read_bytes()
    tables = st.current("tables", "labv/bvic-hi-turkey-labv")
    cited = json.loads((st.version_dir(tables) / "PROVENANCE.json").read_text())["inputs"]
    assert {"store": ref.to_json()} in cited
    assert not any("LabV first test A" in json.dumps(i) for i in cited if "external" in i)

    # the root can go: the next run reads the archived copy and publishes the same tables
    shutil.rmtree(root)
    status, report, st = run(tmp_path, tmp_path / "gone")
    assert status == 0, report
    assert f"LABV {SET}: 1 workbooks read under their source_names names; 1 from" in report[0]
    assert st.current("tables", "labv/bvic-hi-turkey-labv").version == tables.version
    assert st.current("raw", archive.dataset("LABV", SET)) == ref


def test_a_root_file_that_differs_from_its_archived_copy_is_an_error(tmp_path):
    root = originals(tmp_path / "originals")
    assert run(tmp_path, root)[0] == 0
    workbook(root / "2030" / "LabV first test A.xlsx", date="Test Date: 03/01/2030")
    with pytest.raises(archive.ArchiveError, match="differs from its archived copy"):
        run(tmp_path, root)


def test_a_set_named_by_two_entries_is_an_error(tmp_path, rules_dir, rules):
    root = originals(tmp_path / "originals")
    entry = update.VIDRLInputs("LABV", root, "2030-01-01", "B", "VICTORIA", names=SET)
    given = update.TablesSettings(rules=rules_dir, run="t", vidrl=[entry, entry])
    with pytest.raises(ValueError, match=f"set {SET} is named by two folder entries"):
        update._stage_sources(given, rules, tmp_path / "s", store(tmp_path))
