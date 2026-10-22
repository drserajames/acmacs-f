"""A folder entry with ``names`` reads a lab's original workbooks under source_names names."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables import update
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


def settings(rules_dir: Path, root: Path, names: str = "") -> update.TablesSettings:
    entry = update.VIDRLInputs(
        lab="LABV", dir=root, start="2030-01-01", subtype="B", lineage="VICTORIA", names=names
    )
    return update.TablesSettings(rules=rules_dir, run="t", vidrl=[entry])


def test_originals_are_read_under_their_read_names(tmp_path, rules_dir, rules):
    root = originals(tmp_path / "originals")
    given = settings(rules_dir, root, names=SET)
    staged, report = update._stage_sources(given, rules, tmp_path / "work" / "sources")
    tables, _, errors = update._read_all(staged, rules)
    assert errors == []
    (table,) = tables
    assert (table.date, table.meta["file"]) == ("2030-01-02", "labv-20300102.xlsx")
    assert rules.source_names.rules[0].hits == 1
    assert report[0] == f"LABV {SET}: 1 originals in {root}, read under their source_names names"
    assert report[1:] == [
        f"LABV workbooks in {root} with no source_names row: 2, 1 dated on or after 2030-01-01",
        "  2030/20300105 LabV.xlsx",
    ]
    # provenance names the original, not the link the run read it through
    assert update._input_files(given, rules) == [root / "2030" / "LabV first test A.xlsx"]


def test_the_staging_folder_is_rebuilt_every_run(tmp_path, rules_dir, rules):
    root = originals(tmp_path / "originals")
    sources = tmp_path / "work" / "sources"
    (sources / SET).mkdir(parents=True)
    (sources / SET / "labv-20291231.xlsx").write_bytes(b"")  # a row since removed
    staged, _ = update._stage_sources(settings(rules_dir, root, names=SET), rules, sources)
    assert [p.name for p in (sources / SET).iterdir()] == ["labv-20300102.xlsx"]
    assert staged.vidrl[0].dir == sources / SET


def test_a_missing_original_or_a_group_with_no_rows_is_fatal(tmp_path, rules_dir, rules):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="originals not found"):
        update._stage_sources(settings(rules_dir, empty, names=SET), rules, tmp_path / "s")
    with pytest.raises(ValueError, match="no source_names rows for LABV no-such-set"):
        update._stage_sources(
            settings(rules_dir, empty, names="no-such-set"), rules, tmp_path / "s"
        )


def test_an_entry_without_names_is_not_staged(tmp_path, rules_dir, rules):
    given = settings(rules_dir, tmp_path)
    staged, report = update._stage_sources(given, rules, tmp_path / "s")
    assert (staged, report) == (given, [])
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
        update._stage_sources(given, rules, tmp_path / "s")
