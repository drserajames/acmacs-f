"""Which workbooks a [[tables.ac21]] / [[tables.niid]] folder entry reads."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables.update import AC21Inputs, TablesSettings, _dated_files, _read_all


def folder(tmp_path: Path, *names: str) -> Path:
    for name in names:
        (tmp_path / name).write_bytes(b"")
    return tmp_path


def test_reads_from_start_skipping_lock_files(tmp_path):
    d = folder(tmp_path, "lab-20290101.xlsx", "lab-20300101.xlsx", "~$lab-20300101.xlsx")
    files, errors = _dated_files(AC21Inputs("LABX", d, "2030-01-01"))
    assert ([p.name for p in files], errors) == (["lab-20300101.xlsx"], [])


def test_an_undated_name_is_an_error_unless_excluded(tmp_path):
    d = folder(tmp_path, "lab--0011130.xlsx", "lab-20300101.xlsx")
    files, errors = _dated_files(AC21Inputs("LABX", d, "2030-01-01"))
    assert [p.name for p in files] == ["lab-20300101.xlsx"]
    assert errors == [f"{d / 'lab--0011130.xlsx'}: no YYYYMMDD date in the file name"]
    inputs = AC21Inputs("LABX", d, "2030-01-01", exclude=["lab--0011130.xlsx"])
    assert _dated_files(inputs) == ([d / "lab-20300101.xlsx"], [])


def test_an_undated_name_does_not_stop_the_folder_being_read(tmp_path, rules):
    """One bad file name is counted as an error (it still blocks a publish) and every
    other workbook in the folder is still read."""
    pytest.importorskip("openpyxl")
    from .test_niid import workbook

    workbook(tmp_path / "labn-20300102.xlsx")
    (tmp_path / "labn--0011130.xlsx").write_bytes(b"")
    settings = TablesSettings(
        rules=tmp_path, run="x", niid=[AC21Inputs("LABN", tmp_path, "2030-01-01")]
    )
    tables, _, errors = _read_all(settings, rules)
    assert [t.date for t in tables] == ["2030-01-02"]
    assert errors == [f"{tmp_path / 'labn--0011130.xlsx'}: no YYYYMMDD date in the file name"]


def test_an_exclude_that_matches_nothing_is_an_error(tmp_path):
    d = folder(tmp_path, "lab-20300101.xlsx")
    with pytest.raises(FileNotFoundError, match="excluded workbooks not found"):
        _dated_files(AC21Inputs("LABX", d, "2030-01-01", exclude=["gone.xlsx"]))
