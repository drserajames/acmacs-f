"""Which workbooks a [[tables.ac21]] / [[tables.niid]] folder entry reads."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables.update import AC21Inputs, _dated_files


def folder(tmp_path: Path, *names: str) -> Path:
    for name in names:
        (tmp_path / name).write_bytes(b"")
    return tmp_path


def test_reads_from_start_skipping_lock_files(tmp_path):
    d = folder(tmp_path, "lab-20290101.xlsx", "lab-20300101.xlsx", "~$lab-20300101.xlsx")
    assert [p.name for p in _dated_files(AC21Inputs("LABX", d, "2030-01-01"))] == [
        "lab-20300101.xlsx"
    ]


def test_an_undated_name_is_an_error_unless_excluded(tmp_path):
    d = folder(tmp_path, "lab--0011130.xlsx", "lab-20300101.xlsx")
    with pytest.raises(ValueError, match="no YYYYMMDD"):
        _dated_files(AC21Inputs("LABX", d, "2030-01-01"))
    inputs = AC21Inputs("LABX", d, "2030-01-01", exclude=["lab--0011130.xlsx"])
    assert [p.name for p in _dated_files(inputs)] == ["lab-20300101.xlsx"]


def test_an_exclude_that_matches_nothing_is_an_error(tmp_path):
    d = folder(tmp_path, "lab-20300101.xlsx")
    with pytest.raises(FileNotFoundError, match="excluded workbooks not found"):
        _dated_files(AC21Inputs("LABX", d, "2030-01-01", exclude=["gone.xlsx"]))
