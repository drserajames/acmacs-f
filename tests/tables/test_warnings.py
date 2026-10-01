"""Warnings are sorted so the report can be read: counted when known, listed when not."""

from __future__ import annotations

import pytest

from af.tables import warnings

from .test_cdc import read, row


@pytest.mark.parametrize(
    ("warning", "expected"),
    [
        ("x.xlsx[S]!C9: name 'C130301388': expected type/location/isolate/year", "expected"),
        ("x.xlsx[S]!C9: name 'B/\\1': expected type/location/isolate/year", "review"),
        (
            "x.xlsx[S]!C9: name 'A/EXAMPLETOWN 9/2030': expected type/location/isolate/year",
            "review",
        ),
        ("x.xlsx[S]!E9: '32' fixed as '320' by cell_fixes.tsv:9", "applied"),
        ("x.xlsx[S]!N9: passage 'EX9_1': cannot read '_1'", "tolerated"),
        ("x.xlsx[S]!N9: something no reader has said before", "review"),
    ],
)
def test_classify(warning, expected):
    assert warnings.classify(warning)[0] == expected


def test_report_counts_the_known_and_lists_the_rest(tmp_path):
    (table,) = read(tmp_path, [row()]).tables
    table.warnings = [
        "a.xlsx[S]!C9: name 'C1': expected type/location/isolate/year",
        "a.xlsx[S]!C10: name 'C2': expected type/location/isolate/year",
        "a.xlsx[S]!C11: name 'B/\\1': expected type/location/isolate/year",
    ]
    lines = warnings.report([table])
    assert lines[0] == "warnings: applied 0, expected 2, tolerated 0, review 1"
    assert any("B/\\1" in line for line in lines)  # listed in full
    assert not any("'C1'" in line for line in lines)  # counted only
