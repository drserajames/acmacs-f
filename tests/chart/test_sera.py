"""af.chart.sera: which sera are not ferret, by which test, and what we cannot tell."""

import pytest

from af.chart.model import Chart, Serum, Titres, empty_table
from af.chart.sera import Marker, SeraPolicyError, classify, non_ferret, read_markers

MARKERS = [
    Marker("name", "TESTPOOL", "pooled", "synthetic", "test"),
    Marker("serum_id", "Humans", "human", "synthetic", "test"),
]


@pytest.mark.parametrize(
    ("serum", "why"),
    [
        (Serum("TEST-1", serum_id="F1", species="FERRET"), None),
        (Serum("TEST-2", serum_id="F2"), None),  # empty species, no marker: ferret by default
        (Serum("TEST-3", serum_id="M1", species="MOUSE"), "species MOUSE"),
        (Serum("2020 TESTPOOL", serum_id="P1"), "marker name~/TESTPOOL/ (pooled)"),
        (Serum("TEST-4", serum_id="20HumanS01"), "marker serum_id~/Humans/ (human)"),
        # a recorded species wins: a ferret serum whose name happens to match is still ferret
        (Serum("TESTPOOL-FERRET", serum_id="F3", species="FERRET"), None),
    ],
)
def test_classify(serum, why):
    assert classify(serum, MARKERS) == why


def test_report_counts_the_blind_spot():
    sera = [
        Serum("TEST-1", serum_id="F1", species="FERRET"),
        Serum("TEST-2", serum_id="F2"),
        Serum("TEST-3", serum_id="F3"),
        Serum("TEST-4", serum_id="M1", species="MOUSE"),
    ]
    chart = Chart({}, [], sera, Titres(empty_table(0, 4)))
    r = non_ferret(chart, MARKERS)
    assert [d for _, d, _ in r.non_ferret] == ["TEST-4 M1"]
    assert (r.ferret_recorded, r.ferret_by_default) == (1, 2)


def test_markers_file_is_required_and_checked(tmp_path):
    with pytest.raises(SeraPolicyError, match="missing"):
        read_markers(tmp_path / "none.tsv")
    path = tmp_path / "m.tsv"
    path.write_text("field\tpattern\tkind\treason\tevidence\nname\tPOOL\tpooled\t\tx\n")
    with pytest.raises(SeraPolicyError, match="empty reason"):
        read_markers(path)
    path.write_text("field\tpattern\tkind\treason\tevidence\npassage\tPOOL\tpooled\tr\tx\n")
    with pytest.raises(SeraPolicyError, match="field must be"):
        read_markers(path)
    path.write_text("# comment\nfield\tpattern\tkind\treason\tevidence\nname\tPOOL\tpooled\tr\tx\n")
    assert [m.pattern for m in read_markers(path)] == ["POOL"]


def test_the_verification_statement_is_honest_about_the_blind_spot():
    unrecorded = Chart({}, [], [Serum("TEST-1", serum_id="F1")], Titres(empty_table(0, 1)))
    text = non_ferret(unrecorded, MARKERS).verification()
    assert text.startswith("UNVERIFIED: 1 of 1") and "assumption, not a verified fact" in text
    recorded = Chart(
        {}, [], [Serum("TEST-1", serum_id="F1", species="FERRET")], Titres(empty_table(0, 1))
    )
    assert non_ferret(recorded, MARKERS).verification().startswith("VERIFIED: all 1")
