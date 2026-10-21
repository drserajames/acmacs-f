"""Section selection for the serum addenda (af.chart.serum_selection), on invented sera."""

import datetime as dt

import pytest

from af.chart.serum_selection import Candidate, Section, SelectionError, select_sera

D = dt.date
RECENT = D(2026, 6, 1)


def serum(n: int, *, clades=(), aa=None, passage="cell", titrated=RECENT, name=None) -> Candidate:
    designation = name or f"A(H3N2)/EXAMPLETOWN/{n}/2025 F{n}/25"
    return Candidate(
        key=("lab-a", str(n)),
        designation=designation,
        passage_class=passage,
        clades=frozenset(clades),
        aa=aa or {},
        last_titrated=titrated,
    )


SERA = [
    serum(1, clades={"X", "X.1"}, name="A(H3N2)/EXAMPLEVILLE/1/2025 F1/25"),  # the vaccine
    serum(2, clades={"X", "X.1"}, passage="egg", name="A(H3N2)/EXAMPLEVILLE/1/2025 F2/25"),
    serum(3, clades={"X", "X.1"}),
    serum(4, clades={"X", "X.1"}, aa={155: "E"}),
    serum(5, clades={"X"}, aa={205: "K"}),
    serum(6, clades={"X", "X.1"}, titrated=D(2024, 1, 1)),  # too old
    serum(7, clades={"X", "X.1"}, titrated=None),  # undated: kept
]


def labels(sel, section):
    return [c.serum.key[1] for c in sel.sections[section]]


def test_each_serum_goes_to_one_section_with_its_reason():
    sections = [
        Section("vaccine", strains=("EXAMPLEVILLE/1/2025",)),
        Section("X.1", clade="X.1"),
        Section("X 205K", clade="X", aa=("205K",)),
        Section("155E", aa=("155E",), priority=-1),
    ]
    sel = select_sera(SERA, sections, since=D(2025, 9, 1))
    # the vaccine sera are X.1 too: the smaller match set wins them
    assert labels(sel, "vaccine") == ["1", "2"]
    assert labels(sel, "X.1") == ["3", "7"]
    assert labels(sel, "X 205K") == ["5"]
    assert labels(sel, "155E") == ["4"]  # priority beats its clade
    assert sel.sections["vaccine"][0].reason == "designation contains 'EXAMPLEVILLE/1/2025'"
    assert sel.sections["X 205K"][0].reason == "clade X + aa 205K"
    assert sel.excluded == {"last titrated before 2025-09-01": 1, "no titration date (kept)": 1}
    assert sel.report()["sections"] == {"vaccine": 2, "X.1": 2, "X 205K": 1, "155E": 1}


def test_drop_claims_and_extra_shares():
    sections = [
        Section("X.1", clade="X.1"),
        Section("lead page", strains=("F3/25", "F5/25"), extra=True),
        Section("155E", aa=("155E",), drop=True, priority=-1),
    ]
    sel = select_sera(SERA, sections, since=None)
    assert labels(sel, "lead page") == ["3", "5"]
    assert "4" not in labels(sel, "X.1") and "3" in labels(sel, "X.1")  # extra did not claim 3
    assert [c.serum.key[1] for c in sel.dropped["155E"]] == ["4"]
    assert "155E" not in sel.sections


def test_passage_classes_and_exclusions():
    sections = [Section("vaccine", strains=("EXAMPLEVILLE/1/2025",), exclude=("F2/25",))]
    sel = select_sera(SERA, sections, since=None, passage_classes=frozenset({"cell"}))
    assert labels(sel, "vaccine") == ["1"]
    assert sel.excluded == {"passage class egg": 1}
    sel = select_sera(SERA, sections, since=None)
    assert labels(sel, "vaccine") == ["1"]
    assert sel.excluded == {"excluded from 'vaccine' by designation": 1}


def test_a_section_matching_nothing_is_an_error_unless_optional():
    with pytest.raises(SelectionError, match="'X 145R' matches no serum"):
        select_sera(SERA, [Section("X 145R", clade="X", aa=("145R",))], since=None)
    sel = select_sera(SERA, [Section("X 145R", clade="X", aa=("145R",), optional=True)], since=None)
    assert sel.sections["X 145R"] == []


def test_sections_from_config_are_checked():
    s = Section.from_config({"label": "X 96R", "clade": "X", "aa": ["96R"], "priority": -1})
    assert s == Section("X 96R", clade="X", aa=("96R",), priority=-1)
    with pytest.raises(SelectionError, match="unknown keys"):
        Section.from_config({"label": "a", "strain": ["x"]})
    with pytest.raises(SelectionError, match="give strains, or clade"):
        Section.from_config({"label": "a", "strains": ["x"], "clade": "X"})
    with pytest.raises(SelectionError, match="not like 155E"):
        Section("a", aa=("E155",))
    with pytest.raises(SelectionError, match="labels repeat"):
        select_sera(SERA, [Section("a", clade="X"), Section("a", clade="X.1")], since=None)
