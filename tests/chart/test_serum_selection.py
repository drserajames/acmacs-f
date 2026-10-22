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
    from af.chart.serum_selection import Exclusion

    older = Exclusion("F2/25", "an older pair")
    sections = [Section("vaccine", strains=("EXAMPLEVILLE/1/2025",), exclude=(older,))]
    sel = select_sera(SERA, sections, since=None, passage_classes=frozenset({"cell"}))
    assert labels(sel, "vaccine") == ["1"]
    assert sel.excluded == {"passage class egg": 1}
    sel = select_sera(SERA, sections, since=None)
    assert labels(sel, "vaccine") == ["1"]
    assert sel.excluded == {"excluded from 'vaccine': an older pair": 1}


def test_a_section_matching_nothing_is_an_error_unless_optional():
    with pytest.raises(SelectionError, match="'X 145R' matches no serum"):
        select_sera(SERA, [Section("X 145R", clade="X", aa=("145R",))], since=None)
    sel = select_sera(SERA, [Section("X 145R", clade="X", aa=("145R",), optional=True)], since=None)
    assert sel.sections["X 145R"] == []


def test_sections_from_config_are_checked():
    """Each section says why it is there, and each exclusion is a named entry with a reason
    (design rule 9); the reasons reach the selection's report."""
    from af.chart.serum_selection import Exclusion

    why = "a test section"
    s = Section.from_config(
        {"label": "X 96R", "clade": "X", "aa": ["96R"], "priority": -1, "reason": why,
         "exclude": [{"text": "F9/25", "reason": "raised against another virus"}]}
    )  # fmt: skip
    assert s == Section(
        "X 96R", clade="X", aa=("96R",), priority=-1, reason=why,
        exclude=(Exclusion("F9/25", "raised against another virus"),),
    )  # fmt: skip
    with pytest.raises(SelectionError, match="'X 96R': no reason given"):
        Section.from_config({"label": "X 96R", "clade": "X"})
    with pytest.raises(SelectionError, match=r"exclude 'F9/25' must be \{ text"):
        Section.from_config({"label": "a", "clade": "X", "reason": why, "exclude": ["F9/25"]})
    with pytest.raises(SelectionError, match="needs text and a reason"):
        Exclusion("F9/25", " ")
    with pytest.raises(SelectionError, match="unknown keys"):
        Section.from_config({"label": "a", "strain": ["x"], "reason": why})
    with pytest.raises(SelectionError, match="give strains, or clade"):
        Section.from_config({"label": "a", "strains": ["x"], "clade": "X", "reason": why})
    with pytest.raises(SelectionError, match="not like 155E"):
        Section("a", aa=("E155",))
    with pytest.raises(SelectionError, match="labels repeat"):
        select_sera(SERA, [Section("a", clade="X"), Section("a", clade="X.1")], since=None)
    sel = select_sera(SERA, [Section("X.1", clade="X.1", reason=why)], since=None)
    assert sel.report()["reasons"] == {"X.1": why}


def test_titration_dates_come_from_the_tables_each_serum_is_in():
    from af.chart.model import Antigen, Chart, Serum, Titres
    from af.chart.serum_selection import titration_dates
    from af.chart.titre import Titre

    t = Titre.parse
    table = [[t("40"), t("80"), t("*")]]
    layers = [{(0, 0): t("40")}, {(0, 0): t("40"), (0, 1): t("80")}, {(0, 2): t("*")}]
    sources = [{"D": "20250105"}, {"D": "20260210-002"}, {"D": "20260301"}]
    sera = [Serum(f"A(H3N2)/EXAMPLETOWN/{i}/2025", f"F{i}") for i in range(3)]
    c = Chart({"S": sources}, [Antigen("A(H3N2)/EXAMPLEOTHER/1/2025")], sera, Titres(table, layers))
    assert titration_dates(c) == [D(2026, 2, 10), D(2026, 2, 10), None]
    single = Chart({"D": "20240301"}, c.antigens, sera, Titres(table))
    assert titration_dates(single) == [D(2024, 3, 1)] * 3
    with pytest.raises(SelectionError, match="3 titre layers but 2 sources"):
        titration_dates(Chart({"S": sources[:2]}, c.antigens, sera, Titres(table, layers)))


def test_columns_pair_cell_over_egg_per_map_and_pages_keep_centres_whole():
    from af.chart.serum_selection import Chosen, columns, pages

    def s(n, strain, passage, centre, chart):
        return Chosen(
            Candidate(("x", str(n)), f"S{n}", passage, name=strain, centre=centre, chart=chart),
            "test",
        )

    chosen = [
        s(1, "A", "cell", "lab1", "lab1-hi"),
        s(2, "B", "cell", "lab1", "lab1-hi"),
        s(3, "A", "egg", "lab1", "lab1-hi"),
        s(4, "A", "egg", "lab1", "lab1-neut"),  # another map: not paired with 1
        s(5, "C", "cell", "lab2", "lab2-hi"),
        s(6, "D", "reassortant", "lab3", "lab3-hi"),
    ]
    cols = columns(chosen)
    key = lambda c: c and c.key[1]  # noqa: E731
    assert [(centre, [(key(t), key(b)) for t, b in cc]) for centre, cc in cols] == [
        ("lab1", [("1", "3"), ("2", None), (None, "4")]),
        ("lab2", [("5", None)]),
        ("lab3", [(None, "6")]),
    ]
    laid = [[(key(t), key(b)) for t, b in page] for page in pages(cols)]
    # lab1 fills a page; lab2 and lab3 share the next
    assert laid == [[("1", "3"), ("2", None), (None, "4")], [("5", None), (None, "6")]]


def test_a_part_filled_page_takes_the_first_later_centre_that_fits():
    from af.chart.serum_selection import Column, pages

    def c(n: str) -> Column:
        return (Candidate(("x", n), n, "cell"), None)

    blocks = [("a", [c("a1"), c("a2")]), ("b", [c("b1"), c("b2")]), ("c", [c("c1")])]
    laid = [[t.key[1] if t else None for t, _ in page] for page in pages(blocks)]
    assert laid == [["a1", "a2", "c1"], ["b1", "b2"]]
    big = [("a", [c(f"a{i}") for i in range(7)])]
    assert [len(p) for p in pages(big)] == [3, 3, 1]


def test_candidates_carry_each_serums_strain_passage_and_date():
    from af.chart.model import Antigen, Chart, Serum, Titres
    from af.chart.serum_selection import candidates
    from af.chart.titre import Titre
    from af.map.colouring import SerumStrain

    t = Titre.parse
    sera = [
        Serum("A(H3N2)/EXAMPLETOWN/1/2025", "F1/25", passage="SIAT1"),
        Serum("A(H3N2)/EXAMPLETOWN/1/2025", "F2/25", passage="E3"),
    ]
    chart = Chart({"D": "20260110"}, [Antigen("A(H3N2)/EXAMPLEOTHER/1/2025")], sera,
                  Titres([[t("80"), t("40")]]))  # fmt: skip
    strains = [
        SerumStrain(0, "EPI_1", "A1", "SIAT1", "X.1", ("X", "X.1"), {155: "E"}, "test"),
        SerumStrain(1, None, None, None, None, None, {}, "no sequence record"),
    ]
    cell, egg = candidates(chart, strains, chart_name="lab-a-hi", centre="lab-a")
    assert cell.key == ("lab-a-hi", sera[0].name, "F1/25", "SIAT1", "") and cell.centre == "lab-a"
    assert (cell.passage_class, cell.clades, cell.aa) == (
        "cell",
        frozenset({"X", "X.1"}),
        {155: "E"},
    )
    assert (egg.passage_class, egg.clades, egg.last_titrated) == (
        "egg",
        frozenset(),
        D(2026, 1, 10),
    )
    sel = select_sera([cell, egg], [Section("X.1 155E", clade="X.1", aa=("155E",))], since=None)
    assert [c.serum.key[2] for c in sel.sections["X.1 155E"]] == ["F1/25"]
    twin = Serum(sera[0].name, "F1/25", passage="E3")  # one id for a cell and an egg serum
    pair = Chart(chart.info, chart.antigens, [sera[0], twin], chart.titres)
    keys = [c.key for c in candidates(pair, strains, chart_name="lab-a-hi", centre="lab-a")]
    assert len(set(keys)) == 2
    with pytest.raises(SelectionError, match="1 serum strains for 2 sera"):
        candidates(chart, strains[:1], chart_name="lab-a-hi", centre="lab-a")
