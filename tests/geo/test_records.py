import datetime

import pytest

from af.geo.colours import UNCOLOURED
from af.geo.records import Month, geo_counts, months, to_i7
from af.report.compare import geo as compare_geo
from af.serology.query import Preparation


def prep(
    name: str, passage: str, collected: datetime.date | None, lab: str = "LABX"
) -> Preparation:
    return Preparation(
        subtype="A(H3N2)",
        lineage="",
        name=name,
        reassortant="",
        annotations=(),
        passage=passage,
        collection_date=collected,
        first_lab=lab,
        first_table_date=datetime.date(2021, 6, 1),
    )


def place(name: str) -> str | None:
    """Invented names are 'Place-N'; 'Nowhere-N' has no location."""
    return None if name.startswith("Nowhere") else name.split("-")[0]


def test_months_cross_a_year_and_refuse_backwards() -> None:
    assert [str(m) for m in months(Month(2020, 11), Month(2021, 2))] == [
        "2020-11",
        "2020-12",
        "2021-01",
        "2021-02",
    ]
    with pytest.raises(ValueError):
        months(Month(2021, 2), Month(2020, 11))


def test_one_dot_per_preparation_and_nothing_dropped_silently() -> None:
    d = datetime.date
    preps = [
        prep("Alpha-1", "E3", d(2021, 1, 9)),
        prep("Alpha-1", "MDCK1", d(2021, 1, 9)),  # same virus, second preparation: second dot
        prep("Beta-2", "MDCK1", d(2021, 2, 1)),
        prep("Beta-3", "MDCK1", d(2020, 12, 31)),  # before the window
        prep("Nowhere-4", "MDCK1", d(2021, 1, 2)),  # no location
        prep("Alpha-5", "MDCK1", None),  # no date
    ]
    result = geo_counts(preps, Month(2021, 1), Month(2021, 3), place)
    assert result.dots == {
        ("A(H3N2)", Month(2021, 1), "Alpha", UNCOLOURED): 2,
        ("A(H3N2)", Month(2021, 2), "Beta", UNCOLOURED): 1,
    }
    assert [str(m) for m in result.months] == ["2021-01", "2021-02", "2021-03"]
    assert result.undated == {"A(H3N2)": 1}
    assert result.no_location == {("A(H3N2)", "Nowhere-4"): 1}


def test_i7_lists_every_month_and_compares_equal_to_itself() -> None:
    d = datetime.date
    preps = [prep("Alpha-1", "E3", d(2021, 1, 9)), prep("Alpha-2", "E3", d(2021, 1, 10))]
    doc = to_i7(geo_counts(preps, Month(2021, 1), Month(2021, 2), place), "A(H3N2)")
    assert [p["period"] for p in doc["periods"]] == ["2021-01", "2021-02"]
    assert doc["periods"][0]["locations"] == [
        {"name": "Alpha", "points": [{"color": "transparent", "count": 2}]}
    ]
    assert doc["periods"][1]["locations"] == []
    result = compare_geo.compare(doc, doc)
    assert result["per_month"]["2021-01"]["location"]["abs_diff"] == 0


def test_one_dot_per_virus_merges_its_preparations_and_says_how() -> None:
    """Sarah, 7 Oct: one dot per virus (egg and cell merged), in the month of the earliest
    date any preparation reports; its colour is the one its coloured preparations agree on,
    else its cell preparations', else none. Every case is counted."""
    from af.geo.colours import DotStyle
    from af.geo.records import (
        AGREE,
        CELL_CHOSEN,
        COLOURED_OVER_UNCOLOURED,
        DISAGREE,
        SINGLE,
        DotRule,
    )  # fmt: skip

    d = datetime.date
    red, blue = DotStyle("Clade R", "#aa0000"), DotStyle("Clade B", "#0000aa")
    preps = [
        prep("Alpha-1", "E3", d(2021, 2, 9)),  # egg and cell, same clade: one red dot
        prep("Alpha-1", "MDCK1", d(2021, 1, 20)),  # ... dated by the earlier, January
        prep("Beta-2", "E3", d(2021, 2, 1)),  # egg red, cell blue: the cell's colour
        prep("Beta-2", "MDCK1", d(2021, 2, 1)),
        prep("Gamma-3", "MDCK1", d(2021, 2, 1)),  # two cell preparations disagree: none
        prep("Gamma-3", "SIAT1", d(2021, 2, 1)),
        prep("Delta-4", "MDCK1", d(2021, 3, 1)),  # coloured + uncoloured: the colour
        prep("Delta-4", "SIAT1", d(2021, 3, 1)),
        prep("Eps-5", "SIAT1", d(2021, 3, 1)),  # a single preparation
    ]
    colour = {
        ("Alpha-1", "E3"): red, ("Alpha-1", "MDCK1"): red,
        ("Beta-2", "E3"): red, ("Beta-2", "MDCK1"): blue,
        ("Gamma-3", "MDCK1"): red, ("Gamma-3", "SIAT1"): blue,
        ("Delta-4", "MDCK1"): blue, ("Delta-4", "SIAT1"): UNCOLOURED,
        ("Eps-5", "SIAT1"): red,
    }  # fmt: skip

    def style_of(p: Preparation) -> DotStyle:
        return colour[p.name, p.passage]

    def class_of(p: Preparation) -> str:
        return "egg" if p.passage.startswith("E") else "cell"

    result = geo_counts(
        preps, Month(2021, 1), Month(2021, 3), place, DotRule.VIRUS, style_of, class_of
    )
    assert result.dots == {
        ("A(H3N2)", Month(2021, 1), "Alpha", red): 1,
        ("A(H3N2)", Month(2021, 2), "Beta", blue): 1,
        ("A(H3N2)", Month(2021, 2), "Gamma", UNCOLOURED): 1,
        ("A(H3N2)", Month(2021, 3), "Delta", blue): 1,
        ("A(H3N2)", Month(2021, 3), "Eps", red): 1,
    }
    assert result.merges == {
        "A(H3N2)": {AGREE: 1, CELL_CHOSEN: 1, DISAGREE: 1, COLOURED_OVER_UNCOLOURED: 1, SINGLE: 1}
    }
    assert result.merged_preparations == {"A(H3N2)": 9}
    with pytest.raises(ValueError, match="class_of"):
        geo_counts(preps, Month(2021, 1), Month(2021, 3), place, DotRule.VIRUS, style_of)


def test_unpainted_viruses_are_counted_per_month_and_clade_not_drawn_apart() -> None:
    """Sarah, 8 Oct: dots left unpainted by clade evidence are warned about, never silent.
    Counted per virus dot: a virus is unpainted when none of its preparations is coloured and
    at least one was left so by clade evidence; its other preparations are not counted as such.
    On the map an unpainted dot looks like any uncoloured one, so they are one point group."""
    from af.geo.colours import STATE_NO_SCHEME_ROW, STATE_NO_SUPPORTED, DotStyle
    from af.geo.records import DotRule

    d = datetime.date
    red = DotStyle("Clade R", "#aa0000")
    gap = DotStyle("", None, state=STATE_NO_SCHEME_ROW, clade="Q.1", tree_clade="Q.1.2")
    none = DotStyle("", None, state=STATE_NO_SUPPORTED, tree_clade="Q.3")
    preps = [
        prep("Alpha-1", "E3", d(2021, 1, 9)),  # both preparations unpainted for Q.1: 1 dot
        prep("Alpha-1", "MDCK1", d(2021, 1, 9)),
        prep("Alpha-2", "MDCK1", d(2021, 1, 9)),  # unpainted Q.1 + no sequence: 1 dot, 1 prep
        prep("Alpha-2", "E3", d(2021, 1, 9)),
        prep("Alpha-3", "MDCK1", d(2021, 1, 9)),  # no sequence at all: not unpainted
        prep("Beta-4", "E3", d(2021, 1, 9)),  # unpainted, but its cell preparation is red
        prep("Beta-4", "MDCK1", d(2021, 1, 9)),
        prep("Beta-5", "MDCK1", d(2021, 2, 9)),  # no supported clade, in February
        prep("Nowhere-6", "MDCK1", d(2021, 2, 9)),  # not drawn: not counted
    ]
    colour = {
        ("Alpha-1", "E3"): gap, ("Alpha-1", "MDCK1"): gap,
        ("Alpha-2", "MDCK1"): gap, ("Alpha-2", "E3"): UNCOLOURED,
        ("Alpha-3", "MDCK1"): UNCOLOURED,
        ("Beta-4", "E3"): gap, ("Beta-4", "MDCK1"): red,
        ("Beta-5", "MDCK1"): none, ("Nowhere-6", "MDCK1"): gap,
    }  # fmt: skip
    result = geo_counts(
        preps, Month(2021, 1), Month(2021, 2), place, DotRule.VIRUS,
        lambda p: colour[p.name, p.passage], lambda p: "egg" if p.passage[0] == "E" else "cell",
    )  # fmt: skip
    doc = to_i7(result, "A(H3N2)")
    january, february = doc["periods"]
    assert january["unpainted"] == [
        {"clade": "Q.1", "tree_clades": {"Q.1.2": 2}, "points": 2, "preparations": 3}
    ]
    assert february["unpainted"] == [
        {"clade": None, "tree_clades": {"Q.3": 1}, "points": 1, "preparations": 1}
    ]
    alpha = next(loc for loc in january["locations"] if loc["name"] == "Alpha")
    assert alpha["points"] == [{"color": "transparent", "count": 3}]  # one group, not two


def test_viruses_painted_by_placement_are_counted_per_month() -> None:
    """Q183 b: a dot painted by tree placement alone is drawn in its colour and counted for the
    flag; a virus whose other preparation is painted by evidence is not flagged."""
    from af.geo.colours import BASIS_PLACEMENT, DotStyle
    from af.geo.records import DotRule

    d = datetime.date
    placed = DotStyle("Clade R", "#aa0000", BASIS_PLACEMENT, tree_clade="R")
    red = DotStyle("Clade R", "#aa0000")
    preps = [
        prep("Alpha-1", "MDCK1", d(2021, 1, 9)),  # placed: flagged
        prep("Alpha-2", "E3", d(2021, 1, 9)),  # egg placed, cell by evidence: cell chosen
        prep("Alpha-2", "MDCK1", d(2021, 1, 9)),
    ]
    colour = {("Alpha-1", "MDCK1"): placed, ("Alpha-2", "E3"): placed,
              ("Alpha-2", "MDCK1"): red}  # fmt: skip
    result = geo_counts(
        preps, Month(2021, 1), Month(2021, 1), place, DotRule.VIRUS,
        lambda p: colour[p.name, p.passage], lambda p: "egg" if p.passage[0] == "E" else "cell",
    )  # fmt: skip
    (january,) = to_i7(result, "A(H3N2)")["periods"]
    assert january["painted_by_placement"] == [
        {"tree_clades": {"R": 1}, "points": 1, "preparations": 1}
    ]
    assert january["unpainted"] == []
    alpha = next(loc for loc in january["locations"] if loc["name"] == "Alpha")
    assert alpha["points"] == [{"clade": "Clade R", "color": "#aa0000", "count": 2}]


def test_a_placement_painted_preparation_abstains_beside_evidence() -> None:
    """Q183 b: a virus whose preparations are painted alike, one by evidence and one by tree
    placement alone, takes the evidence's dot (unflagged); it does not become a disagreement.
    With only placement-painted preparations, the dot keeps the flag."""
    from af.geo.colours import BASIS_PLACEMENT, BASIS_SUPPORTED_CLADE, DotStyle
    from af.geo.records import AGREE, merged_style

    placed = DotStyle("Clade R", "#aa0000", BASIS_PLACEMENT, tree_clade="R")
    supported = DotStyle("Clade R", "#aa0000", BASIS_SUPPORTED_CLADE)
    assert merged_style([(placed, "egg"), (supported, "egg"), (placed, "cell")]) == (
        supported, AGREE
    )  # fmt: skip
    assert merged_style([(placed, "egg"), (placed, "cell")]) == (placed, AGREE)
