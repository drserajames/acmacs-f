"""A map's subtype row comes from the chart's own data, never a "B means B/Vic" default."""

import pytest

from af.chart.model import Antigen, Chart, Titres
from af.map.build import BuildError, chart_subtype, lineage_minority, map_title


def chart(virus: str, lineages: list[str]) -> Chart:
    antigens = [
        Antigen("/".join(("B", "OLDTOWN", str(i + 1), "2021")), extra={"L": c} if c else {})
        for i, c in enumerate(lineages)
    ]
    return Chart({"V": virus, "l": "LAB", "A": "HI"}, antigens, [], Titres([[] for _ in antigens]))


def test_b_chart_takes_its_antigens_lineage() -> None:
    row = chart_subtype(chart("B", ["V", "V"]))
    assert row.name == "B/Vic" and row.acmacs_data == "BV"
    assert map_title(chart("B", ["V"]), None) == "LAB B/Vic by clade"


def test_a_chart_has_no_lineage_code() -> None:
    assert chart_subtype(chart("A(H3N2)", ["", ""])).name == "A(H3N2)"


def test_a_few_of_another_lineage_are_reported_not_fatal() -> None:
    c = chart("B", ["V", "V", "V", "Y"])
    row = chart_subtype(c)
    assert row.name == "B/Vic"
    assert lineage_minority(c, row) == {
        "map_lineage": "V",
        "other": {"Y": 1},
        "examples": ["/".join(("B", "OLDTOWN", "4", "2021"))],
    }
    assert lineage_minority(chart("B", ["V"]), row) == {}


def test_lineages_carried_equally_are_an_error() -> None:
    with pytest.raises(BuildError, match="lineages equally"):
        chart_subtype(chart("B", ["V", "Y"]))


def test_b_chart_without_lineage_is_an_error_not_a_default() -> None:
    with pytest.raises(BuildError, match="lineage code"):
        chart_subtype(chart("B", ["", ""]))
