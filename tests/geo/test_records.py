import datetime

import pytest

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
        ("A(H3N2)", Month(2021, 1), "Alpha"): 2,
        ("A(H3N2)", Month(2021, 2), "Beta"): 1,
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
        {"name": "Alpha", "points": [{"color": "unassigned", "count": 2}]}
    ]
    assert doc["periods"][1]["locations"] == []
    result = compare_geo.compare(doc, doc)
    assert result["per_month"]["2021-01"]["location"]["abs_diff"] == 0
