import datetime

from af.geo.records import Month
from af.serology.query import Preparation, SerumRecord
from af.stat.counts import ALL, UNKNOWN, stat_counts

d = datetime.date


def prep(
    name: str,
    lab: str,
    collected: datetime.date | None,
    subtype: str = "B",
    lineage: str = "VICTORIA",
) -> Preparation:
    return Preparation(subtype, lineage, name, "", (), "MDCK1", collected, lab, d(2021, 6, 1))


def serum(key: str, name: str, lab: str, collected: datetime.date | None) -> SerumRecord:
    return SerumRecord("B", "VICTORIA", key, name, lab, collected)


def location(name: str) -> str | None:
    return name.split("-")[0]


CONTINENTS = {"Alpha": "CONTINENT-1", "Beta": "CONTINENT-2"}


def test_counts_roll_up_every_axis_and_split_lineage() -> None:
    counts = stat_counts(
        [prep("Alpha-1", "LABX", d(2021, 1, 9)), prep("Beta-2", "LABY", d(2021, 2, 3))],
        [],
        Month(2021, 1),
        Month(2021, 3),
        location,
        CONTINENTS.get,
        split_by_lineage={"B"},
    )
    a = counts.antigens
    assert a["B", "LABX", "2021-01", "CONTINENT-1"] == 1
    assert a["B", ALL, "2021", ALL] == 2
    assert a[ALL, ALL, ALL, ALL] == 2
    assert a["B/VICTORIA", "LABY", "2021-02", "CONTINENT-2"] == 1
    assert (ALL, "LABX", "2021-01", ALL) in a
    assert not any(k[0] == ALL and k[1] == "B/VICTORIA" for k in a)


def test_window_undated_and_unknown_continent_are_reported() -> None:
    counts = stat_counts(
        [
            prep("Alpha-1", "LABX", d(2020, 12, 30)),  # outside
            prep("Alpha-2", "LABX", None),  # undated
            prep("Gamma-3", "LABX", d(2021, 1, 5)),  # no continent
        ],
        [],
        Month(2021, 1),
        Month(2021, 1),
        location,
        CONTINENTS.get,
    )
    assert counts.antigens[ALL, ALL, ALL, ALL] == 1
    assert counts.antigens["B", "LABX", "2021-01", UNKNOWN] == 1
    assert counts.undated == {"antigens": 1}
    assert counts.unknown_continent == {"Gamma": 1}


def test_sera_dated_by_strain_and_deduplicated_by_name() -> None:
    counts = stat_counts(
        [],
        [
            serum("k1", "Alpha-1", "LABX", d(2021, 1, 2)),
            serum("k2", "Alpha-1", "LABY", d(2021, 1, 2)),  # same strain, another serum
            serum("k3", "Beta-2", "LABX", None),  # strain never dated
        ],
        Month(2021, 1),
        Month(2021, 1),
        location,
        CONTINENTS.get,
    )
    assert counts.sera_unique[ALL, ALL, ALL, ALL] == 2
    assert counts.sera[ALL, ALL, ALL, ALL] == 1
    assert counts.undated == {"sera": 1}
