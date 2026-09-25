import datetime

from af.geo.records import Month
from af.serology.query import Preparation, SerumUse
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


def use(key: str, lab: str, day: datetime.date, table: str, suffix: int = 1) -> SerumUse:
    """Serum ``key`` (strain 'Alpha-1') used in ``table`` on ``day``."""
    return SerumUse("B", "VICTORIA", key, "Alpha-1", table, lab, day, suffix)


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


def test_new_sera_count_once_at_first_use_used_sera_count_distinct_per_cell() -> None:
    uses = [
        use("k1", "LABY", d(2021, 1, 20), "t2"),
        use("k1", "LABX", d(2021, 1, 20), "t1"),  # same day: the tie goes to LABX (then table)
        use("k1", "LABX", d(2021, 2, 3), "t3"),  # used again next month
        use("k2", "LABX", d(2020, 12, 1), "t0"),  # first used before the window...
        use("k2", "LABX", d(2021, 2, 9), "t4"),  # ...and used inside it
    ]
    counts = stat_counts([], uses, Month(2021, 1), Month(2021, 2), location, CONTINENTS.get)
    new, used = counts.sera_new, counts.sera_used
    # new: k1 once, in January, for LABX; k2 was new before the window
    assert new[ALL, ALL, ALL, ALL] == 1
    assert new["B", "LABX", "2021-01", "CONTINENT-1"] == 1
    assert new[ALL, "LABY", ALL, ALL] == 0
    # used: distinct sera per cell, never a sum of months or labs
    assert used[ALL, ALL, "2021-01", ALL] == 1
    assert used[ALL, ALL, "2021-02", ALL] == 2
    assert used[ALL, ALL, "2021", ALL] == 2
    assert used[ALL, ALL, ALL, ALL] == 2
    assert used[ALL, "LABY", ALL, ALL] == 1
    assert used[ALL, "LABX", ALL, ALL] == 2
