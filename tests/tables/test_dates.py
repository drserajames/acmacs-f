"""Lab dates: the lab's order decides, impossible dates are read the one valid way and flagged."""

from __future__ import annotations

import datetime as dt

import pytest

from af.tables import dates


@pytest.mark.parametrize(
    ("text", "order", "iso", "warned"),
    [
        ("9/12/2030", "MDY", "2030-09-12", False),
        ("9/12/2030", "DMY", "2030-12-09", False),
        ("2030/09/16", "MDY", "2030-09-16", False),
        ("2030-09-16 00:00:00", "DMY", "2030-09-16", False),
        ("17/02/30", "MDY", "2030-02-17", True),
    ],
)
def test_parse(text, order, iso, warned):
    value, warning = dates.parse(text, order)
    assert value == iso and (warning is not None) == warned


def test_invalid_everywhere_is_an_error():
    with pytest.raises(dates.DateError):
        dates.parse("31/31/2030", "MDY")


def test_two_digit_year_pivots_on_the_table_date_not_today():
    assert dates.parse("1/2/95", "MDY", not_after=dt.date(1996, 1, 1))[0] == "1995-01-02"
    assert dates.parse("1/2/25", "MDY", not_after=dt.date(2030, 1, 1))[0] == "2025-01-02"


@pytest.mark.parametrize(
    ("text", "iso"),
    [
        ("11/15/36", "2036-11-15"),  # after the test: stays 20yy, for the reader to report
        ("1/1/87", "1987-01-01"),  # an old reference strain's date
    ],
)
def test_a_two_digit_year_after_the_test_is_not_moved_a_century(text, iso):
    assert dates.parse(text, "MDY", not_after=dt.date(2035, 12, 3))[0] == iso
