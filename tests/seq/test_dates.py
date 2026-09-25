from __future__ import annotations

import datetime

import pytest

from af.seq.dates import CollectionDate, DateProblem, Precision, parse


class TestParsing:
    def test_full_day(self) -> None:
        date = parse("2024-03-17")
        assert date.precision is Precision.DAY
        assert date.first == date.last == datetime.date(2024, 3, 17)

    def test_month_spans_the_month(self) -> None:
        date = parse("2024-03")
        assert date.precision is Precision.MONTH
        assert (date.first, date.last) == (datetime.date(2024, 3, 1), datetime.date(2024, 3, 31))

    def test_year_spans_the_year(self) -> None:
        date = parse("2024")
        assert date.precision is Precision.YEAR
        assert (date.first, date.last) == (datetime.date(2024, 1, 1), datetime.date(2024, 12, 31))

    @pytest.mark.parametrize("text", ["2024-02", "2023-02", "2000-02", "1900-02"])
    def test_february_length_including_leap_years(self, text: str) -> None:
        date = parse(text)
        expected = {"2024-02": 29, "2023-02": 28, "2000-02": 29, "1900-02": 28}[text]
        assert date.last.day == expected

    def test_december_ends_on_the_31st(self) -> None:
        assert parse("2024-12").last == datetime.date(2024, 12, 31)

    def test_surrounding_whitespace(self) -> None:
        assert parse("  2024-03-17  ") == parse("2024-03-17")


class TestRefusals:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "unknown",
            "2024-13",  # month 13
            "2024-02-31",  # never existed
            "17/03/2024",  # ambiguous order: refused, not guessed
            "2024-3-7",  # unpadded
            "24-03-17",  # two-digit year
            "March 2024",
        ],
    )
    def test_unreadable_dates_raise(self, text: str) -> None:
        with pytest.raises(DateProblem):
            parse(text)

    def test_the_message_says_what_was_expected(self) -> None:
        with pytest.raises(DateProblem, match="YYYY-MM-DD"):
            parse("nonsense")


class TestFloor:
    """Sarah, 25 Sep 2026: the whole interval must be on or after the floor."""

    def test_day_after_the_floor_passes(self) -> None:
        assert parse("2018-04-02").on_or_after("2018-03")

    def test_day_before_the_floor_fails(self) -> None:
        assert not parse("2018-02-27").on_or_after("2018-03")

    def test_month_equal_to_the_floor_passes(self) -> None:
        assert parse("2018-03").on_or_after("2018-03")

    def test_year_only_does_not_slip_past_a_floor_inside_it(self) -> None:
        """The old pipeline read this as 1 January and dropped it for the wrong reason.

        Here it fails because the record does not say the virus was collected after
        March, which is a different statement and a reportable one.
        """
        assert not parse("2018").on_or_after("2018-03")

    def test_year_only_after_the_floor_year_passes(self) -> None:
        assert parse("2019").on_or_after("2018-03")

    def test_floor_may_be_a_date_object(self) -> None:
        floor = parse("2018-03")
        assert parse("2019").on_or_after(floor)


class TestIntervals:
    def test_overlap_is_symmetric(self) -> None:
        year, march = parse("2024"), parse("2024-03")
        assert year.overlaps(march) and march.overlaps(year)

    def test_no_overlap_between_neighbouring_months(self) -> None:
        assert not parse("2024-03").overlaps(parse("2024-04"))

    def test_sorting_is_by_interval_start(self) -> None:
        dates = [parse("2024-06-01"), parse("2023"), parse("2024-01")]
        assert [str(date) for date in sorted(dates)] == ["2023", "2024-01", "2024-06-01"]


def test_str_round_trips_through_parse() -> None:
    for text in ("2024-03-17", "2024-03", "2024"):
        assert str(parse(text)) == text


def test_precision_is_carried_not_inferred() -> None:
    """A year-only date must stay distinguishable from a real 1 January.

    One lab submits year-only dates, and treating them as 1 January put a false spike
    on the tree's time axis that had to be hidden by hand.
    """
    stated = parse("2024-01-01")
    unknown = parse("2024")
    assert stated.first == unknown.first
    assert stated.precision is not unknown.precision
    assert stated != unknown


def test_dates_are_hashable_and_comparable() -> None:
    assert len({parse("2024"), parse("2024"), parse("2024-01")}) == 2
    assert isinstance(parse("2024"), CollectionDate)


class TestMidpoint:
    def test_a_full_day_is_its_own_midpoint(self) -> None:
        assert parse("2024-03-17").midpoint == datetime.date(2024, 3, 17)

    def test_year_only_sits_mid_year_not_on_1_january(self) -> None:
        """1 January is a date nobody stated, and it biases every such point one way."""
        assert parse("2024").midpoint == datetime.date(2024, 7, 1)

    def test_month_only_sits_mid_month(self) -> None:
        assert parse("2024-03").midpoint == datetime.date(2024, 3, 16)
