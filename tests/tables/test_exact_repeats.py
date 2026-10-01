"""A test entered twice in a lab's export (two test ids, identical content) is read once."""

from __future__ import annotations

from af.tables.update import exact_repeats

from .test_cdc import read, row


def test_the_later_of_two_identical_tests_is_dropped(tmp_path):
    rows = [row(test_id="1"), row(test_id="2"), row(test_id="3", titer_value="320")]
    tables = read(tmp_path, rows).tables
    assert len(tables) == 3
    kept, report = exact_repeats(tables)
    assert [t.source_key for t in kept] == ["CDC test_id 1", "CDC test_id 3"]
    assert report == ["CDC test_id 2 repeats CDC test_id 1"]


def test_tests_that_differ_in_a_reading_are_both_kept(tmp_path):
    tables = read(tmp_path, [row(test_id="1"), row(test_id="2", titer_value="320")]).tables
    assert exact_repeats(tables) == (tables, [])
