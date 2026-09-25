"""passage_class: one class per passage history, stored on antigens and sera (af-table-2)."""

from __future__ import annotations

import dataclasses

import pytest

from af.tables.model import FORMAT, Table
from af.tables.passage import PassageParser, class_of

from .test_cdc import read, row


@pytest.mark.parametrize(
    ("classes", "expected"),
    [
        (["cell", "egg"], "egg"),
        (["cell", "cell"], "cell"),
        (["original", "cell"], "cell"),
        (["original"], "original"),
        (["unknown"], "unknown"),
        ([], "unknown"),
    ],
)
def test_class_of(classes, expected):
    assert class_of(classes) == expected


@pytest.mark.parametrize(
    ("canonical", "expected"),
    [
        ("E3/E1", "egg"),
        ("MDCK1SIAT1/SIAT1", "cell"),
        ("MDCK?/HCK2", "cell"),
        ("AX41HCK2", "cell"),
        ("MDCK2/E1", "egg"),
        ("X3/SIAT1", "cell"),
        ("NC2", "unknown"),
        ("", "unknown"),
        ("SOMETHING ELSE", "unknown"),  # a passage kept as written because it did not parse
    ],
)
def test_passage_class_of_canonical_text(rules, canonical, expected):
    assert PassageParser(rules.passage_tokens, "CDC").passage_class(canonical) == expected


def test_readers_fill_it(tmp_path):
    (table,) = read(tmp_path, [row()]).tables
    assert (table.antigens[0].passage_class, table.sera[0].passage_class) == ("cell", "cell")


def test_it_is_not_part_of_the_map(tmp_path):
    (table,) = read(tmp_path, [row()]).tables
    other = dataclasses.replace(
        table, antigens=[dataclasses.replace(table.antigens[0], passage_class="egg")]
    )
    assert other.map_hash() == table.map_hash()
    assert other.content_hash() != table.content_hash()


def test_an_af_table_1_file_still_reads(tmp_path):
    (table,) = read(tmp_path, [row()]).tables
    old = table.to_json()
    old["format"] = "af-table-1"
    for item in (*old["antigens"], *old["sera"]):
        del item["passage_class"]
    for antigen in table.antigens:
        antigen.passage_class = ""
    for serum in table.sera:
        serum.passage_class = ""
    old["content_hash"] = table.content_hash_as("af-table-1")
    back = Table.from_json(old)
    assert back.antigens[0].passage_class == ""
    assert back.to_json()["format"] == FORMAT
    old["content_hash"] = "0" * 64
    with pytest.raises(ValueError, match="stored hash"):
        Table.from_json(old)


def test_an_unknown_format_is_refused(tmp_path):
    (table,) = read(tmp_path, [row()]).tables
    d = table.to_json()
    d["format"] = "af-table-0"
    with pytest.raises(ValueError, match="not an af-table"):
        Table.from_json(d)
