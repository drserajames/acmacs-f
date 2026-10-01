"""af.util.subtypes: the packaged table, its lookups, and what the loader refuses."""

import copy
import tomllib
from importlib import resources
from typing import Any

import pytest

from af.util.subtypes import SubtypeError, Subtypes, subtypes


def packaged() -> dict[str, Any]:
    return tomllib.loads(resources.files("af").joinpath("subtypes.toml").read_text())


def problems_of(data: dict[str, Any]) -> str:
    with pytest.raises(SubtypeError) as error:
        Subtypes(data, "test")
    return str(error.value)


# ---- the packaged table: the values every switched caller relied on -------------------


def test_packaged_table_loads_in_file_order() -> None:
    assert subtypes().keys() == ("h1", "h3", "bvic", "byam")
    assert [row.key for row in subtypes()] == ["h1", "h3", "bvic", "byam"]


def test_names_and_acmacs_data_keys() -> None:
    table = subtypes()
    assert {row.key: table.by_key(row.key).name for row in table} == {
        "h1": "A(H1N1)",
        "h3": "A(H3N2)",
        "bvic": "B/Vic",
        "byam": "B/Yam",
    }
    assert [table.by_acmacs_data(k).key for k in ("A(H1N1)", "A(H3N2)", "BV", "BY")] == [
        "h1",
        "h3",
        "bvic",
        "byam",
    ]
    assert table.by_name("B/Vic").acmacs_data == "BV"


@pytest.mark.parametrize(
    ("subtype", "lineage", "prefix"),
    [
        ("A(H1N1)", "", "h1pdm"),
        ("A(H3N2)", "", "h3"),
        ("B", "VICTORIA", "bvic"),
        ("B", "YAMAGATA", "byam"),
        ("B", "", "b"),
    ],
)
def test_group_prefix_matches_the_readers(subtype: str, lineage: str, prefix: str) -> None:
    assert subtypes().group_prefix(subtype, lineage) == prefix


def test_for_table_keeps_file_order_for_an_unknown_b_lineage() -> None:
    table = subtypes()
    assert [row.key for row in table.for_table("B", "")] == ["bvic", "byam"]
    assert [row.key for row in table.for_table("B", "YAMAGATA")] == ["byam"]
    assert [row.key for row in table.for_table("A(H3N2)", "")] == ["h3"]


def test_geo_and_split_by_lineage_belong_to_the_table_subtype() -> None:
    table = subtypes()
    assert {name: table.table_subtype(name).geo for name in ("A(H1N1)", "A(H3N2)", "B")} == {
        "A(H1N1)": "h1",
        "A(H3N2)": "h3",
        "B": "b",
    }
    assert table.split_by_lineage() == ("B",)


def test_for_chart_reads_the_ace_lineage_code() -> None:
    table = subtypes()
    assert table.by_key("bvic").ace_lineage == "V"
    assert table.by_key("h3").ace_lineage == ""
    assert table.for_chart("B", "Y").key == "byam"
    assert table.for_chart("A(H3N2)", "").key == "h3"


# ---- lookups that match nothing are errors, naming what exists -------------------------


@pytest.mark.parametrize(
    ("call", "names"),
    [
        (lambda t: t.by_key("h5"), "'h1'"),
        (lambda t: t.by_name("B/Victoria"), "'B/Vic'"),
        (lambda t: t.by_acmacs_data("B"), "'BV'"),
        (lambda t: t.table_subtype("A(H5N1)"), "A(H1N1)"),
        (lambda t: t.for_table("A(H3N2)", "VICTORIA"), "''"),
        (lambda t: t.for_table("B", "Victoria"), "'VICTORIA'"),
        (lambda t: t.group_prefix("C", ""), "B"),
        (lambda t: t.for_chart("B", ""), "'V'"),
        (lambda t: t.for_chart("B", "X"), "'Y'"),
    ],
)
def test_a_miss_is_an_error_that_lists_what_exists(call: Any, names: str) -> None:
    with pytest.raises(SubtypeError, match="af/subtypes.toml") as error:
        call(subtypes())
    assert names in str(error.value)


# ---- the loader refuses an inconsistent table, reporting every problem ----------------


def test_sections_pass_through_raw() -> None:
    data = packaged()
    data["subtype"]["h3"]["clades"] = {"labels": True, "anything": [1, 2]}
    data["subtype"]["h1"].pop("clades", None)  # a row without sections, whatever the package has
    table = Subtypes(data, "test")
    assert table.by_key("h3").sections == {"clades": {"labels": True, "anything": [1, 2]}}
    assert table.by_key("h1").sections == {}


def test_all_problems_reported_together() -> None:
    data = packaged()
    data["subtype"]["h3"]["colour"] = "red"  # unknown key
    del data["subtype"]["h1"]["group_prefix"]  # missing key
    data["subtype"]["byam"]["acmacs_data"] = "BV"  # duplicate
    message = problems_of(data)
    assert "subtype.h3: unknown key 'colour'" in message
    assert "subtype.h1: missing group_prefix" in message
    assert "acmacs_data must be unique; repeated: 'BV'" in message


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda d: d["subtype"]["byam"].update(lineage="VICTORIA"), "(table_subtype, lineage)"),
        (lambda d: d["subtype"]["byam"].update(lineage="VIC2"), ".ace lineage code"),
        (lambda d: d["subtype"]["byam"].update(lineage=""), "each needs a lineage"),
        (
            lambda d: d["table_subtype"]["B"].pop("unresolved_group_prefix"),
            "needs unresolved_group_prefix",
        ),
        (
            lambda d: d["table_subtype"]["A(H3N2)"].update(unresolved_group_prefix="x"),
            "only for subtypes",
        ),
        (lambda d: d["subtype"]["h3"].update(lineage="X"), "must have lineage"),
        (lambda d: d["subtype"]["h3"].update(table_subtype="H3"), "has no [table_subtype] entry"),
        (lambda d: d["table_subtype"].update({"C": {"geo": "c"}}), "no [subtype] row"),
        (lambda d: d["table_subtype"]["B"].pop("geo"), "missing geo"),
        (lambda d: d["subtype"]["h3"].update(group_prefix="b"), "also an unresolved_group_prefix"),
        (lambda d: d["subtype"]["h3"].update(name=""), "non-empty string"),
        (lambda d: d["subtype"]["h3"].update(name=3), "non-empty string"),
        (lambda d: d.update(subtypes={}), "unknown top-level key 'subtypes'"),
        (lambda d: d.pop("subtype"), "missing [subtype.*]"),
    ],
)
def test_inconsistent_table_refused(edit: Any, expected: str) -> None:
    data = copy.deepcopy(packaged())
    edit(data)
    assert expected in problems_of(data)
