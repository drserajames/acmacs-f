"""Passage and name normalisation, on invented values."""

from __future__ import annotations

import pytest

from af.tables import names
from af.tables.passage import PassageParser
from af.tables.rules import Rules


@pytest.fixture
def parser(rules: Rules) -> PassageParser:
    return PassageParser(rules.passage_tokens, "CDC")


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("C1", "MDCK1"),
        ("S1C4/C2", "SIAT1MDCK4/MDCK2"),
        ("E3E9/E1", "E3/E9/E1"),
        ("CXC4/C2", "MDCK?/MDCK4/MDCK2"),
        ("Spf2E1E10", "SPF2E1/E10"),
        ("E3D6/E2", "E3D6/E2"),
        ("QMCX", "QMC?"),
        ("X3/S1", "X3/SIAT1"),
        ("hCK2/C1", "HCK2/MDCK1"),
        ("AX41hCK2/S1", "AX41HCK2/SIAT1"),
        ("NC2NC2", "NC2/NC2"),
        ("", ""),
    ],
)
def test_passage_canonical_like_ae(parser, raw, canonical):
    p = parser.parse(raw)
    assert (p.text, p.problems) == (canonical, [])


def test_unreadable_passage_is_reported_not_guessed(parser):
    p = parser.parse("C1 ZZZ")
    assert p.text == "C1 ZZZ" and p.problems


def test_egg_class(parser):
    assert parser.parse("E3/E1").is_egg and not parser.parse("S1").is_egg


@pytest.mark.parametrize(
    ("raw", "subtype", "name", "reassortant", "annotations"),
    [
        ("A/EXAMPLETOWN/01/2029", "A(H1N1)", "A(H1N1)/EXAMPLETOWN/1/2029", "", []),
        ("B/EXAMPLEB/5/2029 BX-99C", "B", "B/EXAMPLEB/5/2029", "NYMC-99C", []),
        ("A/EXAMPLETOWN/9/2029 IVR-999", "A(H3N2)", "A(H3N2)/EXAMPLETOWN/9/2029", "IVR-999", []),
        ("B/EXAMPLETOWN/2/2029 (V1)", "B", "B/EXAMPLETOWN/2/2029", "", ["V1"]),
        ("B/EXAMPLETOWN/77/2029 (29/228)", "B", "B/EXAMPLETOWN/77/2029", "", ["29/228"]),
        (
            "A/EXAMPLETOWN/9/2029 CNIC-9999 (29/214)",
            "A(H3N2)",
            "A(H3N2)/EXAMPLETOWN/9/2029",
            "CNIC-9999",
            ["29/214"],
        ),
    ],
)
def test_names(rules, raw, subtype, name, reassortant, annotations):
    n = names.parse(raw, subtype, rules.reassortants, "CDC")
    assert (n.name, n.reassortant, n.annotations, n.problems) == (
        name,
        reassortant,
        annotations,
        [],
    )


def test_wrong_type_is_kept_and_reported_t40(rules):
    n = names.parse("A/EXAMPLETOWN/1/2029", "B", rules.reassortants, "CDC")
    assert n.name.startswith("A/") and n.problems


def test_cyrillic_type_and_two_digit_year(rules):
    n = names.parse("А/EXAMPLETOWN/5/29", "A(H3N2)", rules.reassortants, "CDC", not_after=2030)
    assert n.name == "A(H3N2)/EXAMPLETOWN/5/2029"
    assert any("Cyrillic" in p for p in n.problems) and any(
        "two-digit year" in p for p in n.problems
    )


def test_two_digit_year_pivots_on_the_test_year_and_is_reported_without_one(rules):
    assert names.parse(
        "A/EXAMPLETOWN/5/95", "A(H3N2)", rules.reassortants, "CDC", not_after=2030
    ).name.endswith("/1995")
    unpivoted = names.parse("A/EXAMPLETOWN/5/29", "A(H3N2)", rules.reassortants, "CDC")
    assert any("four-digit year" in p for p in unpivoted.problems)
