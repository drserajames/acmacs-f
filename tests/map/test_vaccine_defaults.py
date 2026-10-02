"""Subtype-wide vaccine defaults: shared by every round, optional unless the file says otherwise."""

from pathlib import Path

from af.map.build import load_vaccine_defaults, vaccine_rule_records
from af.map.vaccines import VaccineChoice, VaccineDisable

# Invented names, built rather than written out (see tools/WHO-DATA-GATE.md).
FIRST = "/".join(("OLDTOWN", "1", "2009"))
SECOND = "/".join(("OLDTOWN", "2", "2010"))


def test_keyed_by_subtype_and_optional_by_default(tmp_path: Path) -> None:
    path = tmp_path / "vaccine-defaults.toml"
    path.write_text(
        f'[[disable."A(H1N1)"]]\nname = "{FIRST}"\nreason = "superseded"\n\n'
        f'[[disable."A(H3N2)"]]\nname = "{SECOND}"\nreason = "superseded"\n'
        'passage = "egg"\noptional = false\n'
    )
    defaults = load_vaccine_defaults(path)
    assert set(defaults) == {"A(H1N1)", "A(H3N2)"}
    h1 = defaults["A(H1N1)"][0]
    # Optional unless the file says otherwise: not every lab tested every superseded vaccine,
    # so a subtype default matching nothing in one chart is normal, not an error.
    assert h1.passage == "any" and h1.optional is True
    h3 = defaults["A(H3N2)"][0]
    assert h3.passage == "egg" and h3.optional is False


def test_a_default_may_record_when_it_was_decided(tmp_path: Path) -> None:
    path = tmp_path / "vaccine-defaults.toml"
    path.write_text(
        f'[[disable."A(H1N1)"]]\nname = "{FIRST}"\nreason = "superseded"\ndecided = 2026-09-01\n'
        f'[[disable."A(H1N1)"]]\nname = "{SECOND}"\nreason = "superseded"\n'
    )
    dated, undated = load_vaccine_defaults(path)["A(H1N1)"]
    assert dated.decided == "2026-09-01"
    assert undated.decided is None  # older rules carry no date; the figure records null


def test_no_file_means_no_defaults() -> None:
    assert load_vaccine_defaults(None) == {}


def test_every_vaccine_rule_is_recorded_with_its_scope_and_use() -> None:
    """The figure lists each rule it was built with, so a report can say what was decided
    without reading the config: subtype defaults first, then the map's own."""
    default = VaccineDisable(FIRST, "any", "superseded", optional=True)
    own = VaccineDisable(SECOND, "egg", "lab asked", decided="2026-09-20")
    choice = VaccineChoice(SECOND, "cell", "MDCK1", "the reference preparation")
    records = vaccine_rule_records([default, own], [choice], 1, {own, choice})
    assert [(r["rule"], r["scope"], r["used"]) for r in records] == [
        ("disable", "subtype default", False),
        ("disable", "map", True),
        ("choose", "map", True),
    ]
    assert records[0]["decided"] is None and records[1]["decided"] == "2026-09-20"
    assert records[2]["passage"] == "MDCK1" and records[2]["passage_class"] == "cell"
    assert "passage_class" not in records[0]
