"""Subtype-wide vaccine defaults: shared by every round, optional unless the file says otherwise."""

from pathlib import Path

from af.map.build import load_vaccine_defaults

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


def test_no_file_means_no_defaults() -> None:
    assert load_vaccine_defaults(None) == {}
