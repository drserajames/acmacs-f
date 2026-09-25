"""Colour schemes: what is drawn, in what colour, and which entry wins."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades.colours import (
    ColourSchemeError,
    load_colour_scheme,
    load_colour_schemes,
    unused_entries,
)
from af.clades.groups import load_groups
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence

from .synthetic import build_clone, load_synthetic

HEADER = "order\tkey\tlegend\tcolour\n"
SUBTYPE = "A(H3N2)"


def write_scheme(directory: Path, name: str, *rows: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.tsv"
    path.write_text(HEADER + "".join(row + "\n" for row in rows))
    return path


def synthetic(tmp_path: Path) -> CladeSet:
    return load_synthetic(build_clone(tmp_path).parent)


def sequence(**states: str) -> AlignedSequence:
    residues = ["A"] * 400
    for position, state in states.items():
        residues[int(position[1:]) - 1] = state
    return AlignedSequence(amino_acids="".join(residues))


def test_reads_a_scheme(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_scheme(tmp_path / "s", "report", "1\tP\tP\t#112233", "2\tP.1\tP.1 (old)\t#445566")
    scheme = load_colour_scheme(path, SUBTYPE, clade_set)
    assert scheme.keys == ("P", "P.1")
    assert scheme.legend() == (("P", "#112233"), ("P.1 (old)", "#445566"))


def test_the_most_specific_clade_wins_whatever_the_row_order(tmp_path: Path) -> None:
    """Today the later row wins, so a scheme listing a child above its parent silently
    draws the child in the parent's colour (trap T9). Order must not decide this."""
    clade_set = synthetic(tmp_path)
    child_first = write_scheme(tmp_path / "a", "s", "1\tP.1\tP.1\t#445566", "2\tP\tP\t#112233")
    parent_first = write_scheme(tmp_path / "b", "s", "1\tP\tP\t#112233", "2\tP.1\tP.1\t#445566")
    for path in (child_first, parent_first):
        scheme = load_colour_scheme(path, SUBTYPE, clade_set)
        entry = scheme.entry_for("P.1.1", sequence(), clade_set)
        assert entry is not None and entry.colour == "#445566"


def test_a_group_beats_a_clade(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    groups_path = tmp_path / "groups.tsv"
    groups_path.write_text(
        f"subtype\tgroup\tanchor\tsubstitutions\tnote\n{SUBTYPE}\tP.1 20V\tP.1\t20V\t\n"
    )
    group_set = load_groups(groups_path, {SUBTYPE: clade_set})[SUBTYPE]
    path = write_scheme(
        tmp_path / "s", "report", "1\tP.1\tP.1\t#445566", "2\tP.1 20V\tP.1 20V\t#778899"
    )
    scheme = load_colour_scheme(path, SUBTYPE, clade_set, group_set)
    with_substitution = scheme.entry_for("P.1", sequence(p20="V"), clade_set, group_set)
    without = scheme.entry_for("P.1", sequence(), clade_set, group_set)
    assert with_substitution is not None and with_substitution.colour == "#778899"
    assert without is not None and without.colour == "#445566"


def test_a_virus_no_entry_covers_is_not_drawn(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_scheme(tmp_path / "s", "report", "1\tP.1\tP.1\t#445566")
    scheme = load_colour_scheme(path, SUBTYPE, clade_set)
    assert scheme.entry_for("P.2", sequence(), clade_set) is None
    assert scheme.entry_for(None, sequence(), clade_set) is None


def test_an_unknown_key_is_fatal(tmp_path: Path) -> None:
    """Four rows in today's tables name neither a clade nor a group; they can never
    colour anything and nobody is told."""
    clade_set = synthetic(tmp_path)
    path = write_scheme(tmp_path / "s", "report", "1\tP.9\tP.9\t#112233")
    with pytest.raises(ColourSchemeError, match="neither a clade"):
        load_colour_scheme(path, SUBTYPE, clade_set)


def test_a_bad_colour_is_fatal(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_scheme(tmp_path / "s", "report", "1\tP\tP\tred")
    with pytest.raises(ColourSchemeError, match="is not '#rrggbb'"):
        load_colour_scheme(path, SUBTYPE, clade_set)


def test_a_duplicate_key_is_fatal(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_scheme(tmp_path / "s", "report", "1\tP\tP\t#112233", "2\tP\tP again\t#445566")
    with pytest.raises(ColourSchemeError, match="duplicate key"):
        load_colour_scheme(path, SUBTYPE, clade_set)


def test_a_repeated_order_is_fatal(tmp_path: Path) -> None:
    """Two entries claiming the same legend position would order themselves by luck."""
    clade_set = synthetic(tmp_path)
    path = write_scheme(tmp_path / "s", "report", "1\tP\tP\t#112233", "1\tP.1\tP.1\t#445566")
    with pytest.raises(ColourSchemeError, match="order 1 is already used"):
        load_colour_scheme(path, SUBTYPE, clade_set)


def test_an_empty_scheme_is_fatal(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = tmp_path / "s" / "empty.tsv"
    path.parent.mkdir()
    path.write_text(HEADER)
    with pytest.raises(ColourSchemeError, match="no entries"):
        load_colour_scheme(path, SUBTYPE, clade_set)


def test_loads_every_scheme_in_a_directory(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    directory = tmp_path / "h3"
    write_scheme(directory, "report", "1\tP\tP\t#112233")
    write_scheme(directory, "slides", "1\tP.1\tP.1\t#445566")
    schemes = load_colour_schemes(directory, SUBTYPE, clade_set)
    assert sorted(schemes) == ["report", "slides"]


def test_a_directory_without_schemes_is_fatal(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ColourSchemeError, match="no colour schemes"):
        load_colour_schemes(tmp_path / "empty", SUBTYPE, clade_set)


def test_unused_entries_are_reported(tmp_path: Path) -> None:
    """Not fatal — a scheme kept across seasons will hold clades that died out — but it is
    also what a typo looks like, so it must be visible."""
    clade_set = synthetic(tmp_path)
    path = write_scheme(tmp_path / "s", "report", "1\tP\tP\t#112233", "2\tP.1\tP.1\t#445566")
    scheme = load_colour_scheme(path, SUBTYPE, clade_set)
    assert unused_entries(scheme, {"P": 17}) == ("P.1",)
