"""Colour schemes: what is drawn, in what colour, and which entry wins."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades.colours import (
    ColourScheme,
    ColourSchemeError,
    load_colour_scheme,
    load_colour_schemes,
    shadowed_entries,
    unused_entries,
)
from af.clades.groups import GroupSet, load_groups
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


def test_the_last_matching_row_wins(tmp_path: Path) -> None:
    """Q80 (Sarah, 29 Sep 2026): the round's rule, so maps and geo draw what it drew. A
    child listed above its parent is drawn in the parent's colour (trap T9, reported by
    shadowed_entries rather than refused)."""
    clade_set = synthetic(tmp_path)
    child_first = write_scheme(tmp_path / "a", "s", "1\tP.1\tP.1\t#445566", "2\tP\tP\t#112233")
    parent_first = write_scheme(tmp_path / "b", "s", "1\tP\tP\t#112233", "2\tP.1\tP.1\t#445566")
    for path, colour in ((child_first, "#112233"), (parent_first, "#445566")):
        scheme = load_colour_scheme(path, SUBTYPE, clade_set)
        entry = scheme.entry_for("P.1.1", sequence(), clade_set)
        assert entry is not None and entry.colour == colour


def test_order_is_the_order_column_not_the_file_line(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_scheme(tmp_path / "s", "s", "2\tP\tP\t#112233", "1\tP.1\tP.1\t#445566")
    scheme = load_colour_scheme(path, SUBTYPE, clade_set)
    entry = scheme.entry_for("P.1", sequence(), clade_set)
    assert entry is not None and entry.key == "P"
    assert [label for label, _ in scheme.legend()] == ["P.1", "P"]


def group_set_p1_20v(tmp_path: Path, clade_set: CladeSet) -> GroupSet:
    groups_path = tmp_path / "groups.tsv"
    groups_path.write_text(
        f"subtype\tgroup\tanchor\tsubstitutions\tnote\n{SUBTYPE}\tP.1 20V\tP.1\t20V\t\n"
    )
    return load_groups(groups_path, {SUBTYPE: clade_set})[SUBTYPE]


def colour_of(
    scheme: ColourScheme,
    clade: str,
    virus: AlignedSequence,
    clade_set: CladeSet,
    group_set: GroupSet | None = None,
) -> str | None:
    entry = scheme.entry_for(clade, virus, clade_set, group_set)
    return entry.colour if entry is not None else None


def test_a_group_or_a_clade_wins_by_row_order(tmp_path: Path) -> None:
    """A group is not privileged: listed above its clade, the clade row wins, as in the
    round's tables (where this kept a current clade from taking a legacy group's colour)."""
    clade_set = synthetic(tmp_path)
    group_set = group_set_p1_20v(tmp_path, clade_set)
    group_last = write_scheme(
        tmp_path / "a", "s", "1\tP.1\tP.1\t#445566", "2\tP.1 20V\tP.1 20V\t#778899"
    )
    group_first = write_scheme(
        tmp_path / "b", "s", "1\tP.1 20V\tP.1 20V\t#778899", "2\tP.1\tP.1\t#445566"
    )
    last = load_colour_scheme(group_last, SUBTYPE, clade_set, group_set)
    first = load_colour_scheme(group_first, SUBTYPE, clade_set, group_set)
    with_substitution = sequence(p20="V")
    assert colour_of(last, "P.1", with_substitution, clade_set, group_set) == "#778899"
    assert colour_of(last, "P.1", sequence(), clade_set, group_set) == "#445566"
    assert colour_of(first, "P.1", with_substitution, clade_set, group_set) == "#445566"


def test_a_scheme_with_groups_needs_its_group_set(tmp_path: Path) -> None:
    """Without the groups a group row would silently never match."""
    clade_set = synthetic(tmp_path)
    group_set = group_set_p1_20v(tmp_path, clade_set)
    path = write_scheme(tmp_path / "s", "s", "1\tP.1 20V\tP.1 20V\t#778899")
    scheme = load_colour_scheme(path, SUBTYPE, clade_set, group_set)
    with pytest.raises(ValueError, match="needs the group set"):
        scheme.entry_for("P.1", sequence(p20="V"), clade_set)


def test_shadowed_rows_are_reported(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    group_set = group_set_p1_20v(tmp_path, clade_set)
    path = write_scheme(
        tmp_path / "s",
        "s",
        "1\tP.1 20V\tP.1 20V\t#778899",  # shadowed: every member is in P.1, listed later
        "2\tP.1\tP.1\t#445566",
        "3\tP.2\tP.2\t#aabbcc",  # not shadowed: P.1 does not contain P.2
        "4\tP.1.1\tP.1.1\t#ddeeff",  # not shadowed: nothing later
    )
    scheme = load_colour_scheme(path, SUBTYPE, clade_set, group_set)
    [shadowed] = shadowed_entries(scheme, clade_set, group_set)
    assert (shadowed.entry.key, shadowed.by.key) == ("P.1 20V", "P.1")
    assert "row 1 'P.1 20V' is always overridden by row 2 'P.1'" in str(shadowed)


def test_a_later_group_shadows_only_what_it_always_matches(tmp_path: Path) -> None:
    """A group with a substitution does not match every virus of its clade, so it cannot
    shadow the clade row above it; the clade row above a parent-clade row can."""
    clade_set = synthetic(tmp_path)
    group_set = group_set_p1_20v(tmp_path, clade_set)
    path = write_scheme(tmp_path / "s", "s", "1\tP.1\tP.1\t#445566", "2\tP.1 20V\tP.1 20V\t#778899")
    scheme = load_colour_scheme(path, SUBTYPE, clade_set, group_set)
    assert shadowed_entries(scheme, clade_set, group_set) == ()


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
