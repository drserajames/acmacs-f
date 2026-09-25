"""Local groups: "this clade or anything under it, plus these substitutions"."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades.groups import GroupError, count_matches, load_groups
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence

from .synthetic import build_clone, load_synthetic

HEADER = "subtype\tgroup\tanchor\tsubstitutions\tnote\n"
SUBTYPE = "A(H3N2)"


def write_groups(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "groups.tsv"
    path.write_text(HEADER + "".join(row + "\n" for row in rows))
    return path


def synthetic(tmp_path: Path) -> dict[str, CladeSet]:
    return {SUBTYPE: load_synthetic(build_clone(tmp_path).parent)}


def sequence(**states: str) -> AlignedSequence:
    residues = ["A"] * 400
    for position, state in states.items():
        residues[int(position[1:]) - 1] = state
    return AlignedSequence(amino_acids="".join(residues))


def test_reads_a_group(tmp_path: Path) -> None:
    sets = synthetic(tmp_path)
    groups = load_groups(write_groups(tmp_path, f"{SUBTYPE}\tP.1 20V\tP.1\t20V\t"), sets)
    assert groups[SUBTYPE].names == ("P.1 20V",)


def test_a_group_includes_the_anchors_descendants(tmp_path: Path) -> None:
    """The fix for a row having to name the parent to catch its children: anchoring on
    P.1 must match a virus in P.1.1 too."""
    sets = synthetic(tmp_path)
    groups = load_groups(write_groups(tmp_path, f"{SUBTYPE}\tP.1 20V\tP.1\t20V\t"), sets)
    group = groups[SUBTYPE].groups[0]
    assert group.matches("P.1.1", sequence(p20="V"), sets[SUBTYPE])
    assert group.matches("P.1", sequence(p20="V"), sets[SUBTYPE])


def test_a_group_does_not_match_outside_its_anchor(tmp_path: Path) -> None:
    sets = synthetic(tmp_path)
    groups = load_groups(write_groups(tmp_path, f"{SUBTYPE}\tP.1 20V\tP.1\t20V\t"), sets)
    group = groups[SUBTYPE].groups[0]
    assert not group.matches("P.2", sequence(p20="V"), sets[SUBTYPE])
    assert not group.matches(None, sequence(p20="V"), sets[SUBTYPE])


def test_a_group_without_an_anchor_matches_on_substitutions_alone(tmp_path: Path) -> None:
    sets = synthetic(tmp_path)
    groups = load_groups(write_groups(tmp_path, f"{SUBTYPE}\t20V\t\t20V\t"), sets)
    group = groups[SUBTYPE].groups[0]
    assert group.matches("P.2", sequence(p20="V"), sets[SUBTYPE])
    assert group.matches(None, sequence(p20="V"), sets[SUBTYPE])


def test_an_unobservable_position_does_not_match(tmp_path: Path) -> None:
    """A group is a positive claim. "We cannot see position 20" is not evidence of 20V."""
    sets = synthetic(tmp_path)
    groups = load_groups(write_groups(tmp_path, f"{SUBTYPE}\tP.1 20V\tP.1\t20V\t"), sets)
    assert not groups[SUBTYPE].groups[0].matches("P.1", sequence(p20="X"), sets[SUBTYPE])


def test_an_unknown_anchor_is_fatal(tmp_path: Path) -> None:
    """The 24 inert rows in today's tables are all of this shape: anchored on a clade name
    that no longer exists, labelling nothing, silently."""
    sets = synthetic(tmp_path)
    with pytest.raises(GroupError, match="which A\\(H3N2\\) does not define"):
        load_groups(write_groups(tmp_path, f"{SUBTYPE}\tX 20V\tP.9\t20V\t"), sets)


def test_a_group_with_no_substitutions_is_fatal(tmp_path: Path) -> None:
    sets = synthetic(tmp_path)
    with pytest.raises(GroupError, match="identical to its anchor clade"):
        load_groups(write_groups(tmp_path, f"{SUBTYPE}\tP.1 again\tP.1\t\t"), sets)


def test_an_unreadable_substitution_is_fatal(tmp_path: Path) -> None:
    sets = synthetic(tmp_path)
    with pytest.raises(GroupError, match="is not a substitution"):
        load_groups(write_groups(tmp_path, f"{SUBTYPE}\tbad\tP.1\tV20\t"), sets)


def test_a_duplicate_group_name_is_fatal(tmp_path: Path) -> None:
    sets = synthetic(tmp_path)
    with pytest.raises(GroupError, match="duplicate group"):
        load_groups(
            write_groups(tmp_path, f"{SUBTYPE}\tsame\tP.1\t20V\t", f"{SUBTYPE}\tsame\tP.1\t21V\t"),
            sets,
        )


def test_an_unknown_subtype_is_fatal(tmp_path: Path) -> None:
    sets = synthetic(tmp_path)
    with pytest.raises(GroupError, match="unknown subtype"):
        load_groups(write_groups(tmp_path, "B/Yam\tg\t\t20V\t"), sets)


def test_every_problem_is_reported_together(tmp_path: Path) -> None:
    """One run shows everything wrong with the file, not one error per run."""
    sets = synthetic(tmp_path)
    with pytest.raises(GroupError) as raised:
        load_groups(
            write_groups(
                tmp_path,
                f"{SUBTYPE}\tone\tP.9\t20V\t",
                f"{SUBTYPE}\ttwo\tP.1\tV20\t",
                "B/Yam\tthree\t\t20V\t",
            ),
            sets,
        )
    assert len(raised.value.problems) == 3


def test_matching_returns_the_most_specific_group_first(tmp_path: Path) -> None:
    """So a caller wanting one label takes the first, with no tie-break of its own."""
    sets = synthetic(tmp_path)
    groups = load_groups(
        write_groups(
            tmp_path,
            f"{SUBTYPE}\tP.1 20V\tP.1\t20V\t",
            f"{SUBTYPE}\tP.1 20V 21W\tP.1\t20V 21W\t",
        ),
        sets,
    )
    matched = groups[SUBTYPE].matching("P.1", sequence(p20="V", p21="W"), sets[SUBTYPE])
    assert matched == ("P.1 20V 21W", "P.1 20V")


def test_counts_show_a_group_that_matches_nothing(tmp_path: Path) -> None:
    """Design rule 1: a rule matching nothing must be visible. It is what a mistyped
    substitution looks like."""
    sets = synthetic(tmp_path)
    groups = load_groups(
        write_groups(tmp_path, f"{SUBTYPE}\thit\tP.1\t20V\t", f"{SUBTYPE}\tmiss\tP.1\t21W\t"),
        sets,
    )
    counts = count_matches(
        groups[SUBTYPE], [("P.1", sequence(p20="V")), ("P.1.1", sequence(p20="V"))], sets[SUBTYPE]
    )
    assert counts == {"hit": 2, "miss": 0}


def test_missing_column_is_fatal(tmp_path: Path) -> None:
    sets = synthetic(tmp_path)
    path = tmp_path / "groups.tsv"
    path.write_text("subtype\tgroup\tanchor\n")
    with pytest.raises(GroupError, match="missing column"):
        load_groups(path, sets)
