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


# ---- partial proteins (Sarah, Q95: keep HA1 when HA2 fails; the failed span is "X") ----


def _padded(*, ha1_failed: bool, **states: str) -> AlignedSequence:
    """A mature protein of this synthetic subtype (HA1 1-329, HA2 after), one CDS padded."""
    residues = ["A"] * 400
    span = range(0, 329) if ha1_failed else range(329, 400)
    for index in span:
        residues[index] = "X"
    for position, state in states.items():
        residues[int(position[1:]) - 1] = state
    return AlignedSequence(amino_acids="".join(residues))


def test_a_padded_cds_is_unobservable_never_a_wrong_residue(tmp_path: Path) -> None:
    """A group on a translated locus can match a partial protein; one on the padded CDS
    cannot (unobservable is not evidence), and nor is it contradicted."""
    from af.clades.groups import Group, Substitution
    from af.clades.sequence import Evidence

    clade_set = synthetic(tmp_path)[SUBTYPE]
    on_ha1 = Group(SUBTYPE, "ha1 group", None, (Substitution(20, "V"),))
    on_ha2 = Group(SUBTYPE, "ha2 group", None, (Substitution(331, "W"),))

    ha2_failed = _padded(ha1_failed=False, p20="V")
    assert on_ha1.matches(None, ha2_failed, clade_set)
    assert not on_ha2.matches(None, ha2_failed, clade_set)
    assert ha2_failed.evidence("aa", 331, "W") is Evidence.UNOBSERVABLE

    ha1_failed = _padded(ha1_failed=True, p331="W")
    assert on_ha2.matches(None, ha1_failed, clade_set)
    assert not on_ha1.matches(None, ha1_failed, clade_set)
    assert ha1_failed.evidence("aa", 20, "V") is Evidence.UNOBSERVABLE
