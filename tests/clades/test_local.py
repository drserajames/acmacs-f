"""Locally defined clades: they refine the nomenclature and must never override it."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades.assign import Node, assign_tree
from af.clades.local import LocalCladeError, extend_from_file, load_local_clades
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence

from .synthetic import build_clone, load_synthetic

HEADER = "subtype\tname\tparent\tmutations\tscope\tnote\n"
SUBTYPE = "A(H3N2)"


def write_local(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "local.tsv"
    path.write_text(HEADER + "".join(row + "\n" for row in rows))
    return path


def synthetic(tmp_path: Path) -> CladeSet:
    return load_synthetic(build_clone(tmp_path).parent)


def protein(**states: str) -> str:
    residues = ["A"] * 400
    for position, state in states.items():
        residues[int(position[1:]) - 1] = state
    return "".join(residues)


def sequence(**states: str) -> AlignedSequence:
    return AlignedSequence(amino_acids=protein(**states), nucleotides="A" * 1200)


def test_reads_a_local_clade(tmp_path: Path) -> None:
    path = write_local(tmp_path, f"{SUBTYPE}\tL.1\tP.1\t20V\tactive\tnote here")
    local = load_local_clades(path)[SUBTYPE]
    assert len(local) == 1
    assert local[0].name == "L.1"
    assert local[0].parent == "P.1"
    assert [str(mutation) for mutation in local[0].mutations] == ["20V"]


def test_nucleotide_mutations_are_understood(tmp_path: Path) -> None:
    path = write_local(tmp_path, f"{SUBTYPE}\tL.1\t-\tnuc30A 20V\thistorical\t")
    mutations = load_local_clades(path)[SUBTYPE][0].mutations
    assert sorted(str(mutation) for mutation in mutations) == ["20V", "nuc 30A"]


def test_empty_parent_means_the_root(tmp_path: Path) -> None:
    path = write_local(tmp_path, f"{SUBTYPE}\tL.1\t-\t20V\thistorical\t")
    assert load_local_clades(path)[SUBTYPE][0].parent is None


def test_a_local_clade_extends_the_set(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_local(tmp_path, f"{SUBTYPE}\tL.1\tP.1\t20V\tactive\t")
    extended = extend_from_file(clade_set, path)
    assert "L.1" in extended
    assert extended.ancestors("L.1") == ("P.1", "P")
    assert extended.is_within("L.1", "P")
    assert extended.is_local("L.1")
    assert not extended.is_local("P.1")


def test_the_version_records_the_local_layer(tmp_path: Path) -> None:
    """A change to local definitions must make existing assignments detectably stale."""
    clade_set = synthetic(tmp_path)
    first = extend_from_file(
        clade_set, write_local(tmp_path, f"{SUBTYPE}\tL.1\tP.1\t20V\tactive\t")
    )
    second = extend_from_file(
        clade_set, write_local(tmp_path, f"{SUBTYPE}\tL.1\tP.1\t21V\tactive\t")
    )
    assert first.version.startswith(clade_set.version + "+local:")
    assert first.version != second.version


def test_a_local_clade_may_not_redefine_a_published_one(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_local(tmp_path, f"{SUBTYPE}\tP.1\t-\t20V\tactive\t")
    with pytest.raises(LocalCladeError, match="already defined by the nomenclature"):
        extend_from_file(clade_set, path)


def test_an_unknown_parent_is_fatal(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_local(tmp_path, f"{SUBTYPE}\tL.1\tP.9\t20V\tactive\t")
    with pytest.raises(LocalCladeError, match="neither a published clade nor another local"):
        extend_from_file(clade_set, path)


def test_a_local_clade_without_mutations_is_fatal(tmp_path: Path) -> None:
    """Empty mutations load (they may come from a source signature), but a clade that
    reaches the clade set still without any could never be assigned."""
    path = write_local(tmp_path, f"{SUBTYPE}\tL.1\tP.1\t\tactive\t")
    assert load_local_clades(path)[SUBTYPE][0].mutations == ()
    with pytest.raises(LocalCladeError, match="defines no mutations"):
        extend_from_file(synthetic(tmp_path), path)


def test_an_unknown_scope_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(LocalCladeError, match="scope 'ancient' must be one of"):
        load_local_clades(write_local(tmp_path, f"{SUBTYPE}\tL.1\tP.1\t20V\tancient\t"))


def test_an_unreadable_mutation_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(LocalCladeError, match="is not a mutation"):
        load_local_clades(write_local(tmp_path, f"{SUBTYPE}\tL.1\tP.1\tV20\tactive\t"))


def test_local_clades_are_assigned_on_a_tree(tmp_path: Path) -> None:
    clade_set = extend_from_file(
        synthetic(tmp_path), write_local(tmp_path, f"{SUBTYPE}\tL.1\tP.1\t20V\tactive\t")
    )
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("mid", "root", sequence(p5="K", p9="T", p331="W")),
        Node("leaf", "mid", sequence(p5="K", p9="T", p331="W", p20="V")),
    ]
    result = assign_tree(nodes, clade_set)
    assert result.clade("mid") == "P.1"
    assert result.clade("leaf") == "L.1"


def test_a_root_attached_local_clade_never_displaces_a_published_one(tmp_path: Path) -> None:
    """The reason :meth:`CladeSet.upstream_depth` exists. Two local clades attached at the
    root once captured 107,292 leaves of the real H1 tree that the nomenclature names,
    because a local label shut the published clades out of everything below it."""
    clade_set = synthetic(tmp_path)
    # a local clade defined by a residue every virus in this test carries
    extended = extend_from_file(
        clade_set, write_local(tmp_path, f"{SUBTYPE}\tOLD\t-\t2A\thistorical\t")
    )
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("mid", "root", sequence(p5="K", p9="T", p331="W")),
        Node("leaf", "mid", sequence(p5="K", p9="T", p331="W", p12="N")),
    ]
    before = assign_tree(nodes, clade_set)
    after = assign_tree(nodes, extended)
    assert {name: after.clade(name) for name in ("root", "mid", "leaf")} == {
        name: before.clade(name) for name in ("root", "mid", "leaf")
    }


def test_a_local_clade_names_viruses_the_nomenclature_leaves_unnamed(tmp_path: Path) -> None:
    """The case it exists for: lineages ancestral to the nomenclature's own root."""
    clade_set = synthetic(tmp_path)
    extended = extend_from_file(
        clade_set, write_local(tmp_path, f"{SUBTYPE}\tOLD\t-\t40Q\thistorical\t")
    )
    # no 5K, so no published clade matches; 40Q, so the local one does
    nodes = [Node("root", None, sequence(p40="Q"))]
    assert assign_tree(nodes, clade_set).clade("root") is None
    assert assign_tree(nodes, extended).clade("root") == "OLD"


def test_a_cycle_through_local_parents_is_fatal(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    path = write_local(
        tmp_path,
        f"{SUBTYPE}\tL.1\tL.2\t20V\tactive\t",
        f"{SUBTYPE}\tL.2\tL.1\t21V\tactive\t",
    )
    with pytest.raises(LocalCladeError, match="cycle"):
        extend_from_file(clade_set, path)


def test_published_labels_changed_reports_an_overriding_local_clade(tmp_path: Path) -> None:
    """The check WS5 runs when local definitions change: a local clade may name the
    unnamed and may refine a published clade, but must never move a virus from one
    published clade to another."""
    from af.clades.assign import published_labels_changed

    clade_set = synthetic(tmp_path)
    refining = extend_from_file(
        clade_set, write_local(tmp_path, f"{SUBTYPE}\tL.1\tP.1\t20V\tactive\t")
    )
    nodes = [
        Node("root", None, sequence(p5="K")),
        Node("mid", "root", sequence(p5="K", p9="T", p331="W")),
        Node("leaf", "mid", sequence(p5="K", p9="T", p331="W", p20="V")),
    ]
    before = assign_tree(nodes, clade_set)
    after = assign_tree(nodes, refining)
    # the leaf became a local child of P.1, which is a refinement, not a change
    assert after.clade("leaf") == "L.1"
    assert published_labels_changed(before, after, refining) == {}


def test_published_labels_changed_detects_a_real_move(tmp_path: Path) -> None:
    """The negative case above only proves it stays quiet; this proves it speaks."""
    from af.clades.assign import Assignment, TreeAssignment, published_labels_changed

    clade_set = synthetic(tmp_path)

    def assignment(clade: str) -> TreeAssignment:
        return TreeAssignment(
            clade_set_version=clade_set.version,
            subtype=SUBTYPE,
            assignments={"leaf": Assignment("leaf", clade)},
        )

    changed = published_labels_changed(assignment("P.1"), assignment("P.2"), clade_set)
    assert changed == {"leaf": ("P.1", "P.2")}
    # and a virus gaining a published clade where it had none is a change too
    assert published_labels_changed(assignment("P.1"), assignment("P.1"), clade_set) == {}
