"""The retrospective legacy label (4.10): older clades, labelled from tree nodes."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades.assign import Node, assign_tree
from af.clades.colours import ColourEntry, ColourSchemeError, scheme_from_rows, shadowed_entries
from af.clades.nomenclature import CladeSet, NomenclatureError
from af.clades.sequence import AlignedSequence

from .synthetic import add_legacy_clades, build_clone, load_synthetic

SUBTYPE = "A(H3N2)"
CODON = {"A": "GCT", "V": "GTT", "W": "TGG", "K": "AAA", "R": "CGT", "S": "TCT"}


def legacy_set(tmp_path: Path) -> CladeSet:
    add_legacy_clades(build_clone(tmp_path))
    return load_synthetic(tmp_path)


def nucleotides(**residues: str) -> str:
    """A mature HA of alanines, with ``p<position>=<residue>`` changed."""
    aa = ["A"] * 566
    for key, residue in residues.items():
        aa[int(key[1:]) - 1] = residue
    return "".join(CODON[a] for a in aa)


def virus(**residues: str) -> AlignedSequence:
    return AlignedSequence.from_nucleotides(nucleotides(**residues))


# ------------------------------------------------------------------ the legacy hierarchy


def test_only_self_defining_legacy_clades_can_label(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    assert clade_set.legacy_defining() == ("L", "L.1", "L.1.1", "L.2", "L.3")  # not the pointer Q
    assert clade_set.legacy_ancestors("L.1.1") == ("L.1", "L")
    assert clade_set.legacy_parent("L") is None
    assert clade_set.legacy_is_within("L.1.1", "L") and not clade_set.legacy_is_within("L.2", "L.1")
    assert clade_set.legacy_common_ancestor(["L.1.1", "L.2"]) == "L"


# ------------------------------------------------------------------ labelling from tree nodes


def test_the_legacy_set_is_a_nomenclature_of_the_self_defining_legacy_clades(
    tmp_path: Path,
) -> None:
    legacy = legacy_set(tmp_path).legacy_set()
    assert set(legacy.names) == {"L", "L.1", "L.1.1", "L.2", "L.3"}
    assert legacy.ancestors("L.1.1") == ("L.1", "L")
    assert legacy.version.endswith("+legacy")


def test_the_tree_engine_labels_by_placement_so_a_reversion_keeps_its_lineage(
    tmp_path: Path,
) -> None:
    """Sarah (30 Sep): clades are defined from the tree node. A leaf whose root marker has
    reverted still belongs to the lineage it sits in; no per-sequence tolerance is needed."""
    legacy = legacy_set(tmp_path).legacy_set()
    nodes = [
        Node("root", None, virus(p20="V")),
        Node("n1", "root", virus(p20="V", p21="W")),
        Node("n11", "n1", virus(p20="V", p21="W", p22="K")),
        Node("reverted", "n11", virus(p20="A", p21="W", p22="K")),
        Node("sister", "root", virus(p20="V", p23="R")),
    ]
    result = assign_tree(nodes, legacy, require_gap_support=False)
    assert result.clade("n11") == "L.1.1"
    assert result.clade("reverted") == "L.1.1"
    assert result.clade("sister") == "L.2"


def test_a_subtype_without_self_defining_legacy_clades_has_no_legacy_set(tmp_path: Path) -> None:
    clade_set = load_synthetic(build_clone(tmp_path).parent)  # only the pointer Q
    with pytest.raises(NomenclatureError, match="no legacy clades with their own definitions"):
        clade_set.legacy_set()


# ------------------------------------------------------------------ colour schemes


def rows(*keys: str) -> list[tuple[str, dict[str, str]]]:
    return [
        (f"row {i}", {"order": str(i), "key": key, "legend": key, "colour": f"#0000{i:02d}"})
        for i, key in enumerate(keys, start=1)
    ]


def key_of(entry: ColourEntry | None) -> str | None:
    return entry.key if entry is not None else None


def test_a_scheme_row_may_name_a_legacy_clade(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    scheme = scheme_from_rows(rows("P", "L", "L.1"), SUBTYPE, "s", clade_set)
    assert [(e.key, e.is_legacy) for e in scheme] == [("P", False), ("L", True), ("L.1", True)]
    old = virus(p20="V", p21="W", p22="K")
    assert key_of(scheme.entry_for(None, old, clade_set, legacy_clade="L.1.1")) == "L.1"
    assert key_of(scheme.entry_for(None, old, clade_set, legacy_clade="L.2")) == "L"
    # a virus the subclades name is never coloured by a legacy row
    assert key_of(scheme.entry_for("P.1", old, clade_set)) == "P"
    with pytest.raises(ValueError, match="never both"):
        scheme.entry_for("P", old, clade_set, legacy_clade="L")


def test_a_short_or_pointer_legacy_name_is_not_a_legacy_key(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    with pytest.raises(ColourSchemeError, match="'l1' is neither"):
        scheme_from_rows(rows("l1"), SUBTYPE, "s", clade_set)


def test_a_legacy_row_is_shadowed_only_by_a_later_legacy_ancestor(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    scheme = scheme_from_rows(rows("L.1", "P", "L"), SUBTYPE, "s", clade_set)
    [shadowed] = shadowed_entries(scheme, clade_set)
    assert (shadowed.entry.key, shadowed.by.key) == ("L.1", "L")


def test_the_importer_resolves_a_full_legacy_name_and_refuses_a_short_one(tmp_path: Path) -> None:
    from af.clades.importer import import_semantic_clades

    clade_set = legacy_set(tmp_path)
    module = tmp_path / "semantic_clades.py"
    module.write_text(
        "from ae.utils.org import org_table_to_dict\n"
        'sData = {"A(H3N2)": {"clades-v1": org_table_to_dict("""\n'
        "| name  | legend | color   |\n|-------+--------+---------|\n"
        "| L.1.1 | old    | #112233 |\n| l1    | short  | #445566 |\n"
        '""")}}\n'
    )
    report = import_semantic_clades(module, {SUBTYPE: clade_set})
    assert [row["key"] for row in report.colour_rows[(SUBTYPE, "clades-v1")]] == ["L.1.1"]
    assert report.legacy == [(SUBTYPE, "clades-v1", 1, "L.1.1")]
    [dead] = report.dead
    assert "by its short name 'l1'" in dead.reason
