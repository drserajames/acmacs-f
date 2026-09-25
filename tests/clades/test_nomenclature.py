"""Loading the upstream nomenclature: the pin, the hierarchy, and what it refuses."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from af.clades.nomenclature import (
    CladeSet,
    NomenclatureError,
    Pin,
    load_clade_set,
    load_clade_sets,
)

from .synthetic import build_clone, clone_commit, commit_command

SUBTYPE = "A(H3N2)"


def synthetic(tmp_path: Path) -> tuple[Path, Path, str]:
    """A fresh synthetic clone: where to load from, the clone itself, and its commit."""
    clone = build_clone(tmp_path)
    return clone.parent, clone, clone_commit(clone)


def load(clones: Path, **kwargs: object) -> CladeSet:
    return load_clade_set(SUBTYPE, clones, repository="synthetic_HA", **kwargs)  # type: ignore[arg-type]


def test_reads_the_hierarchy(tmp_path: Path) -> None:
    clade_set = load(synthetic(tmp_path)[0])
    assert set(clade_set.names) == {"P", "P.1", "P.1.1", "P.2", "P.3"}
    assert clade_set.roots == ("P",)
    assert clade_set.ancestors("P.1.1") == ("P.1", "P")
    assert clade_set.children("P") == ("P.1", "P.2", "P.3")


def test_clade_files_are_display_names_not_assignable_clades(tmp_path: Path) -> None:
    """Upstream's two hierarchies must not be merged: doing so labelled every virus with
    a legacy name instead of its clade."""
    clade_set = load(synthetic(tmp_path)[0])
    assert "Q" not in clade_set
    assert clade_set.legacy_name("P.1") == "legacy-1"
    assert clade_set.legacy_name("P.1.1") is None


def test_is_within_covers_self_and_descendants(tmp_path: Path) -> None:
    clade_set = load(synthetic(tmp_path)[0])
    assert clade_set.is_within("P.1.1", "P")
    assert clade_set.is_within("P.1", "P.1")
    assert not clade_set.is_within("P", "P.1")
    assert not clade_set.is_within("P.2", "P.1")


def test_descendants_include_the_clade_itself(tmp_path: Path) -> None:
    assert set(load(synthetic(tmp_path)[0]).descendants("P.1")) == {"P.1", "P.1.1"}


def test_unknown_clade_name_is_fatal(tmp_path: Path) -> None:
    clade_set = load(synthetic(tmp_path)[0])
    with pytest.raises(KeyError, match="no clade named"):
        clade_set.is_within("P.1", "P.9")


def test_cumulative_signature_includes_ancestors(tmp_path: Path) -> None:
    signature = load(synthetic(tmp_path)[0]).cumulative("P.1.1")
    assert signature == {
        ("aa", 5): "K",  # from P
        ("aa", 9): "T",  # from P.1
        ("aa", 331): "W",  # P.1's HA2 2W, converted once
        ("aa", 12): "N",  # P.1.1's own
        ("nuc", 6): "A",
    }


def test_descendant_state_replaces_an_ancestors(tmp_path: Path) -> None:
    """P.3 redefines position 5, which P already defines; the descendant wins."""
    assert load(synthetic(tmp_path)[0]).cumulative("P.3")[("aa", 5)] == "K"


def test_unexpressible_loci_are_reported_not_dropped(tmp_path: Path) -> None:
    clade_set = load(synthetic(tmp_path)[0])
    unexpressible = clade_set.unexpressible()
    assert list(unexpressible) == ["P.2"]
    assert "signal peptide" in str(unexpressible["P.2"][0])
    # the deletion itself is expressible, and is kept
    assert clade_set.cumulative("P.2")[("aa", 7)] == "-"


def test_revoked_clades_are_marked(tmp_path: Path) -> None:
    clade_set = load(synthetic(tmp_path)[0])
    assert clade_set["P.3"].revoked
    assert "P.3" not in {clade.name for clade in clade_set.live}


def test_version_names_the_repository_and_commit(tmp_path: Path) -> None:
    clones, _, commit = synthetic(tmp_path)
    assert load(clones).version == f"synthetic_HA@{commit}"


def test_pin_mismatch_is_fatal(tmp_path: Path) -> None:
    clones = synthetic(tmp_path)[0]
    pin = Pin(subtype=SUBTYPE, repository="synthetic_HA", commit="0" * 40)
    with pytest.raises(NomenclatureError, match="not the pinned"):
        load_clade_set(SUBTYPE, clones, pin)


def test_pin_accepts_an_abbreviated_commit(tmp_path: Path) -> None:
    clones, _, commit = synthetic(tmp_path)
    pin = Pin(subtype=SUBTYPE, repository="synthetic_HA", commit=commit[:8])
    assert load_clade_set(SUBTYPE, clones, pin).pin.commit == commit[:8]


def test_missing_clone_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(NomenclatureError, match="clone not found"):
        load_clade_set(SUBTYPE, tmp_path, repository="absent_HA")


def test_unknown_key_in_a_yaml_file_is_fatal(tmp_path: Path) -> None:
    """A silently skipped line could drop a defining mutation and change what a clade means."""
    clones, clone, _ = synthetic(tmp_path)
    (clone / "subclades" / "P.yml").write_text("name: P\nparent: none\nsurprise: 1\n")
    with pytest.raises(NomenclatureError, match="unknown key 'surprise'"):
        load(clones)


def test_malformed_line_names_the_file_and_line(tmp_path: Path) -> None:
    clones, clone, _ = synthetic(tmp_path)
    (clone / "subclades" / "P.yml").write_text("name: P\nparent: none\nrubbish\n")
    with pytest.raises(NomenclatureError, match=r"P\.yml:3"):
        load(clones)


def test_defining_mutation_missing_a_field_is_fatal(tmp_path: Path) -> None:
    clones, clone, _ = synthetic(tmp_path)
    (clone / "subclades" / "P.yml").write_text(
        "name: P\nparent: none\ndefining_mutations:\n- locus: HA1\n  position: 5\n"
    )
    with pytest.raises(NomenclatureError, match=r"missing \['state'\]"):
        load(clones)


def test_parent_cycle_is_fatal(tmp_path: Path) -> None:
    clones, clone, _ = synthetic(tmp_path)
    (clone / "subclades" / "P.yml").write_text("name: P\nparent: P.1\n")
    with pytest.raises(NomenclatureError, match="cycle"):
        load(clones)


def test_two_pins_for_one_subtype_is_fatal(tmp_path: Path) -> None:
    clones, _, commit = synthetic(tmp_path)
    pins = [Pin(SUBTYPE, "synthetic_HA", commit), Pin(SUBTYPE, "synthetic_HA", commit)]
    with pytest.raises(NomenclatureError, match="two pins"):
        load_clade_sets(clones, pins)


def test_not_a_git_clone_is_fatal(tmp_path: Path) -> None:
    root = tmp_path / "loose_HA" / "subclades"
    root.mkdir(parents=True)
    (root / "P.yml").write_text("name: P\nparent: none\n")
    with pytest.raises(NomenclatureError, match="cannot read the commit"):
        load_clade_set(SUBTYPE, tmp_path, repository="loose_HA")


def test_moving_the_clone_changes_the_version(tmp_path: Path) -> None:
    """The clones do move; the version must move with them, or a stale clade set looks
    current."""
    clones, clone, commit = synthetic(tmp_path)
    (clone / "subclades" / "P.4.yml").write_text("name: P.4\nparent: P\n")
    subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
    subprocess.run(commit_command(clone, "add"), check=True)
    assert load(clones).version != f"synthetic_HA@{commit}"
