"""Resolving other systems' clade labels onto the upstream scheme. Synthetic data only.

The synthetic nomenclature (``synthetic.py``): ``P`` → ``P.1`` → ``P.1.1`` and ``P.2``;
``P.1`` carries the older name ``legacy-1``, and the legacy file ``Q`` aliases ``P.1``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades.coordinates import Position
from af.clades.labels import LabelError, canonical_labels, older_names
from af.clades.local import LocalClade, extend
from af.clades.nomenclature import CladeSet

from .synthetic import build_clone, load_synthetic, write_clade


def clade_set(tmp_path: Path, *, extra_legacy: dict[str, str] | None = None) -> CladeSet:
    clone = build_clone(tmp_path / "clone")
    for name, body in (extra_legacy or {}).items():
        write_clade(clone / "clades", name, body)
    return load_synthetic(clone.parent)


def with_local(clades: CladeSet) -> CladeSet:
    def local(name: str, parent: str | None) -> LocalClade:
        return LocalClade(clades.subtype, name, parent, (Position("aa", 20, "R"),))

    return extend(
        clades,
        [local("P.1.x", "P.1"), local("P.1.x.y", "P.1.x"), local("L0", None)],
        version_suffix="test",
    )


def test_each_route(tmp_path: Path) -> None:
    clades = with_local(
        clade_set(tmp_path, extra_legacy={"R": "name: R\nparent: none\nrepresentatives: []"})
    )
    labels = ["P.1.1", "P.1.x", "P.1.x.y", "L0", "P.1 (legacy-1)", "legacy-1", "Q", "R", None]
    result = canonical_labels(labels, clades)
    assert {label: (r.clade, r.route) for label, r in result.resolved.items()} == {
        "P.1.1": ("P.1.1", "subclade"),
        "P.1.x": ("P.1", "local"),
        "P.1.x.y": ("P.1", "local"),
        "L0": (None, "local"),
        "P.1 (legacy-1)": ("P.1", "display"),
        "legacy-1": ("P.1", "legacy"),
        "Q": ("P.1", "legacy"),
        "R": (None, "outside"),
        None: (None, "unnamed"),
    }
    assert result.clade_set_version == clades.version
    assert result.counts()["legacy"] == 2 and result.counts()["unmapped"] == 0


def test_a_bracket_that_is_not_the_older_name_is_not_guessed_away(tmp_path: Path) -> None:
    """Legend text such as a substitution in brackets must not pass as the clade."""
    result = canonical_labels(
        ["P.1 (145K)", "P.2 (legacy-1)"], clade_set(tmp_path), allow_unmapped=True
    )
    assert set(result.unmapped) == {"P.1 (145K)", "P.2 (legacy-1)"}
    assert "is not 'P.1'" in result.unmapped["P.1 (145K)"]


def test_unmapped_labels_are_fatal_and_all_listed(tmp_path: Path) -> None:
    with pytest.raises(LabelError, match="2 label") as caught:
        canonical_labels(["P", "P.9", "P.1 extra"], clade_set(tmp_path))
    assert "'P.9'" in str(caught.value) and "'P.1 extra'" in str(caught.value)


def test_unmapped_labels_can_be_reported_instead(tmp_path: Path) -> None:
    result = canonical_labels(["P", "P.9", "P"], clade_set(tmp_path), allow_unmapped=True)
    assert list(result.resolved) == ["P"]
    assert list(result.unmapped) == ["P.9"]
    with pytest.raises(KeyError, match="not resolved"):
        result.clade("P.9")


def test_names_are_never_matched_by_prefix(tmp_path: Path) -> None:
    """``P.1.1.z`` is not "somewhere under P.1.1": an unknown name is unknown."""
    result = canonical_labels(["P.1.1.z"], clade_set(tmp_path), allow_unmapped=True)
    assert "P.1.1.z" in result.unmapped


def test_an_older_name_shared_by_two_clades_is_reported_not_chosen(tmp_path: Path) -> None:
    """Upstream gives sibling subclades one older name where the old clade covered both;
    their parent would be too coarse and either sibling a guess."""
    shared = "name: legacy-1\nalias_of: P.2\nparent: none\nrepresentatives: []"
    clades = clade_set(tmp_path, extra_legacy={"legacy-1": shared})
    assert older_names(clades)["legacy-1"] == ("P.1", "P.2")
    result = canonical_labels(["legacy-1", "P.1 (legacy-1)"], clades, allow_unmapped=True)
    assert "shared by P.1, P.2" in result.unmapped["legacy-1"]
    assert result.clade("P.1 (legacy-1)") == "P.1"


def test_a_legacy_alias_to_an_undefined_clade_is_an_error(tmp_path: Path) -> None:
    dangling = "name: S\nalias_of: P.7\nparent: none\nrepresentatives: []"
    with pytest.raises(LabelError, match="'S' aliases 'P.7'"):
        older_names(clade_set(tmp_path, extra_legacy={"S": dangling}))


def test_an_unaliased_legacy_file_a_subclade_claims_is_that_subclade(tmp_path: Path) -> None:
    """Upstream publishes some older clades twice: named on the subclade (``clade:``) and
    as a legacy file that aliases nothing. That is one fact, not a collision."""
    own = "name: legacy-1\nparent: none\nrepresentatives: []"
    result = canonical_labels(["legacy-1"], clade_set(tmp_path, extra_legacy={"legacy-1": own}))
    assert result.resolved["legacy-1"].clade == "P.1"
    assert result.resolved["legacy-1"].route == "legacy"
