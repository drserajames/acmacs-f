"""lineage(): root-first ancestry that keeps "unknown" (None) and "no clade" ("") apart."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades.nomenclature import NomenclatureError, lineage

from .synthetic import build_clone, load_synthetic


def test_a_clade_gives_its_ancestors_root_first(tmp_path: Path) -> None:
    clade_set = load_synthetic(build_clone(tmp_path).parent)
    assert lineage(clade_set, "P.1.1") == ("P", "P.1", "P.1.1")
    assert lineage(clade_set, "P") == ("P",)


def test_unknown_and_no_clade_stay_apart(tmp_path: Path) -> None:
    """None (we do not know) and "" (we know there is none) must not be collapsed."""
    clade_set = load_synthetic(build_clone(tmp_path).parent)
    assert lineage(clade_set, None) is None
    assert lineage(clade_set, "") == ()


def test_a_name_that_is_not_a_clade_is_an_error(tmp_path: Path) -> None:
    clade_set = load_synthetic(build_clone(tmp_path).parent)
    with pytest.raises(NomenclatureError, match="'Q' is not a clade"):
        lineage(clade_set, "Q")  # an old name (a pointer), not a clade
