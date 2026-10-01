"""af.clades' per-subtype facts come from the shared subtype table, and are checked there."""

from __future__ import annotations

import copy
import tomllib
from importlib import resources
from typing import Any

import pytest

from af.clades.coordinates import Coordinates
from af.clades.store import CladeStoreError, dataset_for
from af.clades.subtypes import (
    CladeSubtypeError,
    clade_dataset,
    clade_facts,
    clade_subtypes,
    coordinates_for,
    nomenclature_repository,
    subtype_for_dataset,
)
from af.util.subtypes import Subtypes


def packaged() -> dict[str, Any]:
    return tomllib.loads(resources.files("af").joinpath("subtypes.toml").read_text())


def table(**changes: Any) -> Subtypes:
    """The packaged table with ``<key>=<clades section or None>`` replaced (None: removed)."""
    data = copy.deepcopy(packaged())
    for key, section in changes.items():
        if section is None:
            data["subtype"][key].pop("clades", None)
        else:
            data["subtype"][key]["clades"] = section
    return Subtypes(data, "test table")


def test_the_packaged_table_carries_every_subtypes_clade_facts() -> None:
    assert clade_subtypes() == ("A(H1N1)", "A(H3N2)", "B/Vic")
    assert coordinates_for("B/Vic") == Coordinates(ha1_length=347, nuc_offset=78)
    assert nomenclature_repository("A(H3N2)").endswith("H3N2_HA")
    assert clade_dataset("A(H1N1)") == "h1" and subtype_for_dataset("bvic") == "B/Vic"


def test_a_subtype_without_labels_is_refused_with_its_reason() -> None:
    """Sarah, 25 Sep 2026: no clade labels for B/Yam; asking must fail, saying why."""
    assert not clade_facts("B/Yam").labels
    with pytest.raises(CladeSubtypeError, match="B/Yam has no clade labels: Sarah, 25 Sep 2026"):
        coordinates_for("B/Yam")
    with pytest.raises(CladeStoreError, match="no clade dataset for subtype 'B/Yam'.*25 Sep"):
        dataset_for("B/Yam")
    with pytest.raises(CladeSubtypeError, match="no labelled subtype has clade dataset 'byam'"):
        subtype_for_dataset("byam")


def test_an_unlisted_subtype_is_refused() -> None:
    with pytest.raises(CladeSubtypeError, match="no subtype 'A\\(H5N1\\)'"):
        clade_facts("A(H5N1)")


LABELLED = {"labels": True, "repository": "r", "ha1_length": 329, "nuc_offset": 65, "source": "s"}


@pytest.mark.parametrize(
    ("section", "message"),
    [
        (None, r"\[subtype.h3.clades\] is missing"),
        ({**LABELLED, "colour": "x"}, "unknown key"),
        ({k: v for k, v in LABELLED.items() if k != "source"}, "no source"),
        ({k: v for k, v in LABELLED.items() if k != "labels"}, "labels must be true or false"),
        ({k: v for k, v in LABELLED.items() if k != "nuc_offset"}, "needs nuc_offset"),
        ({**LABELLED, "ha1_length": 0}, "ha1_length must be a positive whole number"),
        ({**LABELLED, "ha1_length": "329"}, "ha1_length must be a positive whole number"),
        ({"labels": False, "repository": "r", "source": "s"}, "labels = false, but repository"),
    ],
)
def test_a_malformed_section_is_refused(section: Any, message: str) -> None:
    """Every row is validated together, so the error comes whichever subtype is asked for."""
    with pytest.raises(CladeSubtypeError, match=message):
        clade_facts("A(H1N1)", table(h3=section))
