"""Geo's clade colouring: the labelling clade set, and the user's schemes read at run time."""

from pathlib import Path

import pytest

from af.clades.importer import UserCladesError
from af.serology.outputs import clade_colouring
from af.util.artefacts import sha256_path
from tests.clades.synthetic import load_synthetic
from tests.clades.test_clade_set_for import clones, published
from tests.clades.test_importer import acmacs_data


def test_colouring_uses_the_labelling_clade_set_and_the_users_scheme(tmp_path: Path) -> None:
    directory = clones(tmp_path)
    store, _ = published(tmp_path, load_synthetic(directory))
    data = acmacs_data(tmp_path)
    colouring = clade_colouring(store, directory, data, {"A(H3N2)": "clades-v1"})
    h3 = colouring["A(H3N2)"]
    assert h3.scheme.keys == ("P.1", "P.1 20V")
    assert h3.clade_set.version == load_synthetic(directory).version
    assert h3.group_set is not None and "P.1 20V" in h3.group_set.names
    # the scheme's sources travel with it, for the report's provenance
    assert [(i.path.name, i.sha256) for i in h3.inputs] == [
        (name, sha256_path(data / name)) for name in ("semantic_clades.py", "clades.json")
    ]


def test_unknown_schemes_and_subtypes_are_errors(tmp_path: Path) -> None:
    directory = clones(tmp_path)
    store, _ = published(tmp_path, load_synthetic(directory))
    data = acmacs_data(tmp_path)
    with pytest.raises(UserCladesError, match="no colour scheme 'clades-v9'"):
        clade_colouring(store, directory, data, {"A(H3N2)": "clades-v9"})
    with pytest.raises(ValueError, match="no clade set for table subtype"):
        clade_colouring(store, directory, data, {"A(H5N1)": "clades-v1"})


def test_maps_read_once_and_colour_per_map(tmp_path: Path) -> None:
    """The maps' use: one read of the user's tables, a scheme chosen per figure."""
    from af.serology.outputs import read_clade_tables, subtype_colouring

    directory = clones(tmp_path)
    store, _ = published(tmp_path, load_synthetic(directory))
    user = read_clade_tables(store, directory, acmacs_data(tmp_path), ["A(H3N2)"])
    first = subtype_colouring(store, directory, user, "A(H3N2)", "clades-v1")
    second = subtype_colouring(store, directory, user, "A(H3N2)", "clades-v1")
    assert first.scheme is second.scheme and first.inputs == second.inputs == user.inputs
    with pytest.raises(ValueError, match="no clade set for table subtype"):
        subtype_colouring(store, directory, user, "A(H5N1)", "clades-v1")
