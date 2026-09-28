"""af-table-1 files written before epi_isl/sequence_pairing existed still verify, and still
fail when any stored value has changed (the hash is an artefact check, design rule 3)."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

from af.tables.model import Table, canonical_json

from .test_cdc import read, row

LATER_FIELDS = ("epi_isl", "sequence_pairing", "passage_class")


def early_file(table: Table) -> dict[str, Any]:
    """The table as an af-table-1 file written before the later fields: they are absent from
    each antigen and serum, and the hash was computed without them."""
    d = table.to_json()
    d.pop("content_hash")
    d.pop("provenance")
    d["format"] = "af-table-1"
    for item in (*d["antigens"], *d["sera"]):
        for key in LATER_FIELDS:
            del item[key]
    d["content_hash"] = hashlib.sha256(canonical_json(d).encode()).hexdigest()
    d["provenance"] = table.provenance
    return d


@pytest.fixture
def old_file(tmp_path) -> dict[str, Any]:
    (table,) = read(tmp_path, [row()]).tables
    return early_file(table)


def test_a_file_from_before_the_later_fields_verifies(old_file):
    back = Table.from_json(copy.deepcopy(old_file))
    assert back.antigens[0].epi_isl == "" and back.sera[0].sequence_pairing == ""


def test_a_changed_titre_still_fails(old_file):
    old_file["titres"][0][0] = ["5120"]
    with pytest.raises(ValueError, match="stored hash"):
        Table.from_json(old_file)


def test_a_changed_name_still_fails(old_file):
    old_file["antigens"][0]["name"] = "A(H3N2)/EXAMPLEOTHER/1/2029"
    with pytest.raises(ValueError, match="stored hash"):
        Table.from_json(old_file)


def test_a_later_field_present_and_set_still_fails(old_file):
    # the field is in the file, so it is hashed: a value the hash did not cover is caught
    old_file["antigens"][0]["epi_isl"] = "EPI_ISL_000001"
    with pytest.raises(ValueError, match="stored hash"):
        Table.from_json(old_file)


def test_the_current_format_does_not_forgive_a_missing_field(tmp_path):
    (table,) = read(tmp_path, [row()]).tables
    d = table.to_json()
    del d["antigens"][0]["epi_isl"]
    d["antigens"][0]["name"] = "A(H3N2)/EXAMPLEOTHER/1/2029"
    with pytest.raises(ValueError, match="stored hash"):
        Table.from_json(d)
