"""strain_aliases canonicals are names as written: a group reference in one is refused."""

from __future__ import annotations

import pytest

from af.tables.rules import Rules

from .synthetic_rules import META, write_rules


def test_a_group_reference_in_a_canonical_is_an_error(tmp_path):
    directory = write_rules(tmp_path / "rules")
    path = directory / "strain_aliases.tsv"
    row = "LABX\tB\tantigen\tregex\tA/(.*)\tB/\\1\t40\t0.5" + META + "yes"
    path.write_text(path.read_text() + row + "\n")
    with pytest.raises(ValueError, match="group reference"):
        Rules(directory)


def test_the_synthetic_rules_load(tmp_path):
    Rules(write_rules(tmp_path / "rules"))


def test_a_canonical_that_does_not_parse_is_an_error(tmp_path):
    directory = write_rules(tmp_path / "rules")
    path = directory / "strain_aliases.tsv"
    row = "LABX\tB\tantigen\texact\tB/EXAMPLE 1/29\tB/EXAMPLE 1 29\t40\t0.5" + META + "yes"
    path.write_text(path.read_text() + row + "\n")
    with pytest.raises(ValueError, match="does not parse"):
        Rules(directory)


def test_an_antigen_alias_needs_its_titre_thresholds(tmp_path):
    directory = write_rules(tmp_path / "rules")
    path = directory / "strain_aliases.tsv"
    row = "LABX\tB\tantigen\texact\tB/EXAMPLE/1/30\tB/EXAMPLE/1/2030\t\t" + META + "yes"
    path.write_text(path.read_text() + row + "\n")
    with pytest.raises(ValueError, match="must be numbers"):
        Rules(directory)


def test_a_serum_ids_canonical_must_be_a_plain_lot(tmp_path):
    directory = write_rules(tmp_path / "rules")
    path = directory / "serum_ids.tsv"
    row = "LABX\tregex\tNo Lot (.*)\tF0-9\t\\1\t40\t0.5" + META + "yes"
    path.write_text(path.read_text() + row + "\n")
    with pytest.raises(ValueError, match="not a plain lot"):
        Rules(directory)
