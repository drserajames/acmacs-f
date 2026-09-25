"""Importing today's tables, and telling the user what does not survive the move."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.clades.groups import GroupError, GroupSet, load_groups
from af.clades.importer import (
    ImportError_,
    clades_json_names,
    dead_row_report,
    import_semantic_clades,
    org_table_to_dict,
    read_semantic_clades,
    write_colour_schemes,
    write_groups,
)
from af.clades.nomenclature import CladeSet

from .synthetic import build_clone, load_synthetic

SUBTYPE = "A(H3N2)"
#: The old file's shape: a module of Org tables needing ae on the path, which the importer
#: supplies a stand-in for. Clade names here are the synthetic ones.
MODULE = '''
from ae.utils.org import org_table_to_dict

sData = {
    "A(H3N2)": {
        "attributes": org_table_to_dict("""
| name     | clade | aa  |
|----------+-------+-----|
| P.1 20V  | P.1   | 20V |
| 21W      |       | 21W |
| GONE 20V | P.9   | 20V |
"""),
        "clades-v1": org_table_to_dict("""
| name    | legend  | color   |
|---------+---------+---------|
| P.1     | P.1     | #112233 |
| P.1 20V | P.1 20V | #445566 |
| P.9     | P.9     | #778899 |
"""),
    },
}
'''


def synthetic(tmp_path: Path) -> dict[str, CladeSet]:
    return {SUBTYPE: load_synthetic(build_clone(tmp_path).parent)}


def write_module(tmp_path: Path, text: str = MODULE) -> Path:
    path = tmp_path / "semantic_clades.py"
    path.write_text(text)
    return path


def test_parses_an_org_table() -> None:
    rows = org_table_to_dict("| a | b |\n|---+---|\n| 1 | 2 |\n")
    assert rows == [{"a": "1", "b": "2"}]


def test_reads_the_module_without_ae_installed(tmp_path: Path) -> None:
    """The old file imports ae.utils.org. Parsing it with a regular expression is how
    three other readers of it broke on a layout change; the importer supplies a stand-in
    and lets Python do the parsing."""
    data = read_semantic_clades(write_module(tmp_path))
    assert set(data) == {SUBTYPE}
    assert len(data[SUBTYPE]["attributes"]) == 3


def test_a_missing_module_is_fatal(tmp_path: Path) -> None:
    with pytest.raises(ImportError_, match="not found"):
        read_semantic_clades(tmp_path / "absent.py")


def test_carries_groups_across(tmp_path: Path) -> None:
    report = import_semantic_clades(write_module(tmp_path), synthetic(tmp_path))
    groups = report.groups[SUBTYPE]
    assert groups.names == ("P.1 20V", "21W")
    assert groups.groups[0].anchor == "P.1"
    assert groups.groups[1].anchor is None


def test_reports_a_row_that_names_a_clade_nothing_defines(tmp_path: Path) -> None:
    """These are the rows that have been labelling nothing, silently, for years."""
    report = import_semantic_clades(write_module(tmp_path), synthetic(tmp_path))
    dead = [row for row in report.dead if row.table == "attributes"]
    assert len(dead) == 1
    assert dead[0].name == "GONE 20V"
    assert "matching nothing" in dead[0].reason


def test_a_legacy_name_is_not_called_dead(tmp_path: Path) -> None:
    """A name the old system defines but the nomenclature does not needs a local
    definition; calling it dead would overstate the problem sevenfold."""
    report = import_semantic_clades(
        write_module(tmp_path), synthetic(tmp_path), defined_locally={SUBTYPE: {"P.9"}}
    )
    assert not [row for row in report.dead if row.table == "attributes"]
    assert [row.name for row in report.needs_local] == ["GONE 20V", "P.9"]


def test_a_colour_row_naming_a_group_is_kept(tmp_path: Path) -> None:
    """A colour key may name a group rather than a clade; checking the clade set first
    reported every such row as dead."""
    report = import_semantic_clades(write_module(tmp_path), synthetic(tmp_path))
    rows = report.colour_rows[(SUBTYPE, "clades-v1")]
    assert [row["key"] for row in rows] == ["P.1", "P.1 20V"]


def test_writes_the_new_tables(tmp_path: Path) -> None:
    report = import_semantic_clades(write_module(tmp_path), synthetic(tmp_path))
    groups_file = tmp_path / "groups.tsv"
    assert write_groups(report, groups_file) == 2
    header, *lines = groups_file.read_text().splitlines()
    assert header.split("\t") == ["subtype", "group", "anchor", "substitutions", "note"]
    assert lines[0].split("\t")[:4] == [SUBTYPE, "P.1 20V", "P.1", "20V"]
    written = write_colour_schemes(report, tmp_path / "colours")
    assert written == {"h3/clades-v1.tsv": 2}


def test_the_written_tables_load_back(tmp_path: Path) -> None:
    """The importer's output must be valid input, or the migration stalls at the first
    file the loader rejects."""
    from af.clades.colours import load_colour_scheme
    from af.clades.groups import load_groups

    sets = synthetic(tmp_path)
    report = import_semantic_clades(write_module(tmp_path), sets)
    groups_file = tmp_path / "groups.tsv"
    write_groups(report, groups_file)
    write_colour_schemes(report, tmp_path / "colours")
    group_set = load_groups(groups_file, sets)[SUBTYPE]
    scheme = load_colour_scheme(
        tmp_path / "colours" / "h3" / "clades-v1.tsv", SUBTYPE, sets[SUBTYPE], group_set
    )
    assert scheme.keys == ("P.1", "P.1 20V")


def test_dead_row_report_is_readable(tmp_path: Path) -> None:
    report = import_semantic_clades(write_module(tmp_path), synthetic(tmp_path))
    text = dead_row_report(report.dead)
    assert "GONE 20V" in text
    assert dead_row_report([]) == "no dead rows"


def test_clades_json_names_reads_entry_names(tmp_path: Path) -> None:
    path = tmp_path / "clades.json"
    path.write_text('{"  version": "clades-v2", "A(H3N2)": [{"N": "P.9", "aa": "20V"}, "note"]}')
    assert clades_json_names(path) == {"A(H3N2)": {"P.9"}}


REPEATS = MODULE.replace(
    "| GONE 20V | P.9   | 20V |\n",
    "| GONE 20V | P.9   | 20V |\n| 21W      |       | 21W |\n| P.1 20V  | P.1   | 22K |\n",
)


def test_a_verbatim_repeat_is_kept_once_and_listed(tmp_path: Path) -> None:
    """The old file repeats some attribute rows exactly. Written twice, the groups file
    would be refused by load_groups; kept once, the repeat is still reported."""
    report = import_semantic_clades(write_module(tmp_path, REPEATS), synthetic(tmp_path))
    assert report.groups[SUBTYPE].names == ("P.1 20V", "21W")
    assert [(row.row, row.name, row.reason) for row in report.repeated] == [
        (4, "21W", "repeats row 2")
    ]


def test_a_name_redefined_differently_is_dead(tmp_path: Path) -> None:
    report = import_semantic_clades(write_module(tmp_path, REPEATS), synthetic(tmp_path))
    [conflict] = [row for row in report.dead if row.name == "P.1 20V"]
    assert conflict.row == 5 and "differently" in conflict.reason


def test_repeats_do_not_break_the_written_groups_file(tmp_path: Path) -> None:
    clade_sets = synthetic(tmp_path)
    report = import_semantic_clades(write_module(tmp_path, REPEATS), clade_sets)
    path = tmp_path / "groups.tsv"
    write_groups(report, path)
    assert load_groups(path, clade_sets)[SUBTYPE].names == ("P.1 20V", "21W")


def test_a_group_set_refuses_two_groups_of_one_name(tmp_path: Path) -> None:
    group = import_semantic_clades(write_module(tmp_path), synthetic(tmp_path)).groups[SUBTYPE]
    with pytest.raises(GroupError, match="duplicate group 'P.1 20V'"):
        GroupSet(SUBTYPE, (*group.groups, group.groups[0]))
