"""Importing today's tables, and telling the user what does not survive the move."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------- run-time reader


def acmacs_data(tmp_path: Path, module: str = MODULE) -> Path:
    """A stand-in acmacs-data directory: the module plus a clades.json naming P.9 locally."""
    directory = tmp_path / "acmacs-data"
    directory.mkdir(exist_ok=True)
    (directory / "semantic_clades.py").write_text(module)
    (directory / "clades.json").write_text('{"A(H3N2)": [{"N": "P.9", "aa": "20V"}]}')
    return directory


def test_reads_schemes_and_groups_at_run_time(tmp_path: Path) -> None:
    from af.clades.importer import read_user_clades

    user = read_user_clades(acmacs_data(tmp_path), synthetic(tmp_path))
    assert user.scheme(SUBTYPE, "clades-v1").keys == ("P.1", "P.1 20V")
    group_set = user.group_set(SUBTYPE)
    assert group_set is not None and group_set.names == ("P.1 20V", "21W")
    # P.9 is defined by the old system, so its rows need a local definition; not dead
    assert {row.name for row in user.report.needs_local} == {"GONE 20V", "P.9"}


def test_inputs_carry_both_files_content_hashes(tmp_path: Path) -> None:
    from af.clades.importer import read_user_clades
    from af.util.artefacts import sha256_path

    directory = acmacs_data(tmp_path)
    user = read_user_clades(directory, synthetic(tmp_path))
    assert [(item.path.name, item.sha256) for item in user.inputs] == [
        (name, sha256_path(directory / name)) for name in ("semantic_clades.py", "clades.json")
    ]


def test_an_edit_is_seen_on_the_next_read(tmp_path: Path) -> None:
    """Read at run time, not converted once: an edited colour must reach the next figure,
    and its provenance must change with it."""
    from af.clades.importer import read_user_clades

    sets = synthetic(tmp_path)
    first = read_user_clades(acmacs_data(tmp_path), sets)
    second = read_user_clades(acmacs_data(tmp_path, MODULE.replace("#112233", "#aabbcc")), sets)
    assert first.scheme(SUBTYPE, "clades-v1").entries[0].colour == "#112233"
    assert second.scheme(SUBTYPE, "clades-v1").entries[0].colour == "#aabbcc"
    assert first.inputs[0].sha256 != second.inputs[0].sha256


def test_an_unknown_scheme_or_subtype_is_an_error(tmp_path: Path) -> None:
    from af.clades.importer import UserCladesError, read_user_clades

    user = read_user_clades(acmacs_data(tmp_path), synthetic(tmp_path))
    with pytest.raises(UserCladesError, match=r"no colour scheme 'clades-v9'.*clades-v1"):
        user.scheme(SUBTYPE, "clades-v9")
    # a subtype without a clade set (B/Yamagata, by design) has no schemes: asking fails
    with pytest.raises(UserCladesError, match="no colour schemes for 'B/Yam'"):
        user.scheme("B/Yam", "clades")
    with pytest.raises(UserCladesError, match="no groups for 'B/Yam'"):
        user.group_set("B/Yam")


@pytest.mark.parametrize("missing", ["directory", "semantic_clades.py", "clades.json"])
def test_missing_inputs_are_fatal(tmp_path: Path, missing: str) -> None:
    from af.clades.importer import UserCladesError, read_user_clades

    directory = acmacs_data(tmp_path)
    if missing == "directory":
        directory = tmp_path / "absent"
    else:
        (directory / missing).unlink()
    with pytest.raises(UserCladesError, match="not found"):
        read_user_clades(directory, synthetic(tmp_path))


def test_a_malformed_scheme_row_is_an_error(tmp_path: Path) -> None:
    """Checked as a scheme file is: a colour that is not #rrggbb fails the read."""
    from af.clades.colours import ColourSchemeError
    from af.clades.importer import read_user_clades

    directory = acmacs_data(tmp_path, MODULE.replace("#445566", "red"))
    with pytest.raises(ColourSchemeError, match="colour 'red' is not '#rrggbb'"):
        read_user_clades(directory, synthetic(tmp_path))


def test_a_file_edited_mid_read_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import af.clades.importer as importer

    directory = acmacs_data(tmp_path)
    original = importer.import_semantic_clades

    def edit_while_reading(*args: Any, **kwargs: Any) -> importer.ImportReport:
        report = original(*args, **kwargs)
        (directory / "clades.json").write_text("{}")
        return report

    monkeypatch.setattr(importer, "import_semantic_clades", edit_while_reading)
    with pytest.raises(importer.UserCladesError, match="changed while being read: clades.json"):
        importer.read_user_clades(directory, synthetic(tmp_path))


# ---------------------------------------------------------------- upstream's old names

#: Rows written under Q, the synthetic clone's older name for P.1 (clades/Q.yml, alias_of P.1).
OLD_NAMES = MODULE.replace(
    "| GONE 20V | P.9   | 20V |\n", "| GONE 20V | P.9   | 20V |\n| Q 22K    | Q     | 22K |\n"
).replace(
    "| P.9     | P.9     | #778899 |\n",
    "| P.9     | P.9     | #778899 |\n"
    "| Q 22K   | Q 22K   | #001122 |\n"
    "| Q       | Q       | #334455 |\n",
)


def module_without_p1_colour(text: str) -> str:
    """Q and P.1 are one clade: a scheme listing both is a duplicate key, so drop P.1."""
    return text.replace("| P.1     | P.1     | #112233 |\n", "")


def test_an_old_name_resolves_through_upstreams_alias(tmp_path: Path) -> None:
    """Upstream publishes which subclade an old name is; a row written under it names a
    live clade, so it is neither dead nor waiting on a local definition."""
    clade_sets = synthetic(tmp_path)
    old = module_without_p1_colour(OLD_NAMES)
    report = import_semantic_clades(write_module(tmp_path, old), clade_sets)
    group = {group.name: group for group in report.groups[SUBTYPE]}["Q 22K"]
    assert group.anchor == "P.1"
    keys = [row["key"] for row in report.colour_rows[(SUBTYPE, "clades-v1")]]
    assert keys == ["P.1 20V", "Q 22K", "P.1"]
    assert {row.name for row in report.dead + report.needs_local} == {"GONE 20V", "P.9"}
    # every resolution is counted, attribute anchors and colour keys alike
    assert (SUBTYPE, "Q", "P.1") in report.renamed
    assert report.renamed.count((SUBTYPE, "Q", "P.1")) == 2


def test_caller_names_add_to_upstreams(tmp_path: Path) -> None:
    report = import_semantic_clades(
        write_module(tmp_path, OLD_NAMES.replace("| GONE 20V | P.9 ", "| GONE 20V | P.8 ")),
        synthetic(tmp_path),
        legacy_names={SUBTYPE: {"P.8": "P.2"}},
    )
    names = {group.name: group.anchor for group in report.groups[SUBTYPE]}
    assert names["GONE 20V"] == "P.2" and names["Q 22K"] == "P.1"
