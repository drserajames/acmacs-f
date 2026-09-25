"""CDC TSV reader, on synthetic rows (invented strains; no real data)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from af.tables import cdc, identity
from af.tables.model import Table
from af.tables.rules import Rules

from .conftest import write_rules


def row(**kw: str) -> dict[str, str]:
    base = dict.fromkeys(cdc.COLUMNS, "")
    base.update(
        test_id="1",
        test_date="2030-01-02",
        test_file="t.xlsx",
        test_protocol="hi_protocol",
        test_subtype="H3",
        ag_position="1",
        ag_cdc_id="100",
        ag_strain_name="A/EXAMPLETOWN/01/2029",
        ag_passage="S1",
        ag_collection_date="2029-12-01",
        ag_date_harvested="2029-12-20",
        ag_type="test",
        ag_do_not_report="FALSE",
        sr_position="A",
        sr_strain_name="A/EXAMPLEREF/2/2028",
        sr_passage="C1S1",
        sr_lot="T29-001",
        sr_boosted="FALSE",
        sr_do_not_report="FALSE",
        titer_reportable="TRUE",
        titer_error="FALSE",
        titer_value="160",
    )
    base.update(kw)
    return base


def write_tsv(path: Path, rows: list[dict[str, str]]) -> Path:
    lines = ["\t".join(cdc.COLUMNS)] + ["\t".join(r[c] for c in cdc.COLUMNS) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def read(tmp_path: Path, rows: list[dict[str, str]], **kw) -> cdc.ReadResult:
    return cdc.read(
        write_tsv(tmp_path / "cdc.tsv", rows), Rules(write_rules(tmp_path / "rules")), **kw
    )


def test_one_titre(tmp_path):
    res = read(tmp_path, [row()])
    assert res.errors == []
    (t,) = res.tables
    assert (t.group, t.date, t.rbc) == ("h3-hi-guinea-pig-cdc", "2030-01-02", "guinea-pig")
    assert t.antigens[0].name == "A(H3N2)/EXAMPLETOWN/1/2029"
    assert t.antigens[0].passage == "SIAT1" and t.antigens[0].passage_date == "2029-12-20"
    assert t.sera[0].serum_id == "CDC T29-001" and t.sera[0].passage == "MDCK1SIAT1"
    assert t.titres == [[["160"]]]


@pytest.mark.parametrize(
    ("protocol", "value", "expected"),
    [
        ("hi_protocol", "5", "<10"),
        ("hint_protocol", "0", "<10"),
        ("hint_protocol", "7", "<10"),
        ("hint_protocol", "5", "<10"),
        ("hint_protocol", "57", "57"),
    ],
)
def test_titre_tokens_t18(tmp_path, protocol, value, expected):
    res = read(tmp_path, [row(test_protocol=protocol, titer_value=value)])
    assert res.tables[0].titres == [[[expected]]]


def test_hi_zero_is_an_error_not_a_guess(tmp_path):
    res = read(tmp_path, [row(titer_value="0")])
    assert res.tables == [] and "matches no titre_tokens rule" in res.errors[0]


@pytest.mark.parametrize(
    "flag",
    [
        ("ag_do_not_report", "TRUE"),
        ("sr_do_not_report", "TRUE"),
        ("titer_reportable", "FALSE"),
        ("titer_error", "TRUE"),
    ],
)
def test_flagged_rows_dropped_and_counted_t21(tmp_path, flag):
    rows = [
        row(),
        row(
            ag_position="2",
            ag_cdc_id="101",
            ag_strain_name="A/EXAMPLETOWN/2/2029",
            titer_value="80",
            **{flag[0]: flag[1]},
        ),
    ]
    (t,) = read(tmp_path, rows).tables
    assert [a.name for a in t.antigens] == ["A(H3N2)/EXAMPLETOWN/1/2029"]
    assert t.dropped["rows: CDC not-for-use"] == 1 and t.dropped[f"flag {flag[0]}"] == 1
    (kept,) = read(tmp_path, rows, drop_flagged=False).tables
    assert len(kept.antigens) == 2


def test_repeats_kept_as_readings_not_merged(tmp_path):
    rows = [row(titer_value="40"), row(titer_value="80")]
    assert read(tmp_path, rows).tables[0].titres == [[["40", "80"]]]


def test_pool_serum_dropped_mouse_tagged_t20(tmp_path):
    rows = [
        row(),
        row(sr_position="B", sr_lot="29/30 H3-CELL HUMAN POOL", titer_value="40"),
        row(sr_position="C", sr_lot="29MouseS0001", titer_value="20"),
    ]
    (t,) = read(tmp_path, rows).tables
    assert [s.serum_id for s in t.sera] == ["CDC T29-001", "CDC 29MouseS0001"]
    assert t.sera[1].species == "MOUSE" and t.dropped["sera: control"] == 1


def test_reassortant_and_annotation_after_year(tmp_path):
    rows = [
        row(ag_strain_name="A/EXAMPLETOWN/1/2029 X-999A"),
        row(ag_position="2", ag_cdc_id="7", ag_strain_name="A/EXAMPLETOWN/03/2029-LAB-J99J"),
    ]
    (t,) = read(tmp_path, rows).tables
    assert (t.antigens[0].name, t.antigens[0].reassortant) == (
        "A(H3N2)/EXAMPLETOWN/1/2029",
        "NYMC-999A",
    )
    assert (t.antigens[1].name, t.antigens[1].annotations) == (
        "A(H3N2)/EXAMPLETOWN/3/2029",
        ["LAB-J99J"],
    )


def test_non_iso_date_is_an_error(tmp_path):
    assert "not an ISO date" in read(tmp_path, [row(test_date="01/02/2030")]).errors[0]


def test_unknown_column_is_fatal(tmp_path):
    path = write_tsv(tmp_path / "cdc.tsv", [row()])
    path.write_text(path.read_text().replace("titer_logfold", "new_column"))
    with pytest.raises(cdc.CDCFormatError, match="unknown columns"):
        cdc.read(path, Rules(write_rules(tmp_path / "rules")))


def test_same_day_tests_get_stable_suffixes(tmp_path):
    first = read(tmp_path, [row(test_id="20")]).tables
    identity.assign(first, None)
    manifest = identity.Manifest.from_tables(first, [])
    # A second test on the same day arrives later, with a *lower* test_id.
    both = read(tmp_path, [row(test_id="20"), row(test_id="7", titer_value="320")]).tables
    identity.assign(both, manifest)
    ids = {t.source_key: t.table_id for t in both}
    assert ids == {
        "CDC test_id 20": "h3-hi-guinea-pig-cdc-20300102",
        "CDC test_id 7": "h3-hi-guinea-pig-cdc-20300102.2",
    }


def test_diff_new_changed_removed_restart(tmp_path):
    old = read(tmp_path, [row(test_id="1"), row(test_id="2", test_date="2030-02-01")]).tables
    identity.assign(old, None)
    m_old = identity.Manifest.from_tables(old, [])
    new = read(
        tmp_path,
        [
            row(test_id="1", titer_value="320"),  # changed
            row(test_id="3", test_date="2030-03-01"),
        ],
    ).tables  # new; test 2 removed
    identity.assign(new, m_old)
    d = identity.diff(m_old, identity.Manifest.from_tables(new, [], m_old))
    assert d.changed == ["h3-hi-guinea-pig-cdc-20300102"]
    assert d.new == ["h3-hi-guinea-pig-cdc-20300301"]
    assert d.removed == ["h3-hi-guinea-pig-cdc-20300201"]
    assert d.restart == {"h3-hi-guinea-pig-cdc": "2030-01-02"}


def test_json_round_trip_checks_hash(tmp_path):
    (t,) = read(tmp_path, [row()]).tables
    identity.assign([t], None)
    d = t.to_json()
    assert Table.from_json(d).content_hash() == t.content_hash()
    d["titres"][0][0] = ["640"]
    with pytest.raises(ValueError, match="stored hash"):
        Table.from_json(d)


def test_unmatched_rule_is_reported(tmp_path, rules_dir):
    with (rules_dir / "titre_tokens.tsv").open("a") as f:
        f.write("CDC\tFRA\texact\t3\t<10\tinvented\ttest\t2030-01-01\t\n")
    rules = Rules(rules_dir)
    cdc.read(write_tsv(tmp_path / "cdc.tsv", [row()]), rules)
    assert any(r["pattern"] == "3" for r in rules.titre_tokens.unmatched())


def test_metadata_only_change_does_not_restart(tmp_path):
    old = read(tmp_path, [row(ag_epi_isolate_id="")]).tables
    identity.assign(old, None)
    m_old = identity.Manifest.from_tables(old, [])
    new = read(tmp_path, [row(ag_epi_isolate_id="EPI_ISL_1")]).tables  # sequence linked later
    identity.assign(new, m_old)
    d = identity.diff(m_old, identity.Manifest.from_tables(new, [], m_old))
    assert (d.changed, d.metadata, d.restart) == ([], ["h3-hi-guinea-pig-cdc-20300102"], {})


def test_parsed_date_and_order_key(tmp_path):
    tables = read(tmp_path, [row(test_id="2"), row(test_id="1", titer_value="320")]).tables
    identity.assign(tables, None)
    by_suffix = sorted(tables, key=lambda t: t.order_key)
    assert [t.order_key for t in by_suffix] == [(dt.date(2030, 1, 2), 1), (dt.date(2030, 1, 2), 2)]


def test_check_reports_bad_shapes_without_crashing(tmp_path):
    (t,) = read(
        tmp_path,
        [row(), row(ag_position="2", ag_cdc_id="101", ag_strain_name="A/EXAMPLETOWN/2/2029")],
    ).tables
    t.titres = [[["40"]]]  # two antigens, one row
    assert t.check() == ["1 titre rows for 2 antigens"]
    t.titres = [[["40"]], []]  # a short row
    assert t.check() == ["titre row 1: 0 cells for 1 sera"]


def _typo_rows(values: list[str]) -> list[dict[str, str]]:
    sera = [("A", "T29-001"), ("B", "T29-002"), ("C", "T29-003"), ("D", "T29-004")]
    return (
        [row()]
        + [
            row(
                ag_position="2",
                ag_cdc_id="7",
                ag_strain_name="B/EXAMPLETYPO/7/2029",
                sr_position=pos,
                sr_lot=lot,
                titer_value=value,
            )
            for (pos, lot), value in zip(sera, values, strict=True)
        ]
        + [row(sr_position=pos, sr_lot=lot) for pos, lot in sera[1:]]
    )


def test_named_rename_with_titres_that_agree(tmp_path):
    res = read(tmp_path, _typo_rows(["640", "320", "20", "1280"]))
    assert res.errors == []
    renamed = res.tables[0].antigens[1]
    assert (
        renamed.name == "A(H3N2)/EXAMPLETYPO/7/2029" and renamed.raw_name == "B/EXAMPLETYPO/7/2029"
    )
    assert renamed.source["alias_rule"] == "strain_aliases.tsv:2"
    assert any("renamed" in w for w in res.tables[0].warnings)


def test_rename_refused_when_titres_disagree(tmp_path):
    res = read(tmp_path, _typo_rows(["5", "20", "5", "40"]))  # 1 of 4 reads >= 40
    assert any("renamed by strain_aliases.tsv:2" in e for e in res.errors)
    assert any("1/4 cells read >= 40" in e for e in res.errors)


def test_rule_locations_name_the_real_file_line(rules_dir):
    path = rules_dir / "titre_tokens.tsv"
    path.write_text("# a comment\n\n" + path.read_text())
    assert [r.where for r in Rules(rules_dir).titre_tokens.rules][:1] == ["titre_tokens.tsv:4"]
