import datetime
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from af.serology import query
from af.serology.rows import IdentityRules
from af.serology.store import StoreError, build
from af.tables.model import Table


def _tables(syn: Any) -> list[Table]:
    """Two groups, two years; one antigen preparation in two tables, two labs."""
    shared = {"name": syn.virus("Somewhere", 1), "passage": "MDCK1", "date": "2021-01-05"}
    return [
        syn.table(
            "a-2021-02-01",
            [shared, {"name": syn.virus("Place", 7, prefix="A(H3N2)"), "passage": "", "date": ""}],
            [{"name": syn.virus("Elsewhere", 2), "serum_id": "S-1"}],
            [[["80"]], [["40"]]],
            date="2021-02-01",
            lab="LABY",
            group="h3-hi-laby",
        ),
        syn.table(
            "x-2021-03-04",
            [shared],
            [{"name": syn.virus("Elsewhere", 2), "serum_id": "S-1"}],
            [[["160", "320"]]],
        ),
        syn.table(
            "x-2022-01-10",
            [{"name": syn.virus("Somewhere", 5, 2022), "passage": "SIAT1", "date": "2021-12-30"}],
            [{"name": syn.virus("Elsewhere", 2), "serum_id": "S-1"}],
            [[["<10"]]],
            date="2022-01-10",
        ),
    ]


def test_build_writes_partitions_and_counts(tmp_path: Path, syn: Any) -> None:
    report = build(_tables(syn), tmp_path / "v1", syn.rules)
    assert sorted(report.rebuilt) == ["h3-hi-labx/2021", "h3-hi-labx/2022", "h3-hi-laby/2021"]
    assert (report.tables, report.antigens, report.sera, report.readings) == (3, 4, 3, 5)
    assert report.without_identity == {"antigens": 1, "sera": 0}
    saved = json.loads((tmp_path / "v1" / "partitions.json").read_text())
    assert saved["identity_version"] == "test-1"


def test_rebuild_reuses_unchanged_partitions_by_hard_link(tmp_path: Path, syn: Any) -> None:
    tables = _tables(syn)
    build(tables, tmp_path / "v1", syn.rules)
    changed = list(tables)
    changed[2] = replace(changed[2], titres=[[[">1280"]]])
    report = build(changed, tmp_path / "v2", syn.rules, previous=tmp_path / "v1")
    assert report.rebuilt == ["h3-hi-labx/2022"]
    assert sorted(report.reused) == ["h3-hi-labx/2021", "h3-hi-laby/2021"]
    old = tmp_path / "v1/partitions/h3-hi-labx/2021/titres.parquet"
    new = tmp_path / "v2/partitions/h3-hi-labx/2021/titres.parquet"
    assert old.stat().st_ino == new.stat().st_ino
    assert report.readings == 5  # counts carried over for reused partitions


def test_new_identity_rules_rebuild_everything(tmp_path: Path, syn: Any) -> None:
    build(_tables(syn), tmp_path / "v1", syn.rules)
    rules2 = IdentityRules(syn.rules.antigen, syn.rules.serum, version="test-2")
    report = build(_tables(syn), tmp_path / "v2", rules2, previous=tmp_path / "v1")
    assert report.reused == []


def test_refuses_duplicates_empty_input_and_nonempty_output(tmp_path: Path, syn: Any) -> None:
    tables = _tables(syn)
    with pytest.raises(StoreError, match="given twice"):
        build([tables[0], tables[0]], tmp_path / "a", syn.rules)
    with pytest.raises(StoreError, match="no tables"):
        build([], tmp_path / "b", syn.rules)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "x").write_text("x")
    with pytest.raises(StoreError, match="not empty"):
        build(tables, tmp_path / "c", syn.rules)


def test_queries(tmp_path: Path, syn: Any) -> None:
    build(_tables(syn), tmp_path / "v1", syn.rules)
    con = query.connect(tmp_path / "v1")
    preps = {p.name: p for p in query.preparations(con)}
    shared = preps[syn.virus("Somewhere", 1)]
    # earliest table is LABY's (1 Feb) even though LABX sorts first alphabetically
    assert shared.first_lab == "LABY"
    assert shared.collection_date == datetime.date(2021, 1, 5)
    uses = query.serum_uses(con)
    # one serum (same identity) used in all three tables
    assert len({u.serum_key for u in uses}) == 1
    assert sorted((u.lab, u.table_date) for u in uses) == [
        ("LABX", datetime.date(2021, 3, 4)),
        ("LABX", datetime.date(2022, 1, 10)),
        ("LABY", datetime.date(2021, 2, 1)),
    ]
    since = query.strains_with_titres(con, datetime.date(2021, 3, 1))
    assert {name for _, name in since} == {
        syn.virus("Somewhere", 1),
        syn.virus("Somewhere", 5, 2022),
    }
    month = query.centre_month(con, "LABX", 2022, 1)
    assert [t["table_id"] for t in month] == ["x-2022-01-10"]
    assert query.centre_month(con, "LABX", 2021, 12) == []
