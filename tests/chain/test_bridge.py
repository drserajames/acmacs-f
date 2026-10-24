"""Seed bridge (af.chain.bridge): a seed chart spelled one way, store tables spelled another.

Invented names only (acmacs-f is public). The "seed" spells points as another program might: no
lab code in serum ids, harvest dates in passages, hyphens in place names.
"""

import dataclasses
import datetime
import json
import tomllib

import numpy as np
import pytest

from af.chain.backend import StubOptimiser
from af.chain.bridge import (
    KEY,
    TITRES,
    TITRES_NAME,
    BridgeError,
    bridge_seed,
    declaration_fragment,
    seed_source_dates,
    seed_split,
)
from af.chain.config import (
    ChainConfig,
    ChainConfigError,
    MapOptions,
    SeedChart,
    TableRef,
    load_chain_config,
    tables_from_directory,
)
from af.chain.engine import BRIDGE_RECORD, CHOSEN, run_chain
from af.chain.synthetic import make_tables
from af.chart.ace import read_chart, write_chart
from af.chart.model import Antigen, Chart, Projection, Serum, Titres, empty_table
from af.chart.titre import Titre
from af.report.howmade import DECLARATION_FIELDS

from .sera_markers import select_block

GROUP = "h9-hi-test-lab"
LAB = "LABX"


def d(s: str) -> datetime.date:
    return datetime.datetime.strptime(s, "%Y%m%d").date()


def ref(date: str, suffix: int = 1, path=None) -> TableRef:
    tid = f"{GROUP}-{date}" + ("" if suffix == 1 else f".{suffix}")
    return TableRef(tid, path, d(date), suffix)


# ----------------------------------------------------------------------
# a seed and its tables, built point by point


def table_chart(antigens, sera, rows, date):
    table = empty_table(len(antigens), len(sera))
    for i, row in enumerate(rows):
        for j, t in enumerate(row):
            table[i][j] = Titre.parse(t)
    return Chart({"D": date}, list(antigens), list(sera), Titres(table))


def build(tmp_path, seed_ags, seed_srs, tables, layer_order=None):
    """`tables`: dicts with date, suffix, ags [(seed index or None, Antigen)], srs [(seed index,
    Serum)], rows (titre strings, table order) and optional seed_rows overriding the seed layer
    (keyed by (table row, table column)). Returns (seed chart, [(TableRef, path)])."""
    n_ag, n_sr = len(seed_ags), len(seed_srs)
    full = empty_table(n_ag, n_sr)
    layers, sources, refs = [], [], []
    for t in tables:
        layer = {}
        for i, (ka, _) in enumerate(t["ags"]):
            for j, (ks, _) in enumerate(t["srs"]):
                if ka is None or ks is None:
                    continue
                text = t.get("seed_rows", {}).get((i, j), t["rows"][i][j])
                if text != "*":
                    layer[(ka, ks)] = Titre.parse(text)
                    full[ka][ks] = Titre.parse(text)
        if t.get("in_seed", True):
            layers.append(layer)
            sources.append({"D": t["date"]})
        chart = table_chart(
            [a for _, a in t["ags"]], [s for _, s in t["srs"]], t["rows"], t["date"]
        )
        suffix = t.get("suffix", 1)
        name = f"{GROUP}-{t['date']}" + ("" if suffix == 1 else f".{suffix}")
        path = tmp_path / f"{name}.ace"
        write_chart(chart, path)
        refs.append(TableRef(name, path, d(t["date"]), suffix))
    if layer_order is not None:
        layers = [layers[i] for i in layer_order]
        sources = [sources[i] for i in layer_order]
    rng = np.random.default_rng(5)
    layout = rng.normal(size=(n_ag + n_sr, 2))
    proj = Projection(layout, stress=12.5, transformation=np.array([[0.0, 1.0], [-1.0, 0.0]]))
    seed = Chart(
        {"S": sources}, list(seed_ags), list(seed_srs), Titres(full, layers), projections=[proj]
    )
    return seed, refs


def run_bridge(seed, refs, prefix=LAB):
    split = seed_split(seed_source_dates(seed), refs)
    return bridge_seed(seed, split, prefix)


def ag(name, passage="MDCK1", **kw):
    return Antigen(name=name, passage=passage, **kw)


def sr(name, sid):
    return Serum(name=name, serum_id=sid, passage="E3", species="FERRET")


SEED_SR = [sr("TEST-ALPHA-1", "F001"), sr("TEST-BETA-2", "F002")]
AF_SR = [(0, sr("TEST-ALPHA-1", f"{LAB} F001")), (1, sr("TEST-BETA-2", f"{LAB} F002"))]


def by_index(items, kind):
    return {x["index"]: x for x in items if x["kind"] == kind}


# ----------------------------------------------------------------------
# which tables the seed holds


def test_split_by_date_and_suffix():
    seed_dates = [d("20210110"), d("20210110"), d("20210210")]
    tables = [
        ref("20210105"),  # dated within the seed, absent from it
        ref("20210110"),
        ref("20210110", 2),
        ref("20210110", 3),  # a third table of a date the seed holds twice
        ref("20210210"),
        ref("20210301"),  # after the seed
    ]
    s = seed_split(seed_dates, tables)
    assert [t.table_id for t, _ in s.in_seed] == [f"{GROUP}-20210110", f"{GROUP}-20210110.2",
                                                    f"{GROUP}-20210210"]  # fmt: skip
    assert s.in_seed[0][1] == [0, 1] and s.in_seed[2][1] == [2]
    assert [t.table_id for t in s.af_only] == [f"{GROUP}-20210105", f"{GROUP}-20210110.3"]
    assert [t.table_id for t in s.post_seed] == [f"{GROUP}-20210301"]
    assert s.counts() == {"in_seed": 3, "af_only": 2, "post_seed": 1}


def test_seed_without_sources_is_refused():
    seed = Chart({}, [ag("TEST-A-1")], [], Titres([[]]))
    with pytest.raises(BridgeError, match="no source list"):
        seed_source_dates(seed)


# ----------------------------------------------------------------------
# pairing


def test_key_pairing_drops_the_lab_code_and_the_passage_date(tmp_path):
    seed_ags = [ag("TEST-ALPHA-1", "E3 (2020-03-01)"), ag("TEST-GAMMA-7", "MDCK2")]
    rows = [["640", "160"], ["320", "1280"]]
    seed, refs = build(
        tmp_path, seed_ags, SEED_SR,
        [{"date": "20210115", "rows": rows,
          "ags": [(0, ag("TEST-ALPHA-1", "E3")), (1, ag("TEST-GAMMA-7", "MDCK2"))],
          "srs": AF_SR}],
    )  # fmt: skip
    b = run_bridge(seed, refs)
    conv = by_index(b.converted, "antigen")
    assert conv[0]["how"] == KEY and conv[0]["store"]["passage"] == "E3"
    assert conv[0]["seed"]["passage"] == "E3 (2020-03-01)"
    assert by_index(b.converted, "serum")[0]["store"]["serum_id"] == f"{LAB} F001"
    assert b.chart.antigens[0].passage == "E3"
    assert b.chart.sera[1].serum_id == f"{LAB} F002"
    assert b.unpaired == [] and b.rejected == []
    c = b.counts()
    assert c["antigen"]["converted"] == 2 and c["antigen"]["spelling_changed"] == 1
    assert c["serum"]["spelling_changed"] == 2


def test_without_the_prefix_nothing_pairs(tmp_path):
    """No serum pairs by its key, so no antigen can be confirmed by titres: nothing is paired,
    every point says so (the chain file must name the lab code the store writes)."""
    seed_ags = [ag("TEST-A-1"), ag("TEST-B-2"), ag("TEST-C-3")]
    rows = [["640", "160"], ["320", "1280"], ["80", "40"]]
    seed, refs = build(
        tmp_path, seed_ags, SEED_SR,
        [{"date": "20210115", "rows": rows,
          "ags": [(i, a) for i, a in enumerate(seed_ags)], "srs": AF_SR}],
    )  # fmt: skip
    b = run_bridge(seed, refs, prefix=None)
    assert b.converted == []
    assert {x["reason"] for x in b.unpaired} == {"no key or titre match"}


def test_name_variant_pairs_by_titres(tmp_path):
    seed_ags = [ag("TEST-NEW-TOWN-5"), ag("TEST-A-1")]
    rows = [["640", "160"], ["40", "2560"]]
    seed, refs = build(
        tmp_path, seed_ags, SEED_SR,
        [{"date": "20210115", "rows": rows,
          "ags": [(0, ag("TEST-NEW TOWN-5")), (1, ag("TEST-A-1"))], "srs": AF_SR}],
    )  # fmt: skip
    b = run_bridge(seed, refs)
    conv = by_index(b.converted, "antigen")
    assert conv[0]["how"] == TITRES and b.chart.antigens[0].name == "TEST-NEW TOWN-5"


def test_identical_rows_are_settled_by_the_name(tmp_path):
    seed_ags = [ag("TEST-NEW-TOWN-5"), ag("TEST-OLD-TOWN-6"), ag("TEST-A-1")]
    rows = [["<10", "<10"], ["<10", "<10"], ["640", "320"]]
    seed, refs = build(
        tmp_path, seed_ags, SEED_SR,
        [{"date": "20210115", "rows": rows,
          "ags": [(0, ag("TEST-NEW TOWN-5")), (1, ag("TEST-OLD TOWN-6")),
                  (2, ag("TEST-A-1"))], "srs": AF_SR}],
    )  # fmt: skip
    b = run_bridge(seed, refs)
    conv = by_index(b.converted, "antigen")
    assert conv[0]["how"] == TITRES_NAME  # the second is then the one row left
    assert conv[1]["how"] == TITRES
    assert b.chart.antigens[1].name == "TEST-OLD TOWN-6"


def test_titre_rejected_candidate_keeps_the_seed_identity(tmp_path):
    seed_ags = [ag("TEST-A-1", "E3 (2020-01-01)"), ag("TEST-B-2")]
    rows = [["640", "160"], ["40", "2560"]]
    seed, refs = build(
        tmp_path, seed_ags, SEED_SR,
        [{"date": "20210115", "rows": rows, "seed_rows": {(0, 0): "80", (0, 1): "20"},
          "ags": [(0, ag("TEST-A-1", "E3")), (1, ag("TEST-B-2"))], "srs": AF_SR}],
    )  # fmt: skip
    b = run_bridge(seed, refs)
    un = by_index(b.unpaired, "antigen")
    assert un[0]["reason"] == "titre-rejected" and un[0]["tables"] == [refs[0].table_id]
    assert b.chart.antigens[0].passage == "E3 (2020-01-01)"  # never resolved silently
    assert len(b.rejected) == 1 and b.rejected[0]["seed_candidates"] == [0]
    assert b.counts()["rejected_events"] == 1


def test_a_point_paired_two_ways_is_conflicting(tmp_path):
    seed_ags = [ag("TEST-NEW-TOWN-5"), ag("TEST-A-1"), ag("TEST-B-2")]
    t1 = {
        "date": "20210115",
        "rows": [["640", "160"], ["40", "2560"]],
        "ags": [(0, ag("TEST-NEW TOWN-5")), (1, ag("TEST-A-1"))],
        "srs": AF_SR,
    }
    t2 = {
        "date": "20210215",
        "rows": [["640", "160"], ["20", "80"]],
        "ags": [(0, ag("TEST-NEWTOWN-5")), (2, ag("TEST-B-2"))],
        "srs": AF_SR,
    }
    seed, refs = build(tmp_path, seed_ags, SEED_SR, [t1, t2])
    b = run_bridge(seed, refs)
    un = by_index(b.unpaired, "antigen")
    assert un[0]["reason"] == "conflicting" and len(un[0]["store_candidates"]) == 2
    assert b.chart.antigens[0].name == "TEST-NEW-TOWN-5"


def test_two_points_converting_to_one_identity_are_both_reverted(tmp_path):
    # two harvests in the seed (dated passages), the same undated passage in the store's tables
    seed_ags = [ag("TEST-A-1", "E3 (2020-01-01)"), ag("TEST-A-1", "E3 (2020-06-01)")]
    t1 = {
        "date": "20210115",
        "rows": [["640", "160"]],
        "ags": [(0, ag("TEST-A-1", "E3"))],
        "srs": AF_SR,
    }
    t2 = {
        "date": "20210215",
        "rows": [["80", "2560"]],
        "ags": [(1, ag("TEST-A-1", "E3"))],
        "srs": AF_SR,
    }
    seed, refs = build(tmp_path, seed_ags, SEED_SR, [t1, t2])
    b = run_bridge(seed, refs)
    un = by_index(b.unpaired, "antigen")
    assert un[0]["reason"] == un[1]["reason"] == "conflicting"
    assert "collide" in un[0]["note"]
    assert [a.passage for a in b.chart.antigens] == ["E3 (2020-01-01)", "E3 (2020-06-01)"]


def test_unpaired_seed_points_say_why(tmp_path):
    seed_ags = [ag("TEST-A-1"), ag("TEST-B-2"), ag("TEST-LOST-9")]
    t1 = {
        "date": "20210115",
        "rows": [["640", "160"], ["40", "2560"]],
        "seed_rows": {},
        "ags": [(0, ag("TEST-A-1")), (1, ag("TEST-B-2"))],
        "srs": AF_SR,
    }
    seed, refs = build(tmp_path, seed_ags, SEED_SR, [t1])
    seed.titres.layers[0][(1, 0)] = Titre.parse("640")  # B in the layer, other titres
    seed.titres.layers[0][(1, 1)] = Titre.parse("160")
    b = run_bridge(seed, refs)
    un = by_index(b.unpaired, "antigen")
    assert un[2]["reason"] == "not in any shared layer"
    assert un[1]["reason"] == "titre-rejected"


def test_same_date_tables_find_their_own_layers(tmp_path):
    seed_ags = [ag("TEST-A-1"), ag("TEST-B-2"), ag("TEST-C-3"), ag("TEST-D-4")]
    t1 = {
        "date": "20210115",
        "rows": [["640", "160"], ["40", "2560"]],
        "ags": [(0, ag("TEST-A-1")), (1, ag("TEST-B-2"))],
        "srs": AF_SR,
    }
    t2 = {
        "date": "20210115",
        "suffix": 2,
        "rows": [["320", "20"], ["80", "1280"]],
        "ags": [(2, ag("TEST-C-3")), (3, ag("TEST-D-4"))],
        "srs": AF_SR,
    }
    seed, refs = build(tmp_path, seed_ags, SEED_SR, [t1, t2], layer_order=[1, 0])
    b = run_bridge(seed, refs)
    assert [t["layer"] for t in b.tables] == [1, 0]
    assert [t["swapped"] for t in b.tables] == [True, False]
    assert len(b.converted) == 6


def test_the_layout_and_titres_are_untouched(tmp_path):
    seed_ags = [ag("TEST-A-1", "E3 (2020-01-01)"), ag("TEST-B-2")]
    rows = [["640", "160"], ["40", "2560"]]
    seed, refs = build(
        tmp_path, seed_ags, SEED_SR,
        [{"date": "20210115", "rows": rows,
          "ags": [(0, ag("TEST-A-1", "E3")), (1, ag("TEST-B-2"))], "srs": AF_SR}],
    )  # fmt: skip
    b = run_bridge(seed, refs)
    write_chart(b.chart, tmp_path / "converted.ace")
    back = read_chart(tmp_path / "converted.ace")
    p, q = seed.projections[0], back.projections[0]
    assert p.layout.tobytes() == q.layout.tobytes()
    assert p.transformation is not None and q.transformation is not None
    assert p.transformation.tobytes() == q.transformation.tobytes()
    assert [[str(t) for t in r] for r in back.titres.table] == [
        [str(t) for t in r] for r in seed.titres.table
    ]
    assert back.antigens[0].passage == "E3" and seed.antigens[0].passage == "E3 (2020-01-01)"


def test_declaration_fragment_is_a_declaration(tmp_path):
    seed_ags = [ag("TEST-A-1", "E3 (2020-01-01)"), ag("TEST-B-2")]
    seed, refs = build(
        tmp_path, seed_ags, SEED_SR,
        [{"date": "20210115", "rows": [["640", "160"], ["40", "2560"]],
          "ags": [(0, ag("TEST-A-1", "E3")), (1, ag("TEST-B-2"))], "srs": AF_SR}],
    )  # fmt: skip
    b = run_bridge(seed, refs)
    text = declaration_fragment(
        "labx/h9-hi-test-lab", json.loads(json.dumps(b.to_json())),
        ruling="test ruling", version="0123456789abcdef", file="notes/x.md",
    )  # fmt: skip
    (entry,) = tomllib.loads(text)["declaration"]
    assert all(entry.get(k) for k in DECLARATION_FIELDS)
    assert "2 of 2 converted" in entry["text"]


# ----------------------------------------------------------------------
# config


def test_seed_chart_needs_first_map(tmp_path):
    tables = make_tables(tmp_path / "t", GROUP, n_tables=2)
    refs = tables_from_directory(tables[0].parent, GROUP)
    with pytest.raises(ChainConfigError, match="needs first_map"):
        ChainConfig(GROUP, refs, seed=1, seed_chart=SeedChart(bridge=True))


def test_seed_chart_settings_are_checked():
    with pytest.raises(ChainConfigError, match="af_only_tables"):
        SeedChart(bridge=True, af_only_tables="insert")
    with pytest.raises(ChainConfigError, match="only with bridge"):
        SeedChart(bridge=False, drop_serum_id_prefix=LAB)
    with pytest.raises(ChainConfigError, match="step_choice"):
        MapOptions(step_choice="lowest")


def test_seed_chart_is_read_from_the_chain_file(tmp_path):
    make_tables(tmp_path / "t", GROUP, n_tables=2)
    (tmp_path / "seed.ace").write_text("{}")
    f = tmp_path / "chain.toml"
    f.write_text(
        f'name = "{GROUP}"\nseed = 1\nfirst_map = "seed.ace"\n'
        f'[tables]\ndirectory = "t"\ngroup = "{GROUP}"\n'
        '[options]\nminimum_column_basis = "none"\n'
        f'[seed_chart]\nbridge = true\ndrop_serum_id_prefix = "{LAB}"\naf_only_tables = "append"\n'
        + select_block(tmp_path)
    )
    cfg = load_chain_config(f)
    assert cfg.bridged and cfg.seed_chart == SeedChart(True, LAB, "append")


# ----------------------------------------------------------------------
# the engine


OPTS = MapOptions(scratch_starts=4, incremental_starts=3, grid_test=False)


def respell(chart: Chart) -> Chart:
    """The chain's own map spelled as another program might: no lab code, dated passages."""
    ags = [dataclasses.replace(a, passage=f"{a.passage} (2020-01-01)") for a in chart.antigens]
    srs = [dataclasses.replace(s, serum_id=s.serum_id.split(" ", 1)[-1]) for s in chart.sera]
    return dataclasses.replace(chart, antigens=ags, sera=srs)


@pytest.fixture
def seeded(tmp_path):
    """Five tables; a chain over the first three makes the seed, respelled; the store's tables
    carry a lab code in their serum ids."""
    paths = make_tables(tmp_path / "t", GROUP, n_tables=5)
    for p in paths:
        c = read_chart(p)
        c = dataclasses.replace(
            c, sera=[dataclasses.replace(s, serum_id=f"{LAB} {s.serum_id}") for s in c.sera]
        )
        write_chart(c, p)
    refs = tables_from_directory(tmp_path / "t", GROUP)
    run_chain(ChainConfig(GROUP, refs[:3], OPTS, seed=3), tmp_path / "pre", StubOptimiser())
    made = read_chart(tmp_path / "pre" / GROUP / "steps" / "0002" / CHOSEN)
    write_chart(respell(made), tmp_path / "seed.ace")
    return tmp_path, refs, made


def seeded_config(tmp_path, refs, **kw):
    return ChainConfig(
        GROUP,
        refs,
        dataclasses.replace(OPTS, **kw.pop("options", {})),
        seed=3,
        first_map=tmp_path / "seed.ace",
        seed_chart=SeedChart(bridge=True, drop_serum_id_prefix=LAB, **kw),
    )


def test_engine_steps_only_after_the_seed(seeded):
    tmp_path, refs, made = seeded
    results = run_chain(seeded_config(tmp_path, refs), tmp_path / "s", StubOptimiser())
    assert [r.record["table_id"] for r in results] == ["seed.ace"] + [t.table_id for t in refs[3:]]
    step0 = results[0]
    chosen = read_chart(step0.directory / CHOSEN)
    seed = read_chart(tmp_path / "seed.ace")
    for p, q in zip(seed.projections, chosen.projections, strict=True):
        assert p.layout.tobytes() == q.layout.tobytes()
    # converted back to the store's spelling: the chain's own identities
    assert [a.passage for a in chosen.antigens] == [a.passage for a in made.antigens]
    assert [s.serum_id for s in chosen.sera] == [s.serum_id for s in made.sera]
    b = json.loads((step0.directory / BRIDGE_RECORD).read_text())
    assert step0.record["bridge"]["tables"] == {"in_seed": 3, "af_only": 0, "post_seed": 2}
    assert b["counts"]["serum"]["converted"] == made.n_sera
    # the post-seed steps merged onto the converted seed: known points are common, not new
    assert results[1].record["merge"]["common_sera"] > 0


def test_engine_af_only_tables_leave_out_or_append(seeded):
    tmp_path, refs, _ = seeded
    seed = read_chart(tmp_path / "seed.ace")
    # drop the second table's layer from the seed: the store has it, the seed lacks it
    seed.info["S"] = [s for k, s in enumerate(seed.info["S"]) if k != 1]
    seed.titres.layers = [x for k, x in enumerate(seed.titres.layers) if k != 1]
    write_chart(seed, tmp_path / "seed.ace")
    out = run_chain(seeded_config(tmp_path, refs), tmp_path / "a", StubOptimiser())
    assert [r.record["table_id"] for r in out][1:] == [t.table_id for t in refs[3:]]
    out = run_chain(
        seeded_config(tmp_path, refs, af_only_tables="append"), tmp_path / "b", StubOptimiser()
    )
    ids = [r.record["table_id"] for r in out][1:]
    assert ids == [t.table_id for t in refs[3:]] + [refs[1].table_id]
    assert out[-1].record["seed_af_only"] is True


def test_step_choice_incremental_keeps_the_incremental_map(tmp_path):
    make_tables(tmp_path / "t", GROUP, n_tables=4)
    refs = tables_from_directory(tmp_path / "t", GROUP)
    opts = dataclasses.replace(OPTS, step_choice="incremental")
    results = run_chain(ChainConfig(GROUP, refs, opts, seed=3), tmp_path / "s", StubOptimiser())
    assert all(r.record["chosen"] == "incremental" for r in results[1:])
    assert all(r.record["step_choice"] == "incremental" for r in results[1:])
