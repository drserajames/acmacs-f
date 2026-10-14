"""A finished map measured against a reference map: composition and basin, declared together."""

import dataclasses
import json
import os
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from af.chain.__main__ import measure_references
from af.chain.config import ChainConfigError, MapOptions, Reference
from af.chain.reference import BASIN_METHOD, composition, reference_check, sentences
from af.chain.review import build_review
from af.chart.ace import read_chart, write_chart
from af.chart.model import Antigen, Chart, Serum, Titres, empty_table

from .test_engine import config, run, tables  # noqa: F401  (tables is a fixture)


def _chart(antigens, sera):
    return Chart(
        info={"D": "20210115"},
        antigens=antigens,
        sera=sera,
        titres=Titres(empty_table(len(antigens), len(sera)), []),
    )


def test_composition_stages_and_sides():
    ours = _chart(
        [
            Antigen("A(H3N2)/EXAMPLETOWN/1/2021", passage="MDCK1"),
            Antigen(
                "A(H3N2)/EXAMPLE_TOWN/2/2021", passage="MDCK1"
            ),  # the reference writes EXAMPLE-TOWN
            Antigen("A(H3N2)/EXAMPLETOWN/3/2021", passage="MDCK1 (2021-01-02)"),
            Antigen("A(H3N2)/EXAMPLETOWN/9/2021", passage="MDCK1"),  # only here
        ],
        [Serum("A(H3N2)/EXAMPLETOWN/1/2021", serum_id="LABX F1/21")],
    )
    ref = _chart(
        [
            Antigen("A(H3N2)/EXAMPLETOWN/1/2021", passage="MDCK1"),
            Antigen("A(H3N2)/EXAMPLE-TOWN/2/2021", passage="MDCK1"),
            Antigen("A(H3N2)/EXAMPLETOWN/3/2021", passage="MDCK1"),
            Antigen("A(H3N2)/EXAMPLETOWN/8/2021", passage="MDCK1"),  # only there
        ],
        [Serum("A(H3N2)/EXAMPLETOWN/1/2021", serum_id="LABX F1/21")],
    )
    c = composition(ours, ref)
    c.pop("_pairs")
    assert c["antigens"] == {"map": 4, "reference": 4, "matched": 3}
    assert c["sera"]["matched"] == 1
    assert c["matched_by_stage"]["antigens"]["place punctuation"] == 1
    assert c["matched_by_stage"]["antigens"]["passage date"] == 1
    assert c["map_only"]["antigens"]["count"] == 1
    assert "EXAMPLETOWN/9/2021" in c["map_only"]["antigens"]["listed"][0]
    assert "EXAMPLETOWN/8/2021" in c["reference_only"]["antigens"]["listed"][0]
    assert c["jaccard"] == pytest.approx(4 / 6)


def _merged_map(tables_dir, store):
    cfg = config(tables_dir)
    cfg = dataclasses.replace(cfg, options=MapOptions(merge_all=True, scratch_starts=4))
    [result] = run(cfg, store)
    return cfg, result.directory.parent.parent, result.directory / "chosen.ace"


def test_basin_against_itself_is_not_a_shortfall(tables, tmp_path):  # noqa: F811
    _, _, chosen = _merged_map(tables, tmp_path / "s")
    chart = read_chart(chosen)
    check = reference_check(chart, chart, starts=3, seed=1)
    b = check["basin"]
    placed = int(np.isfinite(chart.best().layout).all(axis=1).sum())
    assert b["seeded_points"] == placed  # every placed point starts at its own place
    assert b["seeded_minus_map"] >= -1e-6 * b["map_stress"]
    assert any("at or below" in x for x in sentences(check, "itself"))
    assert check["shipped"]["rmsd"] < 1e-9 and check["shipped"]["n_groups"] == 0


def test_a_shortfall_is_declared_with_numbers_and_movers():
    check = {
        "composition": {
            "antigens": {"map": 10, "reference": 10, "matched": 9},
            "sera": {"map": 2, "reference": 2, "matched": 2},
            "jaccard": 0.9,
            "map_only": {
                "antigens": {"count": 1, "listed": ["X"]},
                "sera": {"count": 0, "listed": []},
            },
            "reference_only": {
                "antigens": {"count": 1, "listed": ["Y"]},
                "sera": {"count": 0, "listed": []},
            },
            "matched_by_stage": {"antigens": {}, "sera": {}},
        },
        "basin": {
            "starts": 50,
            "seeded_points": 11,
            "points": 12,
            "map_stress": 100.0,
            "seeded_stress": 99.8,
            "seeded_minus_map": -0.2,
            "relative": -0.002,
            "rmsd": 1.15,
            "apart_over_0_5": 7,
            "apart_over_1": 3,
            "movers": [{"point": "TEST-CLUSTER-1", "distance": 6.9}],
        },
    }
    text = " ".join(sentences(check, "the reference"))
    assert "NOT the lowest-stress layout found" in text
    assert "the rest at random, best of 50" in text  # a record from before 12 Oct: random only
    assert "0.200 (0.20%)" in text and "RMSD 1.15" in text and "TEST-CLUSTER-1 6.90" in text
    assert "the search" in text and "not the data" in text
    assert "Jaccard 0.900" in text  # composition and basin together, never one alone


def test_two_references_measured_cached_and_shown(tables, tmp_path, monkeypatch):  # noqa: F811
    cfg, root, chosen = _merged_map(tables, tmp_path / "s")
    refs = tmp_path / "refs"
    (refs / "round").mkdir(parents=True)
    write_chart(read_chart(chosen), refs / "round" / "downloaded.ace")
    write_chart(read_chart(chosen), refs / "round" / "styled.ace")
    both = [
        Reference("unadjusted (the round)", "round/downloaded.ace", starts=2),
        Reference("adjusted (the round, shipped)", "round/styled.ace", starts=2),
    ]
    cfg = dataclasses.replace(cfg, references=both)
    settings = cast(Any, SimpleNamespace(references=refs, threads=1))
    measure_references(cfg, settings, root)
    doc = json.loads((root / "chain.json").read_text())
    assert [r["label"] for r in doc["references"]] == [r.label for r in both]
    for r in doc["references"]:
        assert r["basin"]["placements"]["random"]["starts"] == 2
        assert "shipped" in r and "chart_sha256" in r
        assert r["basin_method"] == r["basin"]["method"] == BASIN_METHOD

    import af.chain.reference as reference

    monkeypatch.setattr(reference, "reference_check", lambda *a, **k: pytest.fail("not cached"))
    measure_references(cfg, settings, root)  # nothing changed: nothing measured again
    page = build_review(root).read_text()
    assert "Against the reference maps" in page
    assert "unadjusted (the round)" in page and "adjusted (the round, shipped)" in page


def test_labels_must_differ(tables, tmp_path):  # noqa: F811
    cfg, _, _ = _merged_map(tables, tmp_path / "s")
    with pytest.raises(ChainConfigError, match="share a label"):
        dataclasses.replace(cfg, references=[Reference("x", "a.ace"), Reference("x", "b.ace")])


def test_a_missing_reference_is_an_error(tables, tmp_path):  # noqa: F811
    cfg, root, _ = _merged_map(tables, tmp_path / "s")
    cfg = dataclasses.replace(cfg, references=[Reference("nowhere", "nowhere.ace")])
    with pytest.raises(ChainConfigError, match="reference map 'nowhere' missing"):
        measure_references(cfg, cast(Any, SimpleNamespace(references=tmp_path, threads=1)), root)
    with pytest.raises(ChainConfigError, match="references"):
        measure_references(cfg, cast(Any, SimpleNamespace(references=None, threads=1)), root)


def test_a_block_carried_elsewhere_is_one_tight_group():
    from af.chain.reference import moved_groups

    chart = _chart(
        [Antigen(f"EXAMPLETOWN/{i}/2021", passage="MDCK1", date="2021-03-01") for i in range(60)],
        [],
    )
    rng = np.random.default_rng(3)
    layout = rng.uniform(0, 20, size=(60, 2))
    layout[:8] = [10.0, 10.0] + rng.normal(scale=0.3, size=(8, 2))  # a tight cluster
    other = layout.copy()
    other[:8] -= [3.0, 0.0]  # the cluster sits 3 units away in the other map
    other[30] -= [0.0, 6.0]  # one point far on its own
    m = moved_groups(chart, np.arange(60), layout, other)
    assert m["apart_over_1"] == 9 and m["n_groups"] == 2 and m["group_sizes"] == [8, 1]
    g = m["groups"][0]
    assert g["mean_shift"] == pytest.approx(3.0) and g["spread"] < 1e-9
    assert g["years"] == {"2021": 8}
    assert [f["group"] for f in m["far"]] == [2] and m["far"][0]["distance"] == pytest.approx(6.0)


def test_chain_file_reference_tables(tmp_path):
    from af.chain.config import ChainSettings
    from af.util.config import load_config

    path = tmp_path / "chain.toml"
    path.write_text(
        'name = "x"\nseed = 1\n[tables]\ndataset = "labx/h9"\n'
        '[options]\nminimum_column_basis = "none"\n'
        '[[reference]]\nlabel = "unadjusted"\nchart = "r/downloaded.ace"\n'
        '[[reference]]\nlabel = "adjusted"\nchart = "r/styled.ace"\nstarts = 7\n'
    )
    s = load_config(path, ChainSettings)
    assert [(r.label, r.chart, r.starts) for r in s.reference] == [
        ("unadjusted", "r/downloaded.ace", 50),
        ("adjusted", "r/styled.ace", 7),
    ]


# ---- placing the points that do not pair (06 on gp-cnic, 08 on neut-vidrl, 11-12 Oct 2026) ----


ROTATE = np.array([[0.0, -1.0], [1.0, 0.0]])  # a quarter turn: the reference's own frame


def _reference_of(chart: Chart, unpaired: list[int], *, moved_zero: bool = False) -> Chart:
    """`chart` turned and shifted into another frame, with the antigens `unpaired` renamed so
    they pair with nothing; `moved_zero` puts a scrambled, higher-stress projection first, as a
    hand-moved #0 would be."""
    ref = dataclasses.replace(chart, antigens=list(chart.antigens))
    for i in unpaired:
        ref.antigens[i] = dataclasses.replace(ref.antigens[i], name=f"ZZ/ELSEWHERE/{i}/2021")
    best = chart.best()
    own = dataclasses.replace(best, layout=best.layout @ ROTATE + [5.0, -3.0])
    ref.projections = [own]
    if moved_zero:
        scrambled = np.random.default_rng(1).permutation(own.layout)
        ref.projections = [dataclasses.replace(own, layout=scrambled, stress=1e9), own]
    return ref


def test_unpaired_points_start_at_the_maps_own_positions(tables, tmp_path):  # noqa: F811
    _, _, chosen = _merged_map(tables, tmp_path / "s")
    chart = read_chart(chosen)
    ref = _reference_of(chart, [0, 1])
    b = reference_check(chart, ref, starts=2, seed=1)["basin"]
    fitted, rand = b["placements"]["map-fitted"], b["placements"]["random"]
    assert fitted["placed_from_map"] == 2 and fitted["starts"] == 1
    assert fitted["seeded_points"] == rand["seeded_points"]  # the same pairs, both ways
    # every point starts where this map has it, so the relax stays in this map's basin
    assert abs(fitted["seeded_minus_map"]) <= 1e-6 * b["map_stress"]
    lower = min(b["placements"], key=lambda n: b["placements"][n]["seeded_stress"])
    assert b["seeded_stress"] == b["placements"][lower]["seeded_stress"]
    assert b["placement"] in b["placements"] and b["method"] == BASIN_METHOD


def test_map_fitted_starts_from_the_references_lowest_stress_projection(tables, tmp_path):  # noqa: F811
    _, _, chosen = _merged_map(tables, tmp_path / "s")
    chart = read_chart(chosen)
    b = reference_check(chart, _reference_of(chart, [0], moved_zero=True), starts=2, seed=1)
    tried = b["basin"]["placements"]
    assert tried["random"]["reference_projection"] == 0  # as recorded before: comparable
    assert tried["map-fitted"]["reference_projection"] == 1
    assert b["basin"]["placement"] == "map-fitted"  # the scrambled #0 is a poor start


def test_map_fitted_fills_only_the_unpaired_rows():
    from af.chain.reference import _map_fitted

    layout = np.random.default_rng(2).uniform(0, 10, size=(8, 2))
    start = layout @ ROTATE + [1.0, 2.0]
    start[[3, 6]] = np.nan
    filled, n = cast(tuple[np.ndarray, int], _map_fitted(start, layout))
    assert n == 2 and np.allclose(filled, layout @ ROTATE + [1.0, 2.0])
    few = np.full_like(start, np.nan)
    few[:2] = start[:2]
    assert _map_fitted(few, layout) is None  # two pairs cannot fix a 2-D fit


def test_the_placement_is_named_in_words():
    b = {
        "method": BASIN_METHOD,
        "placement": "map-fitted",
        "placements": {
            "random": {"starts": 50, "seeded_stress": 101.5},
            "map-fitted": {"starts": 1, "seeded_stress": 99.8},
        },
        "starts": 1,
        "seeded_points": 11,
        "points": 12,
        "map_stress": 100.0,
        "seeded_stress": 99.8,
        "seeded_minus_map": -0.2,
        "relative": -0.002,
        "rmsd": 1.15,
        "apart_over_0_5": 7,
        "apart_over_1": 3,
        "movers": [],
    }
    comp = {
        "antigens": {"map": 10, "reference": 10, "matched": 9},
        "sera": {"map": 2, "reference": 2, "matched": 2},
        "jaccard": 0.9,
        "map_only": {"antigens": {"count": 1, "listed": []}, "sera": {"count": 0, "listed": []}},
        "reference_only": {
            "antigens": {"count": 1, "listed": []},
            "sera": {"count": 0, "listed": []},
        },
        "matched_by_stage": {"antigens": {}, "sera": {}},
    }
    text = " ".join(sentences({"composition": comp, "basin": b}, "the reference"))
    assert "the rest at this map's own positions fitted into the reference's frame" in text
    assert "with them at random 101.500" in text
    b |= {"placement": "random", "starts": 50, "seeded_stress": 101.5}
    text = " ".join(sentences({"composition": comp, "basin": b}, "the reference"))
    assert "the rest at random, best of 50; with them at this map's own positions 99.800" in text


def test_pairs_come_in_our_index_order_in_every_process():
    # 13 Oct 2026: pairs were made by iterating a set of names, so the order of the points a
    # comparison walks, and which of two equal groups was listed, changed between runs
    import subprocess
    import sys

    from af.chain.reference import staged_pairs

    def key(name: str) -> str:
        return name

    ours = [f"EXAMPLETOWN/{i}/2021" for i in range(200)]
    theirs = list(reversed(ours))
    pairs, _ = staged_pairs(ours, theirs, [("name", key)])
    assert list(pairs) == list(range(200)) and pairs[0] == 199
    code = (
        "from af.chain.reference import staged_pairs\n"
        "ours = [f'EXAMPLETOWN/{i}/2021' for i in range(200)]\n"
        "half = [ours[i] for i in range(0, 200, 2)]\n"
        "pairs, _ = staged_pairs(ours, half[::-1], [('name', lambda n: n)])\n"
        "print(list(pairs.items()))\n"
    )
    seen = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(seen) == 1


def test_equal_groups_are_listed_furthest_moved_first():
    from af.chain.reference import moved_groups

    chart = _chart(
        [Antigen(f"EXAMPLETOWN/{i}/2021", passage="MDCK1", date="2021-03-01") for i in range(40)],
        [],
    )
    rng = np.random.default_rng(5)
    layout = rng.uniform(0, 40, size=(40, 2))
    other = layout.copy()
    for point, shift in ((3, 2.0), (17, 7.0), (25, 4.0)):  # three points on their own
        other[point] -= [shift, 0.0]
    m = moved_groups(chart, np.arange(40), layout, other)
    assert m["group_sizes"] == [1, 1, 1]
    assert [round(g["mean_shift"], 6) for g in m["groups"]] == [7.0, 4.0, 2.0]
