"""A finished map measured against a reference map: composition and basin, declared together."""

import dataclasses
import json
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from af.chain.__main__ import measure_reference
from af.chain.config import ChainConfigError, MapOptions, Reference
from af.chain.reference import composition, reference_check, sentences
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
    assert "at or below" in sentences(check, "itself")[1]


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
    assert "0.200 (0.20%)" in text and "RMSD 1.15" in text and "TEST-CLUSTER-1 6.90" in text
    assert "the search" in text and "not the data" in text
    assert "Jaccard 0.900" in text  # composition and basin together, never one alone


def test_measured_into_chain_json_cached_and_shown(tables, tmp_path, monkeypatch):  # noqa: F811
    cfg, root, chosen = _merged_map(tables, tmp_path / "s")
    refs = tmp_path / "refs"
    (refs / "round").mkdir(parents=True)
    write_chart(read_chart(chosen), refs / "round" / "map.ace")
    cfg = dataclasses.replace(cfg, reference=Reference("round/map.ace", starts=2))
    settings = cast(Any, SimpleNamespace(references=refs, threads=1))
    measure_reference(cfg, settings, root)
    doc = json.loads((root / "chain.json").read_text())
    assert doc["reference"]["chart"] == "round/map.ace"
    assert doc["reference"]["basin"]["starts"] == 2

    import af.chain.reference as reference

    monkeypatch.setattr(reference, "reference_check", lambda *a, **k: pytest.fail("not cached"))
    measure_reference(cfg, settings, root)  # nothing changed: nothing measured again
    page = build_review(root).read_text()
    assert "Against the reference map" in page and "Jaccard" in page


def test_a_missing_reference_is_an_error(tables, tmp_path):  # noqa: F811
    cfg, root, _ = _merged_map(tables, tmp_path / "s")
    cfg = dataclasses.replace(cfg, reference=Reference("nowhere.ace"))
    with pytest.raises(ChainConfigError, match="reference map missing"):
        measure_reference(cfg, cast(Any, SimpleNamespace(references=tmp_path, threads=1)), root)
    with pytest.raises(ChainConfigError, match="references"):
        measure_reference(cfg, cast(Any, SimpleNamespace(references=None, threads=1)), root)
