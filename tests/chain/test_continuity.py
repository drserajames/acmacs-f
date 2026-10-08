"""A rebuilt map against the version it replaces: pairing, fit, cut-offs, the record on disk."""

import json
from pathlib import Path

import numpy as np
import pytest

from af.chain.continuity import Cutoff, Thresholds, continuity, load_thresholds, main, sentences
from af.chart.ace import write_chart
from af.chart.model import Antigen, Chart, Projection, Serum, Titres, empty_table
from af.util.config import ConfigError

THRESHOLDS = Thresholds(
    apart=[Cutoff(0.5, "invented half unit"), Cutoff(1.0, "invented one unit")],
    movers=Cutoff(1.0, "invented: name what moved more than one unit"),
)
TOML = """
[[apart]]
distance = 0.5
reason = "invented half unit"
[[apart]]
distance = 1.0
reason = "invented one unit"
[movers]
distance = 1.0
reason = "invented: name what moved more than one unit"
"""


def _chart(names: list[str], sera: list[str], layout, stress: float) -> Chart:
    antigens = [Antigen(f"A(H3N2)/EXAMPLETOWN/{n}/2021", passage="MDCK1") for n in names]
    sr = [Serum(f"A(H3N2)/EXAMPLETOWN/{n}/2021", serum_id=f"LABX F{n}/21") for n in sera]
    return Chart(
        info={"D": "20210115"},
        antigens=antigens,
        sera=sr,
        titres=Titres(empty_table(len(antigens), len(sr)), []),
        projections=[Projection(np.asarray(layout, dtype=float), stress=stress)],
    )


def _rotated(layout: np.ndarray, degrees: float, shift: tuple[float, float]) -> np.ndarray:
    t = np.radians(degrees)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return layout @ rot + np.asarray(shift)


BASE = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [5.0, 5.0], [-2.0, 1.0], [1.0, -3.0]])


def test_a_rotated_translated_copy_has_not_moved():
    new = _chart(["1", "2", "3", "4", "5"], ["1"], BASE, 100.0)
    prev = _chart(["1", "2", "3", "4", "5"], ["1"], _rotated(BASE, 37, (10, -4)), 90.0)
    c = continuity(new, prev, THRESHOLDS)
    assert c["procrustes"]["rmsd"] == pytest.approx(0, abs=1e-9)
    assert c["procrustes"]["compared"] == 6
    assert [a["count"] for a in c["procrustes"]["apart"]] == [0, 0]
    assert c["movers"] == []
    assert c["stress"] == pytest.approx(
        {"new": 100.0, "previous": 90.0, "difference": 10.0, "relative": 10 / 90}
    )


def test_points_pair_by_designation_not_index():
    # previous: same points in another order, one point (7) the rebuild dropped, and point 3
    # moved by 2 units; new adds point 6 and repeats point 5 (ambiguous: paired with nothing)
    order = [4, 0, 2, 1, 3]  # antigens 5, 1, 3, 2, 4 of BASE
    prev_layout = np.vstack([BASE[order], [[9.0, 9.0]], BASE[5:]])
    prev_layout[2] += [2.0, 0.0]  # antigen 3
    prev = _chart(["5", "1", "3", "2", "4", "7"], ["1"], prev_layout, 50.0)
    new_layout = np.vstack([BASE[:5], [[7.0, 7.0], [-2.0, 1.0]], BASE[5:]])
    new = _chart(["1", "2", "3", "4", "5", "6", "5"], ["1"], new_layout, 50.0)
    c = continuity(new, prev, THRESHOLDS)
    p = c["points"]
    assert (p["new"], p["previous"], p["paired"]) == (8, 7, 5)
    assert p["added"]["listed"]["antigens"] == ["A(H3N2)/EXAMPLETOWN/6/2021 MDCK1"]
    assert p["removed"]["listed"]["antigens"] == ["A(H3N2)/EXAMPLETOWN/7/2021 MDCK1"]
    assert p["ambiguous"]["count"] == 1
    # the fit is over 5 points with one of them off by 2: only antigen 3 moves past 1
    assert [m["point"] for m in c["movers"]] == ["A(H3N2)/EXAMPLETOWN/3/2021 MDCK1"]
    assert c["movers"][0]["distance"] > 1.0
    assert c["procrustes"]["apart"][1]["count"] == 1


def test_disconnected_points_are_counted_not_fitted():
    layout = BASE.copy()
    layout[1] = np.nan
    new = _chart(["1", "2", "3", "4", "5"], ["1"], layout, 1.0)
    prev = _chart(["1", "2", "3", "4", "5"], ["1"], BASE, 1.0)
    c = continuity(new, prev, THRESHOLDS)
    assert (c["procrustes"]["compared"], c["procrustes"]["disconnected"]) == (5, 1)


def test_thresholds_are_data_with_reasons(tmp_path):
    path = tmp_path / "t.toml"
    path.write_text(TOML)
    assert load_thresholds(path) == THRESHOLDS
    path.write_text(
        TOML.replace('"invented one unit"', '" "').replace("distance = 0.5", "distance = 0")
    )
    with pytest.raises(ConfigError) as e:
        load_thresholds(path)
    assert any("apart[0]: distance" in p for p in e.value.problems)
    assert any("apart[1]: reason is empty" in p for p in e.value.problems)


def _version(root: Path, version: str, parent: str | None, chart: Chart, params: dict) -> None:
    v = root / "versions" / version
    (v / "steps" / "0000").mkdir(parents=True)
    write_chart(chart, v / "steps" / "0000" / "chosen.ace")
    (v / "chain.json").write_text(
        json.dumps({"steps": [{"directory": "steps/0000", "chosen_file": "chosen.ace"}]})
    )
    inputs = [{"store": {"dataset": "labx/h3", "kind": "tables", "version": "t1"}}]
    (v / "PROVENANCE.json").write_text(json.dumps({"inputs": inputs, "parameters": params}))
    with (root / "HISTORY.jsonl").open("a") as f:
        f.write(json.dumps({"event": "published", "version": version, "parent": parent}) + "\n")


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "chains" / "labx" / "h3" / "merged"
    prev = _chart(["1", "2", "3", "4", "5"], ["1"], _rotated(BASE, 90, (1, 1)), 20.0)
    new = _chart(["1", "2", "3", "4", "5"], ["1"], BASE, 19.0)
    _version(root, "v1", None, prev, {"seed": 1})
    _version(root, "v2", "v1", new, {"seed": 1})
    _version(root, "v3", "v2", new, {"seed": 2})
    (tmp_path / "t.toml").write_text(TOML)
    return tmp_path


def _run(store: Path, version: str) -> dict:
    args = [store / "chains", "labx/h3/merged", version, store / "t.toml", store / "records"]
    assert main([str(a) for a in args]) == 0
    return json.loads((store / "records/labx/h3/merged" / f"{version}.continuity.json").read_text())


def test_record_names_its_parent_and_whether_stresses_compare(store):
    drift = store / "records/labx/h3/merged/v1.tables-drift.json"
    drift.parent.mkdir(parents=True)
    drift.write_text("{}")
    rec = _run(store, "v2")
    assert rec["previous"]["version"] == "v1"
    assert rec["same_problem"] is True and rec["parameters_differ"] == []
    assert rec["retires"] == "labx/h3/merged/v1.tables-drift.json"
    assert rec["thresholds"]["movers"]["reason"].startswith("invented")
    assert len(rec["map_sha256"]) == 64 and rec["measured_by"]["script"] == "af.chain.continuity"
    text = sentences(rec)
    assert "stress 20.000 -> 19.000 (-5.00%)" in text[1]
    assert "retires" in text[-1]

    rec3 = _run(store, "v3")
    assert rec3["same_problem"] is False and rec3["parameters_differ"] == ["seed"]
    assert rec3["retires"] is None
    assert "NOT comparable: parameters seed differ" in sentences(rec3)[1]


def test_no_parent_and_no_overwrite(store):
    with pytest.raises(ValueError, match="no previous version"):
        _run(store, "v1")
    _run(store, "v2")
    with pytest.raises(FileExistsError):
        _run(store, "v2")
