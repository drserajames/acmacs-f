"""The ae placer drawn end to end (Geometry.placer = "ae") on the invented tree: it places every
label, reports with ae's measure, keeps hand pins where they are, and changes nothing when off."""

from dataclasses import replace

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")
pytest.importorskip("matplotlib", reason="matplotlib not installed: af.tree.draw needs it")

from af.tree.draw.figure import make_figure  # noqa: E402
from af.tree.draw.render import Geometry  # noqa: E402

from ..synthetic import PARENTS, standard_tree  # noqa: E402
from ..test_figure import _pin_config  # noqa: E402
from ..test_label_editor import _layout, _pin_at  # noqa: E402


def _draw(tmp_path, name, **geometry):
    config = replace(_pin_config([]), label_editor="labels-t.toml")  # the report lists the boxes
    config = replace(config, geometry=replace(config.geometry, **geometry))
    report = make_figure(standard_tree(), PARENTS, config, tmp_path / f"{name}.pdf", {})
    return report, (tmp_path / f"{name}.pdf").read_bytes()


def test_the_ae_placer_places_every_label_and_reports_its_own_measure(tmp_path):
    grid, _ = _draw(tmp_path, "grid")
    ae, _ = _draw(tmp_path, "ae", placer="ae")
    assert {r["text"] for r in ae["aa_label_layout"]} == {
        r["text"] for r in grid["aa_label_layout"]
    }
    rep = ae["aa_label_placement"]["ae"]
    assert rep["metrics"]["n"] == len(ae["aa_label_layout"])
    assert rep["options"] == {
        "tree_ink": "veto",
        "soft_ink_weight": 100000.0,
        "label_clash": "ink",
        "beside_gap": None,
        "overlap_weight": 1,
        "pins_move": False,
        "pin_move_cost": 100.0,
        "seed": 2463534242,
    }
    assert rep["pins_moved"] == []
    assert rep["search"]["conflicts"]["B"] == len(rep["residual_conflicts"]) or rep["warning"]
    assert "ae" not in grid["aa_label_placement"]


def test_the_ae_placer_is_deterministic_and_the_grid_placer_is_untouched(tmp_path):
    _, a = _draw(tmp_path, "a", placer="ae")
    _, b = _draw(tmp_path, "b", placer="ae")
    assert a == b
    _, g1 = _draw(tmp_path, "g1")
    _, g2 = _draw(tmp_path, "g2", placer="grid")
    assert g1 == g2


def test_leaders_end_mid_branch_under_ae(tmp_path):
    ae, _ = _draw(tmp_path, "ae", placer="ae")
    for r in ae["aa_label_layout"]:
        assert r["box"][2] <= r["node"][0]  # left of the branch point, as ae samples them


@pytest.mark.parametrize(
    "switch",
    [
        {"placer_tree_ink": "soft"},
        {"placer_beside_gap": 9.0},
        {"placer_overlap_weight": 8},
        {"placer_pins_move": True, "pin_clearance": 1.0, "pin_clear_dx": 16.0},
    ],
)
def test_the_ruling_switches_run(tmp_path, switch):
    rep, _ = _draw(tmp_path, "s", placer="ae", **switch)
    ae = rep["aa_label_placement"]["ae"]
    opts = ae["options"]
    assert opts["tree_ink"] == switch.get("placer_tree_ink", "veto")
    assert opts["beside_gap"] == switch.get("placer_beside_gap")
    assert opts["overlap_weight"] == switch.get("placer_overlap_weight", 1)
    assert opts["pins_move"] == switch.get("placer_pins_move", False)
    assert all(m["dx"] or m["dy"] for m in ae["pins_moved"])


def test_an_unknown_switch_value_is_refused(tmp_path):
    with pytest.raises(ValueError, match="placer_tree_ink"):
        _draw(tmp_path, "x", placer="ae", placer_tree_ink="sometimes")
    with pytest.raises(ValueError, match="unknown placer"):
        _draw(tmp_path, "y", placer="annealing")


def test_a_hand_pin_stays_where_it_was_put_under_ae(tmp_path):
    _, rows = _layout(tmp_path)
    pin = _pin_at(rows["K2N"], 20.0, 150.0, Geometry().height)
    on = replace(Geometry(), placer="ae")
    _, after = _layout(tmp_path, [pin], geometry=on)
    assert after["K2N"]["box"][:2] == pytest.approx([20.0, 150.0], abs=2e-3)
