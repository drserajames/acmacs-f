"""The ae placer drawn end to end (Geometry.placer = "ae") on the invented tree: it places every
label, reports with ae's measure, keeps hand pins where they are, and changes nothing when off."""

from dataclasses import replace

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")
pytest.importorskip("matplotlib", reason="matplotlib not installed: af.tree.draw needs it")

from af.tree.draw.aa_labels import LabelHide  # noqa: E402
from af.tree.draw.figure import make_figure  # noqa: E402
from af.tree.draw.placer_ae.geometry import ATTACH_INSET, attach_pt  # noqa: E402
from af.tree.draw.placer_ae.measure import LEADER_GAP, NO_LEADER, drawn_leader  # noqa: E402
from af.tree.draw.placer_ae.scene import ROW, Label  # noqa: E402
from af.tree.draw.render import Geometry  # noqa: E402

from ..synthetic import PARENTS, standard_tree  # noqa: E402
from ..test_figure import _pin_config  # noqa: E402
from ..test_label_editor import K2N, _editor_data, _layout, _pin_at  # noqa: E402


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


def _ae_leader_from_editor(lab: dict, fs: float):
    """What the editor page draws under ae (label_editor.js aeLeader), from the page's data
    alone, through placer_ae's own attach_pt and drawn_leader."""
    x1, cy = lab["box"][2] - 1, (lab["box"][1] + lab["box"][3]) / 2
    th = len(lab["lines"]) * ROW * fs
    box = (x1 - lab["advance_w"], cy - th / 2, x1, cy + th / 2)
    end = tuple(lab["tether"])
    label = Label(0, lab["text"], tuple(lab["lines"]), (lab["advance_w"],), fs, end, end)
    return drawn_leader(label, attach_pt(end, fs, box))


def test_the_editor_draws_aes_leaders_under_ae_and_nothing_changes_under_grid(tmp_path):
    ae, _ = _draw(tmp_path, "ae", placer="ae")
    data = _editor_data((tmp_path / "ae.labels.html").read_text())
    page = data["adapter"]
    assert page["ae_leader"] == {
        "row": ROW,
        "inset": ATTACH_INSET,
        "gap": LEADER_GAP,
        "none_within": NO_LEADER,
    }
    rows = {r["text"]: r for r in ae["aa_label_layout"]}
    drawn = 0
    for lab in data["labels"]:
        row = rows[lab["text"]]
        assert lab["tether"] == row["tether"] and lab["advance_w"] == row["advance_w"]
        assert row["tether"][1] == row["node"][1] and row["tether"][0] < row["node"][0]
        seg = _ae_leader_from_editor(lab, page["font_size"])
        drawn += seg is not None
        if seg is not None:
            assert seg[0] == pytest.approx(tuple(row["tether"]))
    assert drawn  # the invented tree's labels all have a leader to draw
    _, grid = _draw(tmp_path, "grid")
    data = _editor_data((tmp_path / "grid.labels.html").read_text())
    assert "ae_leader" not in data["adapter"]
    assert all("advance_w" not in lab for lab in data["labels"])


def test_a_hidden_label_brought_back_under_ae_has_aes_leader(tmp_path):
    hide = LabelHide(*K2N, "K2N", "invented (AB, 2026-10-30)", hand=True)
    on = replace(Geometry(), placer="ae")
    _layout(tmp_path, hides=[hide], geometry=on)
    data = _editor_data((tmp_path / "f.labels.html").read_text())
    (k2n,) = [lab for lab in data["labels"] if lab["text"] == "K2N"]
    assert k2n["hidden"] and k2n["advance_w"] > 0
    assert k2n["tether"][1] == k2n["anchor"][1] and k2n["tether"][0] < k2n["anchor"][0]
