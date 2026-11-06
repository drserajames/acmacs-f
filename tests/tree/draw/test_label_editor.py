"""The tree label editor and the hand labels file it writes (af.tree.draw.editor, .hand)."""

import json
import re
from dataclasses import replace

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")
pytest.importorskip("matplotlib", reason="matplotlib not installed: af.tree.draw needs it")

from af.tree.draw.aa_labels import LabelHide, LabelRuleError  # noqa: E402
from af.tree.draw.figure import LabelPin, LabelPinError, make_figure  # noqa: E402
from af.tree.draw.hand import HandLabelsError, load_hand_labels  # noqa: E402
from af.tree.draw.render import Geometry, HandLabelClash  # noqa: E402
from af.tree.draw.stage import outputs  # noqa: E402

from .synthetic import PARENTS, standard_tree  # noqa: E402
from .test_figure import _pin_config  # noqa: E402

K2N = ("EPI_ISL_0000000", "EPI_ISL_0000019")  # the invented tree's three labels' leaves
# aa_label_layout rounds node points and boxes to 3 decimals: a box drawn from a pin computed
# from those rows can differ by two roundings (and a float hair), never more
ROUNDED = 2e-3
L7F = ("EPI_ISL_0000010", "EPI_ISL_0000019")


def _layout(tmp_path, pins=(), hides=(), **kw):
    config = replace(_pin_config(list(pins), **kw), label_editor="labels-t.toml")
    if hides:
        config = replace(config, hide_labels=list(hides))
    report = make_figure(standard_tree(), PARENTS, config, tmp_path / "f.pdf", {})
    return report, {row["text"]: row for row in report["aa_label_layout"]}


def _editor_data(html: str) -> dict:
    found = re.search(r"<script type='application/json' id='data'>(.*?)</script>", html, re.S)
    assert found
    return json.loads(found.group(1).replace("<\\/", "</"))


def _pin_at(row: dict, box_x0: float, box_y0: float, height: float, **kw) -> LabelPin:
    """A hand pin putting a label's box top-left at (box_x0, box_y0) (render._pinned_box)."""
    first, last = row["first"], row["last"]
    dx = (box_x0 + 1 - row["node"][0]) / height
    dy = (box_y0 - row["node"][1]) / height
    return LabelPin(first, last, row["text"], dx, dy, "invented (AB, 2026-10-30)", hand=True, **kw)


def test_the_editor_is_written_from_the_same_drawing_and_the_pdf_keeps_its_bytes(tmp_path):
    report, rows = _layout(tmp_path)
    plain = make_figure(standard_tree(), PARENTS, _pin_config([]), tmp_path / "p.pdf", {})
    assert (tmp_path / "f.pdf").read_bytes() == (tmp_path / "p.pdf").read_bytes()
    assert "aa_label_layout" not in plain and sorted(rows) == ["A6S", "K2N", "L7F"]
    data = _editor_data((tmp_path / "f.labels.html").read_text())
    assert data["save_as"] == "labels-t.toml"
    assert [lab["box"] for lab in data["labels"]] == [r["box"] for r in report["aa_label_layout"]]
    ink = data["adapter"]["ink"]
    assert len(__import__("base64").b64decode(ink["bits"])) == (ink["rows"] * ink["cols"] + 7) // 8
    assert data["adapter"]["font_size"] == Geometry().label_font
    assert data["page"] == {"width": Geometry().width, "height": Geometry().height}


def _hand_file(tmp_path, rows, leaves=None):
    """A hand labels file as the editor saves it: K2N moved 10 left and 5 down, L7F hidden;
    ``leaves`` the leaf counts written, the nodes' own unless given (None: none written)."""
    k2n = rows["K2N"]
    height = Geometry().height
    dx = (k2n["box"][0] - 10 + 1 - k2n["node"][0]) / height
    dy = (k2n["box"][1] + 5 - k2n["node"][1]) / height
    n = {"K2N": rows["K2N"]["leaves"], "L7F": rows["L7F"]["leaves"]} if leaves is None else leaves
    count = {t: f"leaves = {c}\n" if c else "" for t, c in n.items()}
    path = tmp_path / "labels-t.toml"
    path.write_text(
        'figure = "TEST tree"\n\n[[label_pins]]\n'
        f'first = "{K2N[0]}"\nlast = "{K2N[1]}"\n{count["K2N"]}text = "K2N"\ndx = {dx}\ndy = {dy}\n'
        'reason = "invented: clear of the strain"\nby = "AB"\ndate = "2026-10-30"\n\n'
        f'[[hide_labels]]\nfirst = "{L7F[0]}"\nlast = "{L7F[1]}"\n{count["L7F"]}text = "L7F"\n'
        'reason = "invented: not wanted"\nby = "AB"\ndate = "2026-10-30"\n'
    )
    return path


def test_a_saved_file_draws_each_label_exactly_where_it_was_left(tmp_path):
    _, rows = _layout(tmp_path)
    k2n = rows["K2N"]
    pins, hides = load_hand_labels(_hand_file(tmp_path, rows))
    assert pins[0].hand and hides[0].hand
    assert pins[0].reason == "invented: clear of the strain (AB, 2026-10-30)"
    report, after = _layout(tmp_path, pins, hides)
    assert after["K2N"]["box"] == pytest.approx(
        [k2n["box"][0] - 10, k2n["box"][1] + 5, k2n["box"][2] - 10, k2n["box"][3] + 5], abs=ROUNDED
    )
    assert "L7F" not in after
    assert report["label_pins"]["pinned by hand"] == 1
    assert report["aa_label_hides"][0]["hand"] is True
    assert "hand rows without a leaf count" not in report["label_pins"]
    data = _editor_data((tmp_path / "f.labels.html").read_text())
    hidden = [lab for lab in data["labels"] if lab["hidden"]]
    assert [lab["text"] for lab in hidden] == ["L7F"]  # the editor can bring it back
    assert hidden[0]["hide"] == {"reason": "invented: not wanted", "by": "AB", "date": "2026-10-30"}
    pin = next(lab["pin"] for lab in data["labels"] if lab["text"] == "K2N")
    assert pin == {
        "hand": True,
        "reason": "invented: clear of the strain",
        "by": "AB",
        "date": "2026-10-30",
        "keys": {"first": K2N[0], "last": K2N[1], "leaves": k2n["leaves"]},
    }
    assert hidden[0]["keys"]["leaves"] == rows["L7F"]["leaves"]


@pytest.mark.filterwarnings("ignore:.*without the node's leaf count")
def test_a_row_saved_on_a_node_of_another_size_is_refused(tmp_path):
    # the tree changed shape under the row: its MRCA could be another label of the same text
    _, rows = _layout(tmp_path)
    wrong = rows["K2N"]["leaves"] + 1
    pins, hides = load_hand_labels(_hand_file(tmp_path, rows, {"K2N": wrong, "L7F": None}))
    with pytest.raises(
        LabelPinError, match=f"has {rows['K2N']['leaves']} leaves, the row was saved on {wrong}"
    ):
        _layout(tmp_path, pins)
    pins, hides = load_hand_labels(_hand_file(tmp_path, rows, {"K2N": None, "L7F": 3}))
    with pytest.raises(LabelRuleError, match="L7F: the node of .* the row was saved on 3"):
        _layout(tmp_path, hides=hides)


def test_rows_saved_before_they_carried_a_leaf_count_are_read_and_listed(tmp_path):
    # Sarah's interim rows are saved by the editor before this one (00, 6 Nov 2026)
    _, rows = _layout(tmp_path)
    with pytest.warns(UserWarning, match="2 row\\(s\\) without the node's leaf count"):
        pins, hides = load_hand_labels(_hand_file(tmp_path, rows, {"K2N": None, "L7F": None}))
    report, after = _layout(tmp_path, pins, hides)
    assert "L7F" not in after and report["label_pins"]["pinned by hand"] == 1
    assert report["label_pins"]["hand rows without a leaf count"] == ["K2N", "L7F"]
    assert "leaves" not in report["aa_label_hides"][0]


@pytest.mark.parametrize(
    "row, problem",
    [
        ('reason = ""\nby = "AB"\ndate = "2026-10-30"', "no reason"),
        ('reason = "invented"\nby = " "\ndate = "2026-10-30"', "no by"),
        ('reason = "invented"\nby = "AB"\ndate = "30 Oct"', "is not YYYY-MM-DD"),
    ],
)
def test_a_row_without_a_why_a_name_or_a_date_refuses_the_file(tmp_path, row, problem):
    path = tmp_path / "h.toml"
    path.write_text(
        f'[[hide_labels]]\nfirst = "{L7F[0]}"\nlast = "{L7F[1]}"\ntext = "L7F"\n{row}\n'
    )
    with pytest.raises(HandLabelsError, match=problem):
        load_hand_labels(path)


def test_the_editors_own_reasons_and_name_satisfy_the_loader(tmp_path):
    # Save never asks for a why or a name: rows without one carry the editor's own reason and,
    # when no initials were given, "not given". Those must load (rule 9: named, counted, dated).
    from importlib import resources

    js = resources.files("af.util").joinpath("label_editor.js").read_text()
    auto = dict(re.findall(r'^\s+(moved|hidden|frozen): "([^"]+)",$', js, re.M))
    assert set(auto) == {"moved", "hidden", "frozen"}
    assert 'ui.by.trim() || "not given"' in js
    assert "Not saved" not in js  # nothing typed is required
    path = tmp_path / "h.toml"
    path.write_text(
        f'[[label_pins]]\nfirst = "{K2N[0]}"\nlast = "{K2N[1]}"\ntext = "K2N"\ndx = -0.05\n'
        f'dy = 0.0\nreason = "{auto["moved"]}"\nby = "not given"\ndate = "2026-11-07"\n\n'
        '[[label_pins]]\nfirst = "EPI_ISL_0000001"\nlast = "EPI_ISL_0000002"\n'
        'text = "A6S"\ndx = -0.05\n'
        f'dy = 0.0\nreason = "{auto["frozen"]}"\nby = "AB"\ndate = "2026-11-07"\n\n'
        f'[[hide_labels]]\nfirst = "{L7F[0]}"\nlast = "{L7F[1]}"\ntext = "L7F"\n'
        f'reason = "{auto["hidden"]}"\nby = "AB"\ndate = "2026-11-07"\n'
    )
    pins, hides = load_hand_labels(path)
    assert [p.reason for p in pins] == [
        f"{auto['moved']} (not given, 2026-11-07)",
        f"{auto['frozen']} (AB, 2026-11-07)",
    ]
    assert hides[0].reason == f"{auto['hidden']} (AB, 2026-11-07)"


def test_hand_rows_supersede_config_rows_on_the_same_label(tmp_path):
    _, rows = _layout(tmp_path)
    height = Geometry().height
    config_pin = LabelPin(*K2N, "K2N", -0.05, 0.0, "invented: from the round's config")
    hand_pin = _pin_at(rows["K2N"], 20.0, 150.0, height)
    report, after = _layout(tmp_path, [config_pin, hand_pin])
    assert after["K2N"]["box"][:2] == pytest.approx([20.0, 150.0], abs=ROUNDED)
    assert report["label_pins"]["config pins superseded"] == ["K2N: pinned by hand"]
    hide = LabelHide(*K2N, "K2N", "invented (AB, 2026-10-30)", hand=True)
    report, after = _layout(tmp_path, [config_pin], [hide])
    assert "K2N" not in after
    assert report["label_pins"]["config pins superseded"] == ["K2N: hidden by hand"]


def test_a_label_pinned_and_hidden_by_hand_is_an_error(tmp_path):
    _, rows = _layout(tmp_path)
    pin = _pin_at(rows["K2N"], 20.0, 150.0, Geometry().height)
    hide = LabelHide(*K2N, "K2N", "invented (AB, 2026-10-30)", hand=True)
    with pytest.raises(LabelPinError, match="pinned and hidden, both by hand"):
        _layout(tmp_path, [pin], [hide])


def test_a_hand_label_is_never_moved_a_clash_stops_the_build(tmp_path):
    _, rows = _layout(tmp_path)
    height = Geometry().height
    geometry = replace(Geometry(), pin_clearance=1.0)
    # a config pin on a hand pin's spot: clearing moves the config one, never the hand one
    hand = _pin_at(rows["K2N"], 20.0, 150.0, height)
    config_pin = replace(_pin_at(rows["L7F"], 20.0, 150.0, height), hand=False, reason="invented")
    report, after = _layout(tmp_path, [hand, config_pin], geometry=geometry)
    assert after["K2N"]["box"][:2] == pytest.approx([20.0, 150.0], abs=ROUNDED)
    assert [m["text"] for m in report["label_pins"]["moved to clear"]] == ["L7F"]
    # two hand pins on one spot: refused, with or without clearing, and named
    both = [hand, replace(config_pin, hand=True)]
    for kw in ({"geometry": geometry}, {}):
        with pytest.raises(HandLabelClash, match="K2N: against L7F"):
            _layout(tmp_path, both, **kw)


def test_the_stage_caches_the_editor_with_the_figure(tmp_path):
    pdf = tmp_path / "figure.pdf"
    assert "figure.labels.html" not in outputs("figure", pdf)
    assert outputs("figure", pdf, True)["figure.labels.html"] == tmp_path / "figure.labels.html"
