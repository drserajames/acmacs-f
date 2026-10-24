"""End to end: a synthetic tree to PDF + I7 + draw report."""

import hashlib
import json
from dataclasses import replace
from typing import cast

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")
pytest.importorskip(
    "matplotlib", reason="matplotlib not installed: the tree renderer needs it (pyproject, WS1)"
)

import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from af.tree.draw.aa_labels import LabelHide  # noqa: E402
from af.tree.draw.defaults import load_defaults  # noqa: E402
from af.tree.draw.figure import DashBarError, FigureConfig, Overrides, make_figure  # noqa: E402
from af.tree.draw.render import DashBar  # noqa: E402
from af.tree.draw.sections import SectionOverrideError  # noqa: E402

from .synthetic import PARENTS, standard_tree  # noqa: E402

I7_KEYS = {"i7_version", "kind", "title", "placeholder", "figure", "provenance", "tree", "notes"}


def config(**kw):
    return FigureConfig(
        title="TEST tree",
        window_start="2024-10",
        window_end="2026-10",
        select=replace(load_defaults().clades, min_share=0.1, min_window_leaves=10**9),
        dash_bars=[DashBar(2, {"K": "transparent", "N": "#e72f27"}, [("2N", "#e72f27")])],
        strains=[("EPI_ISL_0000012", "leaf-0012")],
        **kw,
    )


def test_figure_writes_pdf_i7_and_report(tmp_path):
    pdf = tmp_path / "figure.pdf"
    report = make_figure(standard_tree(), PARENTS, config(), pdf, {"tree": "sha256:0"})
    assert pdf.stat().st_size > 1000
    i7 = json.loads((tmp_path / "figure.i7.json").read_text())
    assert set(i7) == I7_KEYS and i7["kind"] == "tree"
    assert i7["figure"]["sha256"] == hashlib.sha256(pdf.read_bytes()).hexdigest()
    tree = i7["tree"]
    assert tree["time_series"] == {"first": "2024-10", "last": "2026-09"}
    assert len(tree["leaves"]) == 50 and [x["order"] for x in tree["leaves"]][:3] == [0, 1, 2]
    # sections = the lettered bands, named by strain name (as the reference extractor writes)
    sections = [(x["prefix"], x["clade"], x["first_leaf"], x["n_leaves"]) for x in tree["sections"]]
    assert sections == [
        ("A", "X.1", "leaf-000", 10),
        ("B", "X.1.1", "leaf-010", 10),
        ("C", "X", "leaf-020", 25),
    ]
    orders = [(x["first_order"], x["last_order"]) for x in tree["sections"]]
    assert orders == [(0, 9), (10, 19), (20, 44)]
    by_order = {leaf["order"]: leaf["name"] for leaf in tree["leaves"] if leaf["shown"]}
    assert all(by_order[x["first_order"]] == x["first_leaf"] for x in tree["sections"])
    names = {leaf["name"] for leaf in tree["leaves"]}
    assert all(x["first_leaf"] in names and x["last_leaf"] in names for x in tree["sections"])
    assert report["aa_label_placement"]["box_overlaps"] == 0
    assert (tmp_path / "figure.draw.json").is_file()


def test_same_input_same_pdf_bytes(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    make_figure(standard_tree(), PARENTS, config(), a, {})
    make_figure(standard_tree(), PARENTS, config(), b, {})
    assert a.read_bytes() == b.read_bytes()


def test_named_label_hides_are_counted_in_the_draw_report(tmp_path):
    labels = replace(load_defaults().labels, target_labels=0, min_share=0.05)
    hide = LabelHide("EPI_ISL_0000010", "EPI_ISL_0000019", "L7F", "invented: not wanted")
    report = make_figure(
        standard_tree(),
        PARENTS,
        config(labels=labels, hide_labels=[hide]),
        tmp_path / "figure.pdf",
        {"tree": "sha256:0"},
    )
    assert report["aa_label_filter"]["hidden by name"] == 1
    assert {" ".join(lab["subs"]) for lab in report["aa_labels"]} == {"K2N", "A6S"}
    assert report["aa_label_filter"]["labels"] == 2
    assert report["aa_label_hides"] == [
        {
            "first": "EPI_ISL_0000010",
            "last": "EPI_ISL_0000019",
            "text": "L7F",
            "reason": "invented: not wanted",
        }
    ]


def test_centre_marks_and_overrides_are_counted(tmp_path):
    cfg = config(
        marked_ids=frozenset({"EPI_ISL_0000001", "EPI_ISL_0000030"}),
        marked_name="LAB",
        overrides=Overrides(hide_clades=frozenset({"X.2"})),
    )
    report = make_figure(standard_tree(), PARENTS, cfg, tmp_path / "m.pdf", {})
    assert report["marked_rows"] == 2
    assert "X.2" not in report["clades_shown"] and report["overrides"]["hide_clades"] == 1
    i7 = json.loads((tmp_path / "m.i7.json").read_text())
    assert sum(x["marked"] for x in i7["tree"]["leaves"]) == 2


def test_override_naming_nothing_fails(tmp_path):
    cfg = config(overrides=Overrides(hide_hz_starting_at=frozenset({"no-such-leaf"})))
    with pytest.raises(SectionOverrideError):
        make_figure(standard_tree(), PARENTS, cfg, tmp_path / "x.pdf", {})


def test_fixed_scale_for_side_by_side_figures(tmp_path):
    from af.tree.draw.render import Geometry

    ok = config(geometry=Geometry(x_max=1.0, row_capacity=80))
    report = make_figure(standard_tree(), PARENTS, ok, tmp_path / "s.pdf", {})
    assert report["rows"] == 50
    too_small = config(geometry=Geometry(row_capacity=10))
    with pytest.raises(ValueError, match="row_capacity"):
        make_figure(standard_tree(), PARENTS, too_small, tmp_path / "t.pdf", {})


def test_unknown_continents_are_counted(tmp_path):
    t = standard_tree()
    t.continent[t.leaves()[0]] = "ATLANTIS"
    report = make_figure(t, PARENTS, config(), tmp_path / "c.pdf", {})
    assert report["continents_not_in_legend"] == {"ATLANTIS": 1}


def test_flags_are_counted_not_acted_on(tmp_path):
    flags = {
        "EPI_ISL_0000001": ["clock_outlier"],
        "EPI_ISL_0000002": ["clock_outlier", "long_branch"],
    }
    report = make_figure(standard_tree(), PARENTS, config(), tmp_path / "f.pdf", {}, flags)
    assert report["flagged_drawn"] == {"clock_outlier": 2, "long_branch": 1}
    assert report["rows"] == 50  # nothing excluded


def test_hide_by_flag_reason_is_opt_in_and_counted(tmp_path):
    from af.tree.draw.layout import HideRuleError, HideRules

    flags = {"EPI_ISL_0000001": ["clock_outlier"], "EPI_ISL_0000002": ["long_branch"]}
    cfg = config(hide=HideRules(flag_reasons=frozenset({"clock_outlier"})))
    report = make_figure(standard_tree(), PARENTS, cfg, tmp_path / "h.pdf", {}, flags)
    assert report["rows"] == 49 and report["hidden"]["flag: clock_outlier"] == 1
    assert report["flagged_drawn"] == {"long_branch": 1}
    wrong = config(hide=HideRules(flag_reasons=frozenset({"no_such_flag"})))
    with pytest.raises(HideRuleError):
        make_figure(standard_tree(), PARENTS, wrong, tmp_path / "w.pdf", {}, flags)


def test_clade_names_in_one_column_do_not_overlap():
    import matplotlib.pyplot as plt

    from af.tree.draw.render import _place_clade_names

    fig, ax = plt.subplots()
    _place_clade_names(
        ax, [(10.0, 100.0, "X.1.1"), (10.0, 101.0, "X.1.2"), (20.0, 100.0, "X")], 7.0
    )
    texts = sorted((t.get_position() for t in ax.texts), key=lambda p: (p[0], p[1]))
    (xa, ya), (xb, yb) = texts[0], texts[1]
    assert xa == xb and yb - ya >= 10  # the second name slid past the first
    assert texts[2][1] == 100.0  # another column is unaffected
    plt.close(fig)


def test_stacked_clade_names_keep_the_gap_clear():
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    from af.tree.draw.render import _place_clade_names, _text_width

    fig, ax = plt.subplots()
    for gap in (6.0, 10.0):
        ax.clear()
        _place_clade_names(ax, [(10.0, 100.0, "X.1.1"), (10.0, 101.0, "X.1.2")], 7.0, gap=gap)
        (_, ya), (_, yb) = sorted(t.get_position() for t in ax.texts)
        width = _text_width(FontProperties(size=7.0))
        assert (yb - width("X.1.2") / 2) - (ya + width("X.1.1") / 2) == pytest.approx(gap)
    plt.close(fig)


def test_store_refs_are_recorded_in_the_i7(tmp_path):
    from af.store import StoreRef

    ref = StoreRef("trees", "t1/report", "0123456789abcdef", "0123456789abcdef" + "0" * 48)
    make_figure(standard_tree(), PARENTS, config(), tmp_path / "r.pdf", {}, store_refs=[ref])
    i7 = json.loads((tmp_path / "r.i7.json").read_text())
    assert i7["provenance"]["store_refs"] == [ref.to_json()]
    make_figure(standard_tree(), PARENTS, config(), tmp_path / "n.pdf", {})
    assert "store_refs" not in json.loads((tmp_path / "n.i7.json").read_text())["provenance"]


def test_leaves_carry_depth_before_page_scaling(tmp_path):
    t = standard_tree()
    make_figure(t, PARENTS, config(), tmp_path / "d.pdf", {}, depth_unit="mutations")
    i7 = json.loads((tmp_path / "d.i7.json").read_text())["tree"]
    assert i7["depth_unit"] == "mutations"
    x = [0.0] * len(t)
    for i in range(1, len(t)):
        x[i] = x[t.parent[i]] + t.edge[i]
    expected = {t.leaf_id[i]: x[i] for i in t.leaves()}
    assert all(leaf["depth"] == pytest.approx(expected[leaf["id"]]) for leaf in i7["leaves"])
    make_figure(t, PARENTS, config(), tmp_path / "e.pdf", {})
    assert "depth_unit" not in json.loads((tmp_path / "e.i7.json").read_text())["tree"]


def test_dash_bars_counted_in_report(tmp_path):
    report = make_figure(standard_tree(), PARENTS, config(), tmp_path / "f.pdf", {"tree": "x"})
    (bar,) = report["dash_bars"]
    assert bar["pos"] == 2
    assert (
        sum(bar["coloured"].values())
        + sum(bar["transparent"].values())
        + sum(bar["no_colour"].values())
        == report["rows"]
    )
    assert set(bar["coloured"]) == {"N"} and set(bar["transparent"]) == {"K"}


def test_dash_bar_key_matching_no_leaf_is_an_error(tmp_path):
    bars = [DashBar(2, {"N": "#e72f27", "Q": "#03569b"}, [])]
    with pytest.raises(DashBarError, match=r"position 2: colour key\(s\) \['Q'\]"):
        make_figure(
            standard_tree(), PARENTS, replace(config(), dash_bars=bars), tmp_path / "f.pdf", {}
        )
    assert not (tmp_path / "f.pdf").exists()


def test_leaf_name_layer_names_every_drawn_leaf_searchably(tmp_path):
    pdf = tmp_path / "f.pdf"
    report = make_figure(
        standard_tree(), PARENTS, replace(config(), leaf_names=True), pdf, {"tree": "x"}
    )
    assert report["leaf_names"] == report["rows"]
    raw = pdf.read_bytes()
    assert b"/FontFile2" in raw  # Type 42 (TrueType) embedded: text carries a Unicode map
    assert b"/ca 0" in raw  # the names are transparent: searchable, no ink on the tree
    plain = tmp_path / "plain.pdf"
    assert make_figure(standard_tree(), PARENTS, config(), plain, {})["leaf_names"] is None
    assert b"/FontFile2" not in plain.read_bytes()


def test_leaf_name_layer_leaves_visible_labels_where_they_were(tmp_path):
    on = make_figure(
        standard_tree(), PARENTS, replace(config(), leaf_names=True), tmp_path / "a.pdf", {}
    )
    off = make_figure(standard_tree(), PARENTS, config(), tmp_path / "b.pdf", {})
    assert on["aa_label_placement"] == off["aa_label_placement"]
    assert on["strains"] == off["strains"]


def test_inset_goes_in_clear_space_and_labels_keep_off_it(tmp_path, monkeypatch):
    import af.tree.draw.figure as figure_module

    drawn = []
    real_render = figure_module.render

    def capture(spec, pdf):
        drawn.append(real_render(spec, pdf))
        return drawn[-1]

    monkeypatch.setattr(figure_module, "render", capture)
    report = make_figure(standard_tree(), PARENTS, config(), tmp_path / "i.pdf", {})
    x0, y0, x1, y1 = report["inset"]
    assert x1 - x0 == pytest.approx(140.0)
    assert y1 <= 975.0 - 7.5 + 1e-6  # Geometry.bottom - inset_gap: above the last row
    assert drawn[0].placed  # the synthetic tree has labels to keep off the inset
    for p in drawn[0].placed:
        bx0, by0, bx1, by1 = p.box
        assert min(bx1, x1) <= max(bx0, x0) or min(by1, y1) <= max(by0, y0)


def test_place_inset_avoids_ink_and_prefers_low_then_left():
    from af.tree.draw.place import Grid
    from af.tree.draw.render import place_inset

    g = Grid(300, 200)
    g.add_box((0, 150, 100, 200))  # ink in the bottom-left corner
    x0, y0, x1, y1 = place_inset(g, 50, 20, lowest=200, right=300)
    assert (x0, y1) == (102.0, 200.0)  # as low as allowed, just right of the ink (pad 2)
    assert g.cost((x0, y0, x1, y1)) == 0


def test_place_inset_without_clear_space_is_an_error():
    from af.tree.draw.place import Grid
    from af.tree.draw.render import InsetPlacementError, place_inset

    g = Grid(100, 100)
    g.add_box((0, 0, 100, 100))
    with pytest.raises(InsetPlacementError):
        place_inset(g, 50, 20, lowest=100, right=100)


def test_clade_label_size_for_a_clade_not_shown_is_an_error(tmp_path):
    report = make_figure(standard_tree(), PARENTS, config(), tmp_path / "a.pdf", {})
    shown = sorted(report["clades_shown"])
    ok = config(clade_label_sizes={shown[0]: 7.0})
    r = make_figure(standard_tree(), PARENTS, ok, tmp_path / "b.pdf", {})
    assert r["clade_label_sizes"] == {shown[0]: 7.0}
    with pytest.raises(SectionOverrideError, match="not shown"):
        make_figure(
            standard_tree(),
            PARENTS,
            config(clade_label_sizes={"NO.SUCH": 7.0}),
            tmp_path / "c.pdf",
            {},
        )


def _captured_spec(tmp_path, monkeypatch, cfg=None):
    import af.tree.draw.figure as figure_module

    specs = []
    real_render = figure_module.render

    def capture(spec, pdf):
        specs.append(spec)
        return real_render(spec, pdf)

    monkeypatch.setattr(figure_module, "render", capture)
    make_figure(standard_tree(), PARENTS, cfg or config(), tmp_path / "s.pdf", {})
    return specs[0]


def test_month_labels_stay_clear_of_the_matrix_and_ticks_are_centred(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from af.tree.draw.render import _draw_matrix, _Page, row_colours

    spec = _captured_spec(tmp_path, monkeypatch)
    g, pg = spec.geometry, _Page(spec)
    fig = plt.figure(figsize=(g.width / 72, g.height / 72))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, g.width)
    ax.set_ylim(g.height, 0)
    _draw_matrix(ax, spec, pg, row_colours(spec))
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    for t in ax.texts:
        (x0, y_a), (x1, y_b) = to_data.transform(t.get_window_extent())
        top, bottom = min(y_a, y_b), max(y_a, y_b)
        assert bottom <= g.top or top >= pg.bottom  # never inside the matrix
        k = round((min(x0, x1) - g.ts_left) / g.ts_slot - 0.5)  # its slot, between two lines
        assert (
            g.ts_left + k * g.ts_slot < min(x0, x1)
            and max(x0, x1) < g.ts_left + (k + 1) * g.ts_slot
        )
    ticks = cast(LineCollection, ax.collections[0]).get_segments()
    starts = {round(((s[0][0] - g.ts_left) / g.ts_slot) % 1, 6) for s in ticks}
    assert starts == {g.tick_start}
    assert {round((s[1][0] - s[0][0]) / g.ts_slot, 6) for s in ticks} == {g.tick_length}
    plt.close(fig)


def test_separator_rules_stop_at_the_bracket_that_ends_on_their_row(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    from af.tree.draw.render import _draw_brackets, _draw_hz_rules, _Page

    spec = _captured_spec(tmp_path, monkeypatch)
    g, pg = spec.geometry, _Page(spec)
    fig, ax = plt.subplots()
    edges = _draw_brackets(ax, spec, pg, 500.0)
    assert edges  # the synthetic tree has shown clades
    n_lines = len(ax.lines)
    _draw_hz_rules(ax, spec, pg, 480.0, edges)
    ends = {
        round(float(np.asarray(line.get_ydata())[0]), 6): float(np.asarray(line.get_xdata())[1])
        for line in ax.lines[n_lines:]
    }
    for r, x in edges.items():
        if 0 < r < spec.layout.n_rows:
            assert ends[round(pg.row_edge(r), 6)] == x
    assert all(c == g.rule_colour for c in (line.get_color() for line in ax.lines[n_lines:]))
    plt.close(fig)


@pytest.mark.parametrize("drawn", [None, "", "SHORT"])
def test_drawn_title_changes_the_page_only(tmp_path, monkeypatch, drawn):
    from matplotlib.axes import Axes

    texts: list[str] = []
    real_text = Axes.text

    def spy(self, x, y, s, *args, **kwargs):
        texts.append(s)
        return real_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(Axes, "text", spy)
    pdf = tmp_path / "t.pdf"
    report = make_figure(standard_tree(), PARENTS, config(drawn_title=drawn), pdf, {})
    i7 = json.loads(pdf.with_suffix(".i7.json").read_text())
    assert i7["title"] == "TEST tree"  # the figure's name never changes
    expected = "TEST tree" if drawn is None else drawn
    assert report["drawn_title"] == expected
    assert texts.count("TEST tree") == (1 if drawn is None else 0)
    if drawn:
        assert texts.count(drawn) == 1


class _CladeSet:
    """The two things legacy_clade_display asks of a clade set (invented names)."""

    version = "invented@0"
    older = {"X": "old1", "X.1": "old1.a", "X.2": None}

    def __contains__(self, name):
        return name in self.older

    def legacy_name(self, name):
        return self.older[name]


def test_brackets_print_the_older_name_where_upstream_records_one(tmp_path):
    from af.tree.draw.figure import legacy_clade_display

    display = legacy_clade_display(_CladeSet(), ["X", "X.1", "X.2"])
    assert display == {"X": "X (old1)", "X.1": "X.1 (old1.a)"}
    with pytest.raises(ValueError, match="not in"):
        legacy_clade_display(_CladeSet(), ["Y"])
    plain = make_figure(standard_tree(), PARENTS, config(), tmp_path / "p.pdf", {})
    named = make_figure(
        standard_tree(), PARENTS, config(clade_display=display), tmp_path / "n.pdf", {}
    )
    assert named["clades_shown"] == plain["clades_shown"]  # the clades are the same ones
    assert (tmp_path / "n.pdf").read_bytes() != (tmp_path / "p.pdf").read_bytes()


def test_margin_strain_labels_stack_clear_of_each_other_and_of_ink():
    from af.tree.draw.place import Grid
    from af.tree.draw.render import _margin_strain_box

    grid = Grid(200, 200)
    first = _margin_strain_box(3.0, 100.0, 40.0, 8.0, [], grid)
    assert first == (3.0, 96.0, 43.0, 104.0)
    second = _margin_strain_box(3.0, 101.0, 40.0, 8.0, [first], grid)
    assert second[0] == 3.0 and (second[3] <= first[1] or second[1] >= first[3])
    grid.add_box((0, 0, 200, 200))
    with pytest.raises(ValueError, match="no clear margin space"):
        _margin_strain_box(3.0, 100.0, 40.0, 8.0, [], grid)


def test_margin_strain_label_moves_off_its_row_in_strain_steps():
    from af.tree.draw.place import Grid
    from af.tree.draw.render import _margin_strain_box

    grid = Grid(200, 200)
    first = _margin_strain_box(3.0, 100.0, 40.0, 8.0, [], grid, step=12.0)
    second = _margin_strain_box(3.0, 101.0, 40.0, 8.0, [first], grid, step=12.0)
    assert (second[1] + second[3]) / 2 == pytest.approx(113.0)  # its row + one step


def test_aa_labels_start_no_further_left_than_label_left(tmp_path, monkeypatch):
    import af.tree.draw.figure as figure_module
    from af.tree.draw.render import Geometry

    drawn = []
    real_render = figure_module.render

    def capture(spec, pdf):
        drawn.append(real_render(spec, pdf))
        return drawn[-1]

    monkeypatch.setattr(figure_module, "render", capture)
    make_figure(standard_tree(), PARENTS, config(), tmp_path / "a.pdf", {})
    leftmost = min(p.box[0] for p in drawn[0].placed)
    bound = leftmost + 2.0  # the default placement has a label left of this
    make_figure(
        standard_tree(),
        PARENTS,
        config(geometry=Geometry(label_left=bound)),
        tmp_path / "b.pdf",
        {},
    )
    assert len(drawn[1].placed) == len(drawn[0].placed)
    assert min(p.box[0] for p in drawn[1].placed) >= bound


def test_place_inset_takes_the_clear_corner_nearest_the_wanted_one():
    from af.tree.draw.place import Grid
    from af.tree.draw.render import place_inset

    g = Grid(300, 200)
    x0, y0, _, _ = place_inset(g, 50, 20, lowest=200, right=300, near=(40.0, 150.0))
    assert (x0, y0) == (40.0, 150.0)
    g.add_box((30, 140, 80, 180))  # ink at the wanted corner: the nearest clear one instead
    box = place_inset(g, 50, 20, lowest=200, right=300, near=(40.0, 150.0))
    assert g.cost(box) == 0 and box != (40.0, 150.0, 90.0, 170.0)


def test_a_figure_in_the_round_fonts_draws_and_records_them(tmp_path):
    from pathlib import Path

    import matplotlib

    from af.util.fonts import FontNames

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # stand-ins every env has
    names = FontNames("STIXGeneral.ttf", "STIXGeneralBol.ttf", "DejaVuSansMono.ttf")
    fonts = names.locate(lambda name: ttf / name).resolve()
    make_figure(standard_tree(), PARENTS, config(fonts=fonts), tmp_path / "f.pdf", {"tree": "x"})
    assert b"STIXGeneral" in (tmp_path / "f.pdf").read_bytes()
    inputs = json.loads((tmp_path / "f.i7.json").read_text())["provenance"]["inputs"]
    assert set(inputs) >= {"tree", "font_sans", "font_sans_bold", "font_mono"}


def test_margin_strain_labels_keep_off_aa_labels_and_their_leaders():
    from af.tree.draw.place import Grid, Placed
    from af.tree.draw.render import _margin_strain_box

    grid = Grid(200, 200)
    # an aa label in the margin on the strain's row, and another whose leader runs across it
    on_row = Placed(1, "K2N", (150.0, 100.0), (2.0, 96.0, 30.0, 104.0), 0.0)
    leader = Placed(2, "A6S", (150.0, 80.0), (2.0, 110.0, 30.0, 118.0), 0.0)
    box = _margin_strain_box(3.0, 100.0, 40.0, 8.0, [], grid, [on_row, leader])
    assert box[0] == 3.0 and box[1] >= 104.0 or box[3] <= 96.0
    assert box[1] >= 118.0 or box[3] <= 96.0  # not between the leader's ends either


# R7 (21-22 Oct 2026): pinned labels, per-tree placer weights, arrowhead size, strain offsets.
def _pin_config(pins, **kw):
    labels = replace(load_defaults().labels, target_labels=0, min_share=0.05)
    return config(labels=labels, label_pins=pins, **kw)


def _capture(monkeypatch):
    import af.tree.draw.figure as figure_module

    seen = []
    real_render = figure_module.render

    def capture(spec, pdf):
        seen.append((spec, real_render(spec, pdf)))
        return seen[-1][1]

    monkeypatch.setattr(figure_module, "render", capture)
    return seen


def test_a_pinned_label_sits_at_its_offset_from_its_node_and_is_counted(tmp_path, monkeypatch):
    from af.tree.draw.figure import LabelPin
    from af.tree.draw.render import _Page

    pin = LabelPin("EPI_ISL_0000010", "EPI_ISL_0000019", "L7F", -0.05, 0.02, "invented")
    seen = _capture(monkeypatch)
    report = make_figure(standard_tree(), PARENTS, _pin_config([pin]), tmp_path / "p.pdf", {})
    spec, drawn = seen[0]
    (k,) = spec.label_pins
    lab = spec.labels[k]
    page = _Page(spec)
    (box,) = [p.box for p in drawn.placed if p.text == "L7F"]
    g = spec.geometry
    assert box[0] == pytest.approx(float(page.nx[lab.node]) - 0.05 * g.height - 1)
    assert box[1] == pytest.approx(float(page.ny[lab.node]) + 0.02 * g.height)
    assert report["label_pins"]["pins"] == 1 and report["label_pins"]["pinned"] == 1
    assert sorted(report["label_pins"]["placed automatically"]) == ["A6S", "K2N"]


def test_a_pin_matching_no_label_is_an_error_unless_optional(tmp_path):
    from af.tree.draw.figure import LabelPin, LabelPinError

    wrong = LabelPin("EPI_ISL_0000010", "EPI_ISL_0000019", "Q9R", -0.05, 0.0, "invented")
    with pytest.raises(LabelPinError, match="no label of that text"):
        make_figure(standard_tree(), PARENTS, _pin_config([wrong]), tmp_path / "a.pdf", {})
    optional = replace(wrong, optional=True, reason="invented: drawn elsewhere")
    report = make_figure(standard_tree(), PARENTS, _pin_config([optional]), tmp_path / "b.pdf", {})
    assert report["label_pins"]["optional pins unused"] == [
        {"text": "Q9R", "reason": "invented: drawn elsewhere"}
    ]
    missing = replace(wrong, first="EPI_ISL_9999999")
    with pytest.raises(LabelPinError, match="not in the tree"):
        make_figure(standard_tree(), PARENTS, _pin_config([missing]), tmp_path / "c.pdf", {})


def test_no_pins_no_report_entry(tmp_path):
    report = make_figure(standard_tree(), PARENTS, _pin_config([]), tmp_path / "n.pdf", {})
    assert "label_pins" not in report


def test_placer_keeps_fixed_boxes_and_places_the_rest_clear_of_them():
    from af.tree.draw.place import Grid, place_labels

    grid = Grid(200, 200)
    items = [(1, "K2N", (150.0, 100.0)), (2, "A6S", (150.0, 101.0))]
    pinned = (60.0, 95.0, 90.0, 107.0)
    placed = place_labels(items, grid, lambda t: 6.0 * len(t), 10.0, fixed={0: pinned})
    boxes = {p.key: p.box for p in placed}
    assert boxes[1] == pinned
    other = boxes[2]
    assert min(other[2], pinned[2]) <= max(other[0], pinned[0]) or min(other[3], pinned[3]) <= max(
        other[1], pinned[1]
    )


def test_tree_weights_heads_and_strain_offsets_come_from_the_round(tmp_path):
    from af.tree.draw.place import Weights
    from af.tree.draw.render import Geometry, _head_scale, _weights

    assert _weights(Geometry()) is None  # defaults: the placer's own
    assert _weights(Geometry(label_ink_weight=50.0, label_flat_weight=0.0)) == Weights(
        ink=50.0, flat=0.0
    )
    g = Geometry(bracket_head_length=4.0, bracket_head_share=0.4)
    assert _head_scale(g, 100.0) == 1 and _head_scale(g, 5.0) == pytest.approx(0.5)
    assert _head_scale(Geometry(), 1.0) == 1  # no cap set: heads as before
    with pytest.raises(ValueError, match="no strain label"):
        make_figure(
            standard_tree(),
            PARENTS,
            config(strain_dy={"EPI_ISL_0000001": 3.0}),
            tmp_path / "s.pdf",
            {},
        )
    report = make_figure(
        standard_tree(),
        PARENTS,
        config(
            strain_dy={"EPI_ISL_0000012": 3.0}, geometry=Geometry(strain_x=3.0, strains_first=True)
        ),
        tmp_path / "t.pdf",
        {},
    )
    assert report["strains"][0]["drawn"]


# R8 (23 Oct 2026): pinned labels clear other labels' glyphs; glyph overprints counted.
def test_overlapping_pins_move_the_least_distance_that_clears():
    from af.tree.draw.aa_labels import AALabel
    from af.tree.draw.render import FigureSpec, Geometry, _clear_pins, _glyph_box, _hits

    g = Geometry(label_font=10.0, pin_clearance=1.0)
    labels = [AALabel(1, ["K2N"], "a", "b", 5), AALabel(2, ["A6S"], "c", "d", 5)]
    spec = cast(FigureSpec, type("S", (), {"labels": labels})())
    width = lambda t: 6.0 * len(t)  # noqa: E731
    fixed = {0: (100.0, 100.0, 120.0, 113.0), 1: (100.0, 104.0, 120.0, 117.0)}
    out, moves = _clear_pins(spec, fixed, [], width, g)
    assert out[0] == fixed[0]  # the upper one, settled first, stays
    a, b = (_glyph_box(out[k], labels[k].subs[0], width, g) for k in (0, 1))
    assert not _hits(a, b, 1.0) and _hits(a, b, 1.5)  # cleared, by no more than needed
    assert moves == [{"text": "A6S", "dy": out[1][1] - fixed[1][1]}] and moves[0]["dy"] > 0


def test_an_automatic_label_printed_over_a_pin_moves_the_pin():
    from af.tree.draw.place import Placed
    from af.tree.draw.render import Geometry, _clear_pins_after, _glyph_box, _hits

    g = Geometry(label_font=10.0, pin_clearance=1.0)
    width = lambda t: 6.0 * len(t)  # noqa: E731
    pin = Placed(1, "K2N", (150.0, 100.0), (100.0, 100.0, 120.0, 113.0), 0.0)
    auto = Placed(2, "A6S", (150.0, 103.0), (102.0, 102.0, 122.0, 115.0), 0.0)
    moves: list[dict] = []
    out = _clear_pins_after([pin, auto], {pin.box}, [], width, g, moves)
    assert out[1] == auto and out[0].box != pin.box
    assert not _hits(
        _glyph_box(out[0].box, "K2N", width, g), _glyph_box(auto.box, "A6S", width, g), 0.0
    )
    assert len(moves) == 1 and moves[0]["text"] == "K2N"


def test_clearance_reports_moves_and_overprints(tmp_path):
    from af.tree.draw.figure import LabelPin
    from af.tree.draw.render import Geometry

    pin = LabelPin("EPI_ISL_0000010", "EPI_ISL_0000019", "L7F", -0.05, 0.02, "invented")
    plain = make_figure(standard_tree(), PARENTS, _pin_config([pin]), tmp_path / "a.pdf", {})
    assert "moved to clear" not in plain["label_pins"]  # clearance off: as before
    cleared = make_figure(
        standard_tree(),
        PARENTS,
        _pin_config([pin], geometry=Geometry(pin_clearance=1.0)),
        tmp_path / "b.pdf",
        {},
    )
    assert cleared["label_pins"]["moved to clear"] == []
    assert cleared["label_pins"]["glyph overprints"] == 0


def test_a_label_stroke_thickens_labels_and_keeps_them_searchable(tmp_path):
    import subprocess

    from af.tree.draw.render import Geometry

    labels = replace(load_defaults().labels, target_labels=0, min_share=0.05)
    plain = make_figure(standard_tree(), PARENTS, config(labels=labels), tmp_path / "a.pdf", {})
    stroked = make_figure(
        standard_tree(),
        PARENTS,
        config(labels=labels, geometry=Geometry(label_stroke=0.25)),
        tmp_path / "b.pdf",
        {},
    )
    assert stroked["aa_label_placement"] == plain["aa_label_placement"]  # the metrics are kept
    assert (tmp_path / "a.pdf").read_bytes() != (tmp_path / "b.pdf").read_bytes()
    try:
        text = subprocess.run(
            ["pdftotext", str(tmp_path / "b.pdf"), "-"], capture_output=True, text=True, check=True
        ).stdout
    except FileNotFoundError:
        pytest.skip("pdftotext not installed")
    assert {"K2N", "A6S", "L7F"} <= set(text.split())  # the invisible copy is text


def test_a_mono_fallback_draws_with_its_own_stroke_and_says_so(tmp_path):
    import hashlib
    import shutil
    from pathlib import Path

    import matplotlib

    from af.tree.draw.render import Geometry
    from af.util.fonts import FontNames

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"  # stand-ins every env has
    mono = tmp_path / "Mono.ttf"
    shutil.copyfile(ttf / "DejaVuSansMono.ttf", mono)
    sha = hashlib.sha256(mono.read_bytes()).hexdigest()
    labels = replace(load_defaults().labels, target_labels=0, min_share=0.05)
    geometry = Geometry(label_stroke_fallback=0.25)
    reports = {}
    for name, path in (("preferred", mono), ("fallback", tmp_path / "no" / "Mono.ttf")):
        names = FontNames(
            "STIXGeneral.ttf",
            "STIXGeneralBol.ttf",
            "Mono.ttf",
            path,
            sha,
            "cmtt10.ttf",
        )
        fonts = names.locate(lambda n: ttf / n).resolve()
        reports[name] = make_figure(
            standard_tree(),
            PARENTS,
            config(labels=labels, geometry=geometry, fonts=fonts),
            tmp_path / f"{name}.pdf",
            {},
        )["font_mono"]
    assert reports["preferred"]["chosen"] == "preferred"
    assert reports["preferred"]["label_stroke"] == 0.0
    assert reports["fallback"]["chosen"] == "fallback"
    assert reports["fallback"]["font"] == "cmtt10.ttf"
    assert reports["fallback"]["label_stroke"] == 0.25
    assert "not on this machine" in reports["fallback"]["reason"]
