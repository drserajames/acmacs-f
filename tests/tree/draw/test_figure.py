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
