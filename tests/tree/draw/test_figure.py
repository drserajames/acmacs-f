"""End to end: a synthetic tree to PDF + I7 + draw report."""

import hashlib
import json
from dataclasses import replace

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")
pytest.importorskip(
    "matplotlib", reason="matplotlib not installed: the tree renderer needs it (pyproject, WS1)"
)

from af.tree.draw.defaults import load_defaults  # noqa: E402
from af.tree.draw.figure import FigureConfig, Overrides, make_figure  # noqa: E402
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
    _place_clade_names(ax, [(10.0, 100.0, "X.1.1"), (10.0, 101.0, "X.1.2"), (20.0, 100.0, "X")])
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
