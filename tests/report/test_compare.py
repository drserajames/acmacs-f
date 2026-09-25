"""Comparison metrics on synthetic maps and trees (invented names only)."""

from __future__ import annotations

import datetime as dt
import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from af.report.compare import geo, maps, trees
from af.report.compare.reference_ae import (
    absolute_viewport,
    colour_labels,
    style_chain,
    transform,
    window_rules,
)
from af.report.compare.run import (
    Expected,
    Limits,
    MapLimits,
    apply_expected,
    compare_report,
    map_checks,
    slot_status,
)
from af.util.config import ConfigError, load_config, parse_config


def _rotate(points: list[tuple[float, float]], degrees: float) -> list[tuple[float, float]]:
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    return [(x * c - y * s, x * s + y * c) for x, y in points]


def _cloud(n: int, seed: int, spread: float = 3.0) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.gauss(0, spread), rng.gauss(0, spread)) for _ in range(n)]


def _map(points: list[tuple[float, float]], clades: list[str]) -> dict[str, Any]:
    antigens = [
        {"id": f"TEST/{i}/2025", "name": f"TEST/{i}/2025", "passage_class": "cell",
         "xy": list(p), "shown": True, "in_viewport": True, "clade": c, "colour": "#000000",
         "greyed": False}
        for i, (p, c) in enumerate(zip(points, clades, strict=True))
    ]  # fmt: skip
    return {"title": "synthetic", "map": {"antigens": antigens, "sera": []}}


def test_procrustes_ignores_rotation_and_translation_and_reports_the_angle() -> None:
    a = _cloud(50, 0)
    b = [(x + 5, y - 2) for x, y in _rotate(a, 30)]
    fit = maps.procrustes(a, b)
    assert fit.rmsd < 1e-9
    assert abs(fit.rotation_deg + 30) < 1e-6 and not fit.reflected


def test_procrustes_detects_reflection() -> None:
    a = _cloud(40, 1)
    fit = maps.procrustes(a, [(x, -y) for x, y in _rotate(a, 70)])
    assert fit.reflected and fit.rmsd < 1e-9


def test_procrustes_does_not_scale() -> None:
    a = _cloud(30, 2)
    assert maps.procrustes(a, [(2 * x, 2 * y) for x, y in a]).rmsd > 1.0


def test_adjusted_rand_ignores_label_names() -> None:
    x = ["a", "a", "b", "b", "c", "c"]
    assert maps.adjusted_rand(x, ["p", "p", "q", "q", "r", "r"]) == 1.0
    assert maps.adjusted_rand(x, ["p", "q", "p", "q", "p", "q"]) < 0.1


def test_whole_clade_move_is_caught_by_centroids_not_p95() -> None:
    big = [(x * 0.3, y * 0.3) for x, y in _cloud(190, 3, 1.0)]
    small = [(6 + x * 0.3, y * 0.3) for x, y in _cloud(10, 4, 1.0)]
    clades = ["X"] * 190 + ["Y"] * 10
    moved = big + [(x + 4, y) for x, y in small]  # clade Y, 5% of points, moves 4 units
    res = maps.compare(_map(big + small, clades), _map(moved, clades))
    assert res["procrustes"]["p95"] < 0.5
    assert res["clade_centroids"]["max_abs_diff"] > 3.0


def test_missing_points_are_counted() -> None:
    pts = [(float(i), 0.0) for i in range(10)]
    res = maps.compare(_map(pts, ["X"] * 10), _map(pts[:8], ["X"] * 8))
    assert res["antigens"]["only_ref"] == 2 and res["antigens"]["common"] == 8


def test_duplicate_keys_are_dropped_and_counted() -> None:
    pts = [(float(i), 0.0) for i in range(6)]
    ref = _map(pts, ["X"] * 6)
    ref["map"]["antigens"][1]["name"] = ref["map"]["antigens"][0]["name"]
    res = maps.compare(ref, _map(pts, ["X"] * 6), how="name")
    assert res["antigens"]["ambiguous_dropped"]["ref"] == 2


def _random_tree(names: list[str], seed: int) -> trees.Tree:
    rng = random.Random(seed)
    nodes: list[Any] = [("leaf", n) for n in names]
    while len(nodes) > 1:
        i, j = sorted(rng.sample(range(len(nodes)), 2))
        b, a = nodes.pop(j), nodes.pop(i)
        nodes.append(("inner", [a, b]))
    tree = trees.Tree()
    stack = [(nodes[0], -1)]
    while stack:
        (kind, value), parent = stack.pop()
        me = tree.add(parent)
        if kind == "leaf":
            tree.leaf_name[me] = value
            tree.leaf_clades[me] = ["C" + value[-1]]
        else:
            stack.extend((child, me) for child in value)
    return tree


def test_rf_is_zero_for_the_same_tree_and_high_for_a_random_one() -> None:
    names = [f"TEST/{i}/2025" for i in range(200)]
    t1 = _random_tree(names, 1)
    same = trees.compare(t1, _random_tree(names, 1))
    assert same["splits_by_size"][">=2"]["rf"] == 0 and same["tip_sets_identical"]
    assert trees.compare(t1, _random_tree(names, 2))["rf_normalised"] > 0.9


def test_leaves_match_through_the_ae_hash_suffix() -> None:
    names = [f"TEST/{i}/2025" for i in range(50)]
    res = trees.compare(
        _random_tree(names, 3), _random_tree([n + "_OR_0A1B2C3D" for n in names], 3)
    )
    assert res["common_leaves"] == 50 and res["rf_normalised"] == 0


def test_tree_rejects_child_before_parent() -> None:
    with pytest.raises(ValueError):
        trees.Tree().add(5)


def test_reference_transform_and_viewport_follow_ae() -> None:
    # t = [a, b, c, d]: x' = a*x + c*y, y' = b*x + d*y (90 degrees here).
    assert transform([[1.0, 0.0], [0.0, 2.0]], [0.0, 1.0, -1.0, 0.0]) == [[0.0, 1.0], [-2.0, 0.0]]
    assert transform([[float("nan"), 0.0]], None) == [None]
    # Hull x 0..4, y 0..2: rounded size 5 x 3, centre (2, 1). Stored V is in the recentred frame.
    xy: list[list[float] | None] = [[0.0, 0.0], [4.0, 2.0]]
    assert absolute_viewport(xy, [1.0, 1.0, 3.0, 2.0]) == [0.5, 0.5, 3.0, 2.0]
    assert absolute_viewport(xy, None) == [-0.5, -0.5, 5, 3]


def test_reference_style_chain_and_colour_collisions_are_visible() -> None:
    styles = {
        "main": {"A": [{"R": "-base"}, {"F": "#111111", "L": {"t": "Clade Q"}}]},
        "-base": {"A": [{"F": "#111111", "L": {"t": "Clade P"}}, {"R": "-missing"}]},
    }
    assert len(style_chain(styles, "main")) == 2
    assert colour_labels(styles, "main") == {"#111111": "Clade P | Clade Q"}


def _checks(res: dict[str, Any]) -> list[dict[str, Any]]:
    return map_checks(res, MapLimits(rotation_deg_max=1.0))


def test_expected_difference_is_reported_and_a_stale_one_fails() -> None:
    a = _cloud(30, 5)
    turned = maps.compare(_map(a, ["X"] * 30), _map(_rotate(a, 8), ["X"] * 30))
    same = maps.compare(_map(a, ["X"] * 30), _map(a, ["X"] * 30))
    reason = [
        Expected(
            "map/test/all",
            "rotation deg",
            "deliberate hand rotation",
            "a reviewer",
            dt.date(2026, 9, 25),
        )
    ]
    checks = _checks(turned)
    assert slot_status(checks) == "FAIL"
    apply_expected("map/test/all", checks, reason)
    assert slot_status(checks) == "ok (expected differences)"
    checks = _checks(same)
    apply_expected("map/test/all", checks, reason)
    assert slot_status(checks) == "FAIL"  # the reason no longer applies: stale
    stale = next(c for c in checks if c["check"] == "rotation deg")
    assert "STALE" in stale["expected"] and "a reviewer, 2026-09-25" in stale["expected"]


def _geo(counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        "periods": [
            {
                "period": month,
                "locations": [
                    {"name": loc, "points": [{"color": "#AA0000", "count": n}]}
                    for loc, n in locs.items()
                ],
            }
            for month, locs in counts.items()
        ]
    }


def test_geo_counts_dot_differences_per_month() -> None:
    ref = _geo({"2026-01": {"PLACE A": 3, "PLACE B": 1}, "2026-02": {"PLACE A": 2}})
    new = _geo({"2026-01": {"PLACE A": 3, "PLACE C": 1}, "2026-03": {"PLACE A": 1}})
    res = geo.compare(ref, new)
    assert res["months_compared"] == ["2026-01"]
    assert res["months_only_ref"] == ["2026-02"] and res["months_only_new"] == ["2026-03"]
    jan = res["per_month"]["2026-01"]["location"]
    assert jan["abs_diff"] == 2 and jan["frac_diff"] == 2 / 8
    assert res["per_month"]["2026-01"]["clade"]["frac_diff"] == 0


def test_expectation_without_approver_is_rejected(tmp_path: Path) -> None:
    limits = tmp_path / "limits.toml"
    limits.write_text(
        '[adoption]\nstatus = "final"\nadopted_by = "a reviewer"\nadopted = 2026-09-25\n'
        '[[expected]]\nslot = "map/test/all"\ncheck = "rotation deg"\nreason = "hand rotation"\n'
    )
    with pytest.raises(ConfigError, match="approved_by"):
        load_config(limits, Limits)


def test_limits_file_must_say_who_adopted_it() -> None:
    with pytest.raises(ConfigError, match="adoption"):
        parse_config({"map": {"p95_displacement_max": 0.5}}, Limits, base_dir=Path())


def test_provisional_limits_must_name_their_review(tmp_path: Path) -> None:
    limits = parse_config(
        {
            "adoption": {
                "status": "provisional",
                "adopted_by": "a reviewer",
                "adopted": dt.date(2026, 9, 25),
            }
        },
        Limits,
        base_dir=tmp_path,
    )
    manifest = {"report": "r", "built": "", "figures": []}
    with pytest.raises(ValueError, match="review"):
        compare_report(manifest, tmp_path, limits, "name")


def test_resolving_a_polytomy_keeps_reference_recovery_at_one() -> None:
    """A new tree that resolves a star the reference left collapsed recovers every ref split."""
    star = trees.Tree()
    root = star.add(-1)
    left, right = star.add(root), star.add(root)
    names = [f"TEST/{i}/2025" for i in range(8)]
    for i, name in enumerate(names):
        leaf = star.add(left if i < 4 else right)
        star.leaf_name[leaf] = name
        star.leaf_clades[leaf] = ["C"]
    resolved = _random_tree(names[:4], 7)  # a resolved left half, re-rooted under a new root
    full = trees.Tree()
    top = full.add(-1)
    offset = full.add(top)
    mapped: dict[int, int] = {}
    for node, parent in enumerate(resolved.parent):
        mapped[node] = full.add(offset if parent < 0 else mapped[parent])
    for node, name in resolved.leaf_name.items():
        full.leaf_name[mapped[node]] = name
        full.leaf_clades[mapped[node]] = ["C"]
    right_node = full.add(top)
    for name in names[4:]:
        leaf = full.add(right_node)
        full.leaf_name[leaf] = name
        full.leaf_clades[leaf] = ["C"]
    res = trees.compare(star, full)["splits_by_size"][">=2"]
    assert res["ref_recovered"] == 1.0 and res["rf"] > 0


def test_newick_reader_and_zero_length_collapse(tmp_path: Path) -> None:
    path = tmp_path / "t.nwk"
    path.write_text("((leaf1:1,leaf2:1):0,('leaf 3':1,leaf4:1)0.9:1,leaf5:1);")
    full = trees.read_newick(path)
    assert sorted(full.leaf_name.values()) == ["leaf 3", "leaf1", "leaf2", "leaf4", "leaf5"]
    assert len(full.parent) == 8
    collapsed = trees.read_newick(path, collapse_at_most=0.0)
    assert len(collapsed.parent) == 7  # the zero-length (1,2) node is gone
    res = trees.compare(full, collapsed)["splits_by_size"][">=2"]
    assert res["ref"] == 2 and res["new"] == 1 and res["shared"] == 1


def test_newick_rejects_unbalanced(tmp_path: Path) -> None:
    path = tmp_path / "bad.nwk"
    path.write_text("((A:1,B:1);")
    with pytest.raises(ValueError, match="unbalanced"):
        trees.read_newick(path)


def test_reference_greying_comes_from_the_window_rule_not_the_colour() -> None:
    styles = {
        "clades-12m": {"A": [{"R": "-clades"}, {"R": "-o12m-grey"}]},
        "clades": {"A": [{"R": "-clades"}]},
        "-clades": {"A": [{"T": {"C": "X"}, "F": "#111111", "L": {"t": "X"}}]},
        "-o12m-grey": {"A": [{"T": {"o12m": True}, "F": "grey"}]},
    }
    assert window_rules(styles, "clades-12m") == {"o12m": "grey"}
    assert window_rules(styles, "clades") == {}


def test_i7_rejects_a_drawn_point_without_colour() -> None:
    from af.report.i7 import I7Error, validate

    point = {"id": "p", "name": "point one", "passage_class": "cell", "xy": [0.0, 0.0],
             "shown": True, "in_viewport": True, "clade": None, "colour": None}  # fmt: skip
    doc = {
        "i7_version": 1, "kind": "map", "title": "t", "placeholder": False,
        "figure": {"pdf": "figure.pdf", "sha256": "0", "pages": 1}, "provenance": {},
        "map": {"chart": "c", "window": {"name": "all"}, "viewport": [0, 0, 1, 1],
                "clade_scheme": "s", "antigens": [point], "sera": [], "legend": []},
    }  # fmt: skip
    with pytest.raises(I7Error, match="drawn but colour is None"):
        validate(doc)


def test_amendments_are_printed_with_the_status(tmp_path: Path) -> None:
    from af.report.compare.run import markdown

    limits = parse_config(
        {"adoption": {"status": "provisional", "adopted_by": "a reviewer",
                      "adopted": dt.date(2026, 9, 25), "review": "later",
                      "amendments": [{"date": dt.date(2026, 9, 26), "by": "a reviewer",
                                      "change": "tree limit withdrawn"}]}},
        Limits, base_dir=tmp_path,
    )  # fmt: skip
    text = markdown({"report": "r", "built": "b"}, [], "l.toml", "name", limits.adoption)
    assert "amended 2026-09-26 by a reviewer: tree limit withdrawn" in text


def test_excused_points_are_removed_and_a_stale_list_fails(tmp_path: Path) -> None:
    from af.report.compare.run import Excused, excuse_points

    pts = [(float(i), float(i % 3)) for i in range(12)]
    ref = _map(pts[:10], ["X"] * 10)
    new = _map(pts, ["X"] * 12)  # new shows two extra points
    keys = {maps.point_key(p, "name") for p in new["map"]["antigens"][10:]}
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("# dropped hide rule\n" + "\n".join(sorted(keys)) + "\n")
    entry = Excused(["map/x/all"], keys_file, "rule dropped", "a reviewer", dt.date(2026, 9, 25))
    notes = excuse_points("map/x/all", ref, new, [(entry, keys)], "name")
    assert notes[0]["stale"] == 0
    assert maps.compare(ref, new)["antigens"]["jaccard"] == 1.0
    again = _map(pts, ["X"] * 12)  # both sides now show them: the excuse no longer applies
    notes = excuse_points("map/x/all", _map(pts, ["X"] * 12), again, [(entry, keys)], "name")
    assert notes[0]["stale"] == 2 and "STALE" in notes[0]["note"]


def test_one_sided_points_are_listed_even_when_the_slot_passes(tmp_path: Path) -> None:
    from af.report.compare.run import markdown

    pts = [(float(i), float(i % 3)) for i in range(200)]
    res = maps.compare(_map(pts[:199], ["X"] * 199), _map(pts, ["X"] * 200))
    checks = map_checks(res, MapLimits(antigens_jaccard_min=0.99))
    row = {"slot": "map/x/all", "status": slot_status(checks), "checks": checks, "detail": res}
    limits = parse_config(
        {"adoption": {"status": "final", "adopted_by": "a reviewer",
                      "adopted": dt.date(2026, 9, 25)}},
        Limits, base_dir=tmp_path,
    )  # fmt: skip
    text = markdown({"report": "r", "built": "b"}, [row], "l.toml", "name", limits.adoption)
    assert row["status"] == "ok"  # Jaccard 0.995 passes the limit ...
    assert "antigens only in new (1)" in text  # ... but the point is still listed


def test_a_point_outside_one_frame_is_a_frame_difference_not_a_missing_virus() -> None:
    pts = [(float(i), float(i % 3)) for i in range(10)]
    ref, new = _map(pts, ["X"] * 10), _map(pts, ["X"] * 10)
    ref["map"]["antigens"][4]["in_viewport"] = False  # the reference frame cuts it off
    res = maps.compare(ref, new)["antigens"]
    assert res["jaccard"] == 1.0 and res["only_new"] == 0
    assert res["in_frame_only_new_keys"] == [maps.point_key(new["map"]["antigens"][4], "name")]


def test_map_flags_are_printed(tmp_path: Path) -> None:
    from af.report.compare.run import markdown

    pts = [(float(i), float(i % 3)) for i in range(10)]
    res = maps.compare(_map(pts, ["X"] * 10), _map(pts, ["X"] * 10))
    checks = map_checks(res, MapLimits())
    row = {"slot": "map/x/all", "status": "ok", "checks": checks, "detail": res,
           "flags": ["move refused: guard 4.5 u > cap 4.0 u"]}  # fmt: skip
    limits = parse_config(
        {"adoption": {"status": "final", "adopted_by": "a reviewer",
                      "adopted": dt.date(2026, 9, 25)}},
        Limits, base_dir=tmp_path,
    )  # fmt: skip
    text = markdown({"report": "r", "built": "b"}, [row], "l.toml", "name", limits.adoption)
    assert "- map/x/all: move refused: guard 4.5 u > cap 4.0 u" in text


def test_orientation_is_of_the_bulk_not_pulled_by_moved_points() -> None:
    a = _cloud(200, 11)
    b = list(a)
    for i in range(20):  # 10% of points move 6 units: a real change in the map
        b[i] = (b[i][0] + 6.0, b[i][1] + 2.0)
    res = maps.compare(_map(a, ["X"] * 200), _map(b, ["X"] * 200))["procrustes"]
    assert abs(res["rotation_deg"]) < 1e-6  # the bulk is not turned at all
    assert abs(res["rotation_deg_all_points"]) > abs(res["rotation_deg"])


def test_loose_matching_ignores_spelling_but_keeps_distinct_points_apart() -> None:
    ref = _map([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], ["X"] * 3)
    new = _map([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)], ["X"] * 3)
    ref["map"]["antigens"][0]["name"] = "PORT-TOWN ONE"
    new["map"]["antigens"][0]["name"] = "PORT TOWN ONE"
    assert maps.compare(ref, new, "name")["antigens"]["jaccard"] < 1.0
    assert maps.compare(ref, new, "loose")["antigens"]["jaccard"] == 1.0
    assert maps.normalise_key("PORT TOWN ONE|cell", "loose") == maps.point_key(
        ref["map"]["antigens"][0], "loose"
    )


def test_reference_passage_class_uses_the_af_rule() -> None:
    from af.report.compare.reference_ae import passage_class

    # ae's attribute says cell here (last step MDCK); af's rule says egg (an egg step occurred).
    assert passage_class({"P": "E2/E1MDCK2", "T": {"p": "c"}}) == "egg"
    assert passage_class({"P": "SIAT1 (2026-01-02)"}) == "cell"
    assert passage_class({"P": "E3", "R": "NYMC X-1"}) == "reassortant"
    assert passage_class({"P": ""}) is None


def _tree_doc(names: list[str], clades: list[str], sections: list[tuple[str, int, int]],
              series: tuple[str, str] = ("2025-01", "2025-12")) -> dict[str, Any]:  # fmt: skip
    leaves = [{"id": n, "name": n, "date": None, "clade": c, "shown": True, "order": i}
              for i, (n, c) in enumerate(zip(names, clades, strict=True))]  # fmt: skip
    return {"title": "t", "tree": {
        "subtype": "x", "leaves": leaves,
        "sections": [{"clade": c, "prefix": "", "first_leaf": names[a], "last_leaf": names[b],
                      "n_leaves": b - a + 1} for c, a, b in sections],
        "time_series": {"first": series[0], "last": series[1]}}}  # fmt: skip


def test_tree_figures_identical_and_reversed() -> None:
    names = [f"leaf{i}" for i in range(40)]
    clades = ["P"] * 20 + ["Q"] * 20
    ref = _tree_doc(names, clades, [("P", 0, 19), ("Q", 20, 39)])
    same = trees.compare_figures(ref, _tree_doc(names, clades, [("P", 0, 19), ("Q", 20, 39)]))
    assert same["jaccard"] == 1.0 and same["order_spearman"] == 1.0
    assert same["sections"]["min_jaccard"] == 1.0 and same["time_series"]["same"]
    flipped = _tree_doc(names[::-1], clades[::-1], [("Q", 0, 19), ("P", 20, 39)])
    assert trees.compare_figures(ref, flipped)["order_spearman"] == -1.0


def test_tree_section_shift_and_window_change_are_measured() -> None:
    names = [f"leaf{i}" for i in range(40)]
    clades = ["P"] * 20 + ["Q"] * 20
    ref = _tree_doc(names, clades, [("P", 0, 19)])
    moved = _tree_doc(names, clades, [("P", 10, 29)], series=("2025-02", "2026-01"))
    res = trees.compare_figures(ref, moved)
    assert res["sections"]["matched"]["P"]["jaccard"] == 10 / 30
    assert not res["time_series"]["same"]
    checks = {c["check"]: c["ok"] for c in tree_checks_for(res)}
    assert checks["time series same"] is False


def tree_checks_for(res: dict[str, Any]) -> list[dict[str, Any]]:
    from af.report.compare.run import TreeLimits, tree_checks

    return tree_checks(res, TreeLimits())


def test_runner_compares_tree_slots(tmp_path: Path) -> None:
    from af.report.compare.run import compare_report, markdown

    names = [f"leaf{i}" for i in range(12)]
    doc = _tree_doc(names, ["P"] * 12, [("P", 0, 11)])
    (tmp_path / "tree.x.report.i7.json").write_text(json.dumps({**doc, "kind": "tree"}))
    new_path = tmp_path / "new.json"
    new_path.write_text(json.dumps({**doc, "kind": "tree"}))
    record = {
        "report": "r",
        "built": "b",
        "figures": [{"slot": "tree/x/report", "i7": str(new_path)}],
    }
    limits = parse_config({"adoption": {"status": "final", "adopted_by": "a reviewer",
                                        "adopted": dt.date(2026, 9, 25)}},
                          Limits, base_dir=tmp_path)  # fmt: skip
    rows, failed = compare_report(record, tmp_path, limits, "loose")
    assert failed == 0 and rows[0]["status"] == "ok"
    assert "| tree/x/report | ok | 0 / 0 |" in markdown(record, rows, "l", "loose", limits.adoption)


def test_section_bounds_with_an_ae_hash_suffix_resolve() -> None:
    names = [f"leaf{i}" for i in range(20)]
    ref = _tree_doc(names, ["P"] * 20, [("P", 0, 9)])
    ref["tree"]["sections"][0]["first_leaf"] = "leaf0_OR_0A1B2C3D"
    res = trees.compare_figures(ref, _tree_doc(names, ["P"] * 20, [("P", 0, 9)]))
    assert res["sections"]["matched"]["P"]["jaccard"] == 1.0
    assert res["sections"]["unresolved"] == {"ref": [], "new": []}


def test_unresolved_section_is_reported_and_fails() -> None:
    names = [f"leaf{i}" for i in range(20)]
    new = _tree_doc(names, ["P"] * 20, [("P", 0, 9)])
    new["tree"]["sections"][0]["last_leaf"] = "not drawn"
    res = trees.compare_figures(_tree_doc(names, ["P"] * 20, [("P", 0, 9)]), new)
    assert len(res["sections"]["unresolved"]["new"]) == 1
    assert {c["check"]: c["ok"] for c in tree_checks_for(res)}["sections resolved"] is False


def test_reference_time_series_records_the_last_month_drawn() -> None:
    from af.report.compare.reference_ae import _drawn_months

    assert _drawn_months(("2024-10", "2026-10")) == {"first": "2024-10", "last": "2026-09"}
    assert _drawn_months(("2025-02", "2026-01")) == {"first": "2025-02", "last": "2025-12"}


def _geo_i7(month: str, counts: dict[str, int]) -> dict[str, Any]:
    return {
        "kind": "geo",
        "title": "g",
        "geo": {
            "subtype": "x",
            "month": month,
            "locations": [
                {"name": loc, "points": [{"color": "#AA0000", "count": n}]}
                for loc, n in counts.items()
            ],
        },
    }


def test_geo_month_comparison_and_checks() -> None:
    from af.report.compare.run import GeoLimits, geo_checks, geo_month

    ref = _geo_i7("2026-04", {"PLACE A": 3, "PLACE B": 1})
    same = geo_checks(geo_month(ref, _geo_i7("2026-04", {"PLACE A": 3, "PLACE B": 1})),
                      GeoLimits(location_frac_diff_max=0.05))  # fmt: skip
    assert all(c["ok"] in (True, None) for c in same)
    shifted = geo_month(ref, _geo_i7("2026-04", {"PLACE A": 3, "PLACE C": 1}))
    assert shifted["month"]["location"]["frac_diff"] == 2 / 8
    wrong_month = geo_checks(geo_month(ref, _geo_i7("2026-05", {"PLACE A": 3})), GeoLimits())
    assert {c["check"]: c["ok"] for c in wrong_month}["same month"] is False
