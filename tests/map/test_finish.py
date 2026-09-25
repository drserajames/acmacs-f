"""Style + frame + labels + PDF + I7 on a synthetic map (needs matplotlib)."""

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from af.map.labels import label_text, place_labels
from af.map.style import ColourRow, ColourScheme, PointIn, Window, style_points
from af.map.viewport import Box

pytest.importorskip("matplotlib")

from af.map.finish import finish_map  # noqa: E402

SCHEME = ColourScheme(
    "test-scheme",
    (
        ColourRow("group A", "#1f77b4", frozenset({"A"})),
        ColourRow("group A sub", "#ff7f0e", frozenset({"A", "a1"})),
        ColourRow("group B", "#2ca02c", frozenset({"B"})),
    ),
)
SINCE = dt.date(2025, 9, 1)


def synthetic_points(seed: int = 2) -> list[PointIn]:
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(120):
        group = ["A"] if i % 3 else ["A", "a1"]
        if i >= 80:
            group = ["B"]
        xy = rng.normal(size=2) * 2 + ([4.0, 0.0] if i >= 80 else [0.0, 0.0])
        date = dt.date(2024, 1, 1) + dt.timedelta(days=int(i * 6))
        pts.append(
            PointIn(
                id=f"ag{i}",
                name=f"ag{i}",
                kind="antigen",
                xy=(float(xy[0]), float(xy[1])),
                labels=frozenset(group),
                date=date,
                passage_class="egg" if i % 7 == 0 else "cell",
                sequenced=True,
            )
        )
    pts.append(PointIn("ag-noxy", "ag-noxy", "antigen", None, frozenset({"A"})))
    # sequenced, but carries only a label the scheme has no row for: must be counted
    pts.append(PointIn("ag-new", "ag-new", "antigen", (0.5, 0.5), frozenset({"C"}), sequenced=True))
    pts.append(PointIn("ag-noseq", "ag-noseq", "antigen", (0.4, 0.4)))
    pts.append(
        PointIn("ag-hidden", "ag-hidden", "antigen", (0.0, 0.0), frozenset({"A"}), hide="rule-1")
    )
    for j in range(6):
        pts.append(PointIn(f"sr{j}", f"sr{j}", "serum", (float(j) - 2, 1.0), serum_id=f"S{j}"))
    return pts


def test_style_paints_last_matching_row_and_greys_old() -> None:
    scene = style_points(synthetic_points(), SCHEME, Window("12m", SINCE), title="t")
    by_id = {p.id: p for p in scene.points}
    assert by_id["ag0"].legend == "group A sub"  # last matching row wins
    assert by_id["ag1"].legend == "group A"
    assert by_id["ag0"].greyed  # 2024-01-01 is before the window
    assert not by_id["ag119"].greyed
    assert by_id["ag-noxy"].hidden_reason == "no_coordinates"
    assert by_id["ag-hidden"].hidden_reason == "override:rule-1"
    assert [t for t, _, _ in scene.legend] == ["group B", "group A sub", "group A"]  # last first
    assert sum(n for _, _, n in scene.legend) == 120  # painted: shown antigens, each counted once
    assert scene.sequenced_unpainted == 1
    matched = style_points(
        synthetic_points(), SCHEME, Window("12m", SINCE), title="t", legend_counts="matched"
    )
    assert {t: n for t, _, n in matched.legend}["group A"] == 80  # includes the painted-over sub


def strain(prefix: str, place: str) -> str:
    """Invented strain names, generated rather than written out (see tools/WHO-DATA-GATE.md)."""
    return "/".join((prefix, place, "5", "2021"))


def test_label_text_rule() -> None:
    assert label_text(strain("A(H3N2)", "SOMEPLACE"), "cell", "", {}) == "So/21-cell"
    assert label_text(strain("B", "TWO WORDS"), "egg", "", {}) == "TW/21-egg"
    assert label_text(strain("A(H1N1)", "SOMEPLACE"), "egg", "", {"SOMEPLACE": "SP"}) == "SP/21-egg"
    assert label_text(strain("A(H3N2)", "SOMEPLACE"), "reassortant", "XR-1", {}) == "So/21-XR-1"


def test_labels_avoid_furniture_and_each_other() -> None:
    anchors = np.array([[0.5, 0.5], [0.51, 0.5], [0.1, 0.64]])  # last one just above the legend
    legend = Box("legend", 0.0, 0.7, 0.3, 1.0)
    placed = place_labels(anchors, ["AAA/21-cell"] * 3, np.zeros((0, 2)), [legend])
    assert all(p.overlaps == 0 for p in placed)


def test_finish_map_writes_pdf_and_matching_i7(tmp_path: Path) -> None:
    out = tmp_path / "map" / "figure.pdf"
    result = finish_map(
        synthetic_points(),
        chart="synthetic",
        scheme=SCHEME,
        window=Window("12m", SINCE),
        title="Synthetic by group",
        frame_size=14.0,
        must_show_since=SINCE,
        vaccine_labels={"ag100": "Te/25-cell"},
        out_pdf=out,
        created=dt.datetime(2026, 9, 25, 12, 0, tzinfo=dt.UTC),
        provenance={"inputs": {"chain": {"name": "synthetic", "sha256": "0" * 64}}},
        flags=["189R move refused: 4.499 u > cap 4.0 u"],
    )
    assert result.recent_hidden == 0
    doc = json.loads(result.i7.read_text())
    assert doc["figure"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert doc["kind"] == "map" and doc["i7_version"] == 1
    assert len(doc["map"]["antigens"]) == 124 and len(doc["map"]["sera"]) == 6  # complete list
    assert doc["map"]["colour_coverage"] == {
        "shown_antigens": 122,
        "sequenced": 121,
        "painted": 120,
        "sequenced_unpainted": 1,
        "unsequenced": 1,
        "vaccines_recoloured_from_cell": 0,
        "vaccines_without_cell_preparation": [],  # ag100 IS the cell preparation
    }
    hidden = [a for a in doc["map"]["antigens"] if not a["shown"]]
    assert {a["hidden_reason"] for a in hidden} == {"no_coordinates", "override:rule-1"}
    assert all(a["colour"] for a in doc["map"]["antigens"])  # never null
    assert {s["colour"] for s in doc["map"]["sera"]} == {"transparent"}
    by_id = {a["id"]: a for a in doc["map"]["antigens"]}
    assert not by_id["ag-new"]["greyed"]  # unpainted: base grey, not the window rule
    assert by_id["ag0"]["greyed"] and by_id["ag0"]["colour"] == "#c8c8c8"
    assert doc["map"]["flags"] == ["189R move refused: 4.499 u > cap 4.0 u"]
    vac = [a for a in doc["map"]["antigens"] if a["vaccine"]]
    assert vac[0]["label"] == "Te/25-cell" and not vac[0]["greyed"]


def test_i7_refuses_naive_timestamp(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="time zone"):
        finish_map(
            synthetic_points(),
            chart="synthetic",
            scheme=SCHEME,
            window=Window("all", None),
            title="t",
            frame_size=14.0,
            must_show_since=SINCE,
            vaccine_labels={},
            out_pdf=tmp_path / "f.pdf",
            created=dt.datetime(2026, 9, 25, 12, 0),
            provenance={"inputs": {"chain": {"sha256": "0" * 64}}},
        )


def test_vaccines_to_label_most_specific_list_wins() -> None:
    from af.map.labels import LabelRule, vaccines_to_label
    from af.map.vaccines import VaccineMark, VaccineRow

    names = ["/".join((p, "1", "2020")) for p in ("ALPHA", "BETA", "GAMMA")]
    marks = [
        VaccineMark(VaccineRow(n, None, "2020"), "cell", i, 1, "rule") for i, n in enumerate(names)
    ]
    assert vaccines_to_label(marks, subtype_list=None, lab_list=None) == marks
    sub = [LabelRule(names[0], "any"), LabelRule(names[1], "cell")]
    assert [m.antigen for m in vaccines_to_label(marks, subtype_list=sub, lab_list=None)] == [0, 1]
    lab = [LabelRule(names[2], "any")]
    assert [m.antigen for m in vaccines_to_label(marks, subtype_list=sub, lab_list=lab)] == [2]
    with pytest.raises(ValueError, match="matches no marked vaccine"):
        vaccines_to_label(marks, subtype_list=[LabelRule(names[0], "egg")], lab_list=None)


def test_provenance_needs_hashes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sha256"):
        finish_map(
            synthetic_points(),
            chart="synthetic",
            scheme=SCHEME,
            window=Window("all", None),
            title="t",
            frame_size=14.0,
            must_show_since=SINCE,
            vaccine_labels={},
            out_pdf=tmp_path / "f.pdf",
            created=dt.datetime(2026, 9, 25, 12, 0, tzinfo=dt.UTC),
            provenance={"inputs": {"chain": {"name": "no hash"}}},
        )


def test_vaccine_takes_its_cell_preparations_colour() -> None:
    """Sarah, 25 Sep 2026: a vaccine is coloured by its cell preparation; the egg one follows it
    even when egg-adaptation puts it in another clade."""
    vac = strain("A(H3N2)", "VACVILLE")
    lone = strain("A(H3N2)", "EGGONLY")
    pts = [
        PointIn("cell", vac, "antigen", (0.0, 0.0), frozenset({"A"}), passage_class="cell"),
        PointIn("egg", vac, "antigen", (1.0, 0.0), frozenset({"B"}), passage_class="egg"),
        PointIn("reass", vac, "antigen", (2.0, 0.0), frozenset({"B"}), passage_class="reassortant"),
        PointIn("solo", lone, "antigen", (3.0, 0.0), frozenset({"B"}), passage_class="egg"),
        PointIn("other", strain("A(H3N2)", "BYSTANDER"), "antigen", (4.0, 0.0), frozenset({"B"})),
    ]
    labels = {"cell": "Va/20-cell", "egg": "Va/20-egg", "reass": "Va/20-X", "solo": "Eg/20-egg"}
    scene = style_points(pts, SCHEME, Window("all", None), title="t", vaccines=labels)
    by = {p.id: p for p in scene.points}
    assert by["cell"].colour == SCHEME.rows[0].colour  # group A
    assert by["egg"].colour == by["cell"].colour  # follows its cell preparation
    assert by["reass"].colour == by["cell"].colour
    assert by["solo"].colour == SCHEME.rows[2].colour  # no cell counterpart: keeps group B
    assert by["other"].colour == SCHEME.rows[2].colour  # not a vaccine: untouched
    assert scene.vaccines_recoloured == 2
    assert scene.vaccines_without_cell == ["solo"]
    # the legend counts each shown antigen under the colour it is DRAWN in
    counts = {t: n for t, _, n in scene.legend}
    assert counts["group A"] == 3 and counts["group B"] == 2
