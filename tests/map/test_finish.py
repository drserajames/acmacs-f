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
            )
        )
    pts.append(PointIn("ag-noxy", "ag-noxy", "antigen", None, frozenset({"A"})))
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
    counts = {t: n for t, _, n in scene.legend}
    assert sum(counts.values()) == 120  # shown antigens only; hidden/no-xy excluded


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
        vaccine_labels={"ag100": "Te/25-cell"},
        out_pdf=out,
        created=dt.datetime(2026, 9, 25, 12, 0, tzinfo=dt.UTC),
        provenance={"inputs": {"chain": "synthetic"}},
    )
    assert result.recent_hidden == 0
    doc = json.loads(result.i7.read_text())
    assert doc["figure"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert doc["kind"] == "map" and doc["i7_version"] == 1
    assert len(doc["map"]["antigens"]) == 122 and len(doc["map"]["sera"]) == 6  # complete list
    hidden = [a for a in doc["map"]["antigens"] if not a["shown"]]
    assert {a["hidden_reason"] for a in hidden} == {"no_coordinates", "override:rule-1"}
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
            vaccine_labels={},
            out_pdf=tmp_path / "f.pdf",
            created=dt.datetime(2026, 9, 25, 12, 0),
            provenance={},
        )
