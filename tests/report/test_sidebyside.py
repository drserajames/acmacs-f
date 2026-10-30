"""Side-by-side documents and Procrustes arrows, on synthetic maps (invented names only)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from af.report import build
from af.report.compare import arrows, maps, pixels, sidebyside
from af.util.artefacts import sha256_path

from .test_compare import _cloud, _rotate

NAME = "A(H3N2)/TESTTOWN/{}/2025"  # invented


def label(i: int) -> str:
    return NAME.format(i) + " cell"


def test_the_fitted_points_are_what_the_distances_measure() -> None:
    a = _cloud(40, 3)
    b = [(x - 4, y + 1) for x, y in _rotate(a, 25)]
    b[0] = (b[0][0] + 2.0, b[0][1])  # one point moved
    for moving in (b, [(x, -y) for x, y in b]):  # and through the mirror
        fit = maps.procrustes(a, moving)
        assert len(fit.fitted) == len(a)
        assert fit.distances == pytest.approx(
            [math.dist(p, q) for p, q in zip(a, fit.fitted, strict=True)], abs=1e-9
        )
    assert maps.procrustes(a, b).distances[0] > 1.5


def _doc(points: list[tuple[float, float]], sera: list[tuple[float, float]]) -> dict[str, Any]:
    def point(i: int, xy: tuple[float, float], **extra: Any) -> dict[str, Any]:
        return {"id": f"ag{i}", "name": NAME.format(i), "passage_class": "cell",
                "date": "2025-03-01", "xy": list(xy), "shown": True, "in_viewport": True,
                "clade": "X", "colour": "#000000", "greyed": False, **extra}  # fmt: skip

    antigens = [point(i, p, vaccine=i == 1) for i, p in enumerate(points)]
    serum_points = [
        {"id": f"sr{i}", "name": NAME.format(900 + i), "serum_id": f"F{i}",
         "xy": list(p), "shown": True, "in_viewport": True, "clade": None, "colour": None,
         "greyed": False}
        for i, p in enumerate(sera)
    ]  # fmt: skip
    return {"kind": "map", "title": "synthetic",
            "map": {"y_axis": "up", "viewport": [-8.0, 8.0, 16.0, 16.0],
                    "antigens": antigens, "sera": serum_points}}  # fmt: skip


def _pair(moved: float = 2.0) -> tuple[dict[str, Any], dict[str, Any]]:
    new_ag, new_sr = _cloud(30, 4, 2.0), _cloud(5, 5, 2.0)
    ref_ag = [(x + 1, y - 1) for x, y in _rotate(new_ag, 40)]
    ref_sr = [(x + 1, y - 1) for x, y in _rotate(new_sr, 40)]
    ref_ag[3] = (ref_ag[3][0] + moved, ref_ag[3][1])
    return _doc(ref_ag, ref_sr), _doc(new_ag, new_sr)


def test_arrows_fit_the_reference_onto_the_new_map_as_the_comparison_does() -> None:
    ref, new = _pair()
    a = arrows.measure(ref, new, "map/x/all")
    assert len(a.moves) == 35 and a.antigens == 30
    assert a.fit.rmsd == pytest.approx(maps.compare(ref, new, "identity")["procrustes"]["rmsd"])
    # the arrows are drawn in the new map's coordinates: heads where the new map draws them
    drawn = {f"{p['name']} cell": tuple(p["xy"]) for p in new["map"]["antigens"]}
    assert all(m.new_xy == drawn[m.label] for m in a.moves if m.kind == "antigen")
    far = [m for m in a.moves if m.distance > arrows.FAR]
    assert [m.label for m in far] == [label(3)]
    assert a.count(arrows.FAR) == 1 and a.count(arrows.FAR, "serum") == 0
    row = arrows.summary(a)
    assert row["largest"][0]["label"] == label(3)
    assert [v["label"] for v in row["vaccines"]] == [label(1)]
    assert row["orientation"].startswith("bulk orientation")


def test_arrows_refuse_maps_with_different_y_axes() -> None:
    ref, new = _pair()
    ref["map"]["y_axis"] = "down"
    with pytest.raises(ValueError, match="y axes differ"):
        arrows.measure(ref, new, "map/x/all")


def test_the_overlay_draws_only_moves_over_the_threshold(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    ref, new = _pair(moved=0.2)  # under the threshold: nothing to draw
    a = arrows.measure(ref, new, "map/x/all")
    assert arrows.overlay(a, new["map"]["viewport"], (400.0, 400.0), tmp_path / "o.pdf") == 0
    ref, new = _pair(moved=0.6)
    a = arrows.measure(ref, new, "map/x/all")
    assert arrows.overlay(a, new["map"]["viewport"], (400.0, 400.0), tmp_path / "o.pdf") == 1
    assert (tmp_path / "o.pdf").stat().st_size > 0


def test_pixel_diff_pads_the_smaller_page_and_counts_pixels() -> None:
    a = np.full((10, 20, 3), 255, dtype=np.uint8)
    b = np.full((12, 20, 3), 255, dtype=np.uint8)
    b[0, 0] = (255 - pixels.THRESHOLD, 255, 255)  # at the threshold: not different
    b[1, 1] = (0, 0, 0)
    b[11, 5] = (0, 255, 255)  # in the padding a lacks
    d = pixels.compare_arrays(a, b)
    assert (d.differing, d.canvas) == (2, 240)
    assert d.size_a == (20, 10) and d.size_b == (20, 12)
    assert "padded white" in d.text() and "2 pixels differ (0.83%" in d.text()


def test_ppm_reader_skips_comments(tmp_path: Path) -> None:
    raw = bytes([10, 20, 30, 40, 50, 60])
    (tmp_path / "x.ppm").write_bytes(b"P6\n# made by a test\n2 1\n255\n" + raw)
    assert pixels.read_ppm(tmp_path / "x.ppm").tolist() == [[[10, 20, 30], [40, 50, 60]]]


def _record(root: Path, name: str, docs: dict[str, dict[str, Any]], commit: str) -> Path:
    """A round folder with built map figures and a build record naming them."""
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    figures = []
    for slot, doc in docs.items():
        d = root / name / "figures" / slot / "v1"
        d.mkdir(parents=True)
        (d / "figure.i7.json").write_text(json.dumps(doc))
        fig = plt.figure(figsize=(4, 4), dpi=72)
        xy = np.array([p["xy"] for p in doc["map"]["antigens"]])
        fig.add_axes((0, 0, 1, 1)).scatter(xy[:, 0], xy[:, 1], s=20)
        fig.savefig(d / "figure.pdf")
        plt.close(fig)
        figures.append({"slot": slot, "i7": str(d / "figure.i7.json"),
                        "i7_sha256": sha256_path(d / "figure.i7.json"),
                        "pdf_sha256": sha256_path(d / "figure.pdf")})  # fmt: skip
    report = root / name / "report"
    report.mkdir()
    fig = plt.figure(figsize=(8.27, 11.69), dpi=72)
    fig.text(0.1, 0.9, name)
    fig.savefig(report / "r.pdf")
    plt.close(fig)
    record = {"report": "r", "af": {"commit": commit}, "figures": figures,
              "output": {"pdf": "r.pdf", "sha256": sha256_path(report / "r.pdf")}}  # fmt: skip
    (report / "r.build.json").write_text(json.dumps(record))
    return report / "r.build.json"


def test_map_slots_pair_by_name_and_one_sided_slots_are_listed(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    ref, new = _pair()
    a = _record(tmp_path, "r", {"map/x/all": new, "map/y/all": new}, "c0")
    b = _record(tmp_path, "r-older", {"map/x/all": ref, "map/z/all": ref}, "c0")
    pairs, one_sided = sidebyside.map_pairs(json.loads(a.read_text()), json.loads(b.read_text()))
    assert [p.slot for p in pairs] == ["map/x/all"]
    assert one_sided == ["map/y/all (new only)", "map/z/all (reference only)"]
    assert sidebyside.ref_name(a, b, "r") == "older"
    (tmp_path / "r" / "figures" / "map/x/all" / "v1" / "figure.pdf").write_bytes(b"%PDF changed")
    with pytest.raises(sidebyside.SideBySideError, match="not the file its build recorded"):
        sidebyside.map_pairs(json.loads(a.read_text()), json.loads(b.read_text()))


def test_builds_by_different_af_are_refused_by_name(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    ref, new = _pair()
    a = _record(tmp_path, "r", {"map/x/all": new}, "c0" * 6)
    b = _record(tmp_path, "r-older", {"map/x/all": ref}, "c1" * 6)
    with pytest.raises(sidebyside.SideBySideError, match="r report on c0c0c0c0c0c0, r-older"):
        sidebyside.main([str(a), str(b)])


@pytest.mark.tool(build.LATEX)
@pytest.mark.tool(pixels.PDFTOPPM)
def test_one_command_writes_both_documents_beside_the_report(tmp_path: Path) -> None:
    random.seed(0)
    ref, new = _pair()
    a = _record(tmp_path, "r", {"map/x/all": new}, "c0")
    b = _record(tmp_path, "r-older", {"map/x/all": ref}, "c0")
    assert sidebyside.main([str(a), str(b)]) == 0
    report = tmp_path / "r" / "report"
    pages = {
        name: len(sidebyside.page_sizes(report / f"r-{name}-vs-older.pdf"))
        for name in ("procrustes", "sbs")
    }
    assert pages == {"procrustes": 4, "sbs": 2}  # a cover, then 3 pages per map / 1 per page
    data = json.loads((report / "r-procrustes-vs-older.json").read_text())
    assert data["maps"][0]["moved_gt_1"] == {"antigens": 1, "sera": 0}
    assert data["pages"][0]["differing"] > 0  # the two report pages print different names


@pytest.mark.tool(build.LATEX)
@pytest.mark.tool(pixels.PDFTOPPM)
def test_side_by_side_pages_keep_their_own_size(tmp_path: Path) -> None:
    """Sarah, 30 Oct 2026: "Keep the original page sizes in side by side": a page pair is never
    scaled to fit, so its sheet is at least the two pages side by side (or stacked) plus margins."""
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    from af.report.compare.pages import GAP, MARGIN

    def pdf(name: str, w: float, h: float) -> Path:
        fig = plt.figure(figsize=(w / 72, h / 72), dpi=72)
        fig.text(0.1, 0.5, name)
        fig.savefig(tmp_path / f"{name}.pdf")
        plt.close(fig)
        return tmp_path / f"{name}.pdf"

    for (aw, ah), (bw, bh), stacked in (((300, 400), (600, 800), False),
                                        ((400, 300), (800, 600), True)):  # fmt: skip
        new, ref = pdf("new", aw, ah), pdf("ref", bw, bh)
        out = tmp_path / "sbs.pdf"
        rows = sidebyside.write_sbs(new, ref, ("new", "ref"), out)
        w, h = sidebyside.page_sizes(out)[1]  # page 1 is the cover
        slack = 0.05  # TeX rounds dimensions to its scaled points
        if stacked:
            assert w + slack >= max(aw, bw) + 2 * MARGIN
            assert h + slack >= ah + bh + GAP + 2 * MARGIN
        else:
            assert w + slack >= aw + bw + GAP + 2 * MARGIN
            assert h + slack >= max(ah, bh) + 2 * MARGIN
        assert rows[0]["sizes"][0] != rows[0]["sizes"][1]
        d = pixels.compare_arrays(pixels.rasterise(new, 1), pixels.rasterise(ref, 1))
        for size in (d.size_a, d.size_b):  # the page's pixel line names both sizes
            assert f"{size[0]} x {size[1]} px" in d.text().replace(" and ", " px and ")
        assert "padded white" in d.text()


@pytest.mark.tool(build.LATEX)
@pytest.mark.tool(pixels.PDFTOPPM)
def test_windows_with_identical_arrows_share_one_arrows_page(tmp_path: Path) -> None:
    """A map's windows share one layout: identical arrows get one arrows page and one numbers
    page, naming the windows; a map whose windows differ at all keeps a page per window."""
    ref, new = _pair()
    a = _record(tmp_path, "r", {"map/x/all": new, "map/x/recent": new}, "c0")
    b = _record(tmp_path, "r-older", {"map/x/all": ref, "map/x/recent": ref}, "c0")
    out = tmp_path / "out"
    assert sidebyside.main([str(a), str(b), "--out", str(out)]) == 0
    # a cover, the two windows' figures, then one arrows page and one numbers page
    assert len(sidebyside.page_sizes(out / "r-procrustes-vs-older.pdf")) == 5
    moved = json.loads(json.dumps(new))
    moved["map"]["antigens"][5]["xy"][0] += 0.5  # the recent window's layout differs
    c = _record(tmp_path / "c", "r", {"map/x/all": new, "map/x/recent": moved}, "c0")
    d = _record(tmp_path / "c", "r-older", {"map/x/all": ref, "map/x/recent": ref}, "c0")
    assert sidebyside.main([str(c), str(d), "--out", str(out)]) == 0
    assert len(sidebyside.page_sizes(out / "r-procrustes-vs-older.pdf")) == 7
