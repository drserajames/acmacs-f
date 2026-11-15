"""A map's short note in the frame (Q323): read from the round config, drawn under the title on
every window page, recorded in the I7, and kept clear of the points and labels as furniture."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from af.map.config import NoteConfig
from af.map.roundconfig import ConfigError, load_maps_config
from af.map.style import PointIn, Window
from af.map.viewport import Box, Frame, hidden_mask

from .test_finish import SCHEME, SINCE, synthetic_points
from .test_roundconfig import GOOD
from .test_roundconfig import write as write_config

NOTE = "No new data for this map since the last report (newest table 1 Jan 2026)"
CREATED = dt.datetime(2026, 11, 15, 12, 0, tzinfo=dt.UTC)
PROVENANCE = {"inputs": {"chain": {"name": "synthetic", "sha256": "0" * 64}}}


def with_note(table: str) -> str:
    return GOOD.replace(
        'layout_stand_in = "charts/example.ace"\n',
        f'layout_stand_in = "charts/example.ace"\nnote = {table}\n',
        1,
    )


GOOD_NOTE = f'{{ text = ["{NOTE}"], reason = "no new tables", decided = 2026-11-15 }}'


def test_a_map_note_is_read_with_its_reason_and_date(tmp_path: Path) -> None:
    config, _ = load_maps_config(write_config(tmp_path, with_note(GOOD_NOTE)))
    (m,) = config.maps
    assert m.note == NoteConfig((NOTE,), "no new tables", dt.date(2026, 11, 15))
    assert m.note.record() == {"text": [NOTE], "reason": "no new tables", "decided": "2026-11-15"}
    plain, _ = load_maps_config(write_config(tmp_path, GOOD))
    assert plain.maps[0].note is None


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ('{ text = [], reason = "r", decided = 2026-11-15 }', "1 to 2"),
        ('{ text = ["a", "b", "c"], reason = "r", decided = 2026-11-15 }', "1 to 2"),
        ('{ text = ["a", " "], reason = "r", decided = 2026-11-15 }', "empty line"),
        (f'{{ text = ["{"x" * 121}"], reason = "r", decided = 2026-11-15 }}', "121 characters"),
        ('{ text = ["a"], reason = " ", decided = 2026-11-15 }', "no reason"),
        ('{ text = ["a"], decided = 2026-11-15 }', "no reason"),
        ('{ text = ["a"], reason = "r" }', "no decided"),
        ('{ text = ["a"], reason = "r", decided = 2026-11-15, size = 9 }', "unknown key"),
        ('{ text = "a", reason = "r", decided = 2026-11-15 }', "list of strings"),
        ('"a"', "expected a table"),
    ],
)
def test_a_bad_note_is_refused(tmp_path: Path, table: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_maps_config(write_config(tmp_path, with_note(table)))


def crowded_points() -> list[PointIn]:
    """The synthetic map, plus recent antigens spread over the whole square, so that wherever
    the frame falls some point would sit where the note prints unless the note is furniture."""
    pts = synthetic_points()
    rng = np.random.default_rng(7)
    for k in range(60):
        x, y = rng.uniform(-6, 9), rng.uniform(-6, 6)
        pts.append(
            PointIn(
                f"crowd{k}", f"crowd{k}", "antigen", (float(x), float(y)), frozenset({"B"}),
                date=dt.date(2025, 6, 1), sequenced=True,
            )
        )  # fmt: skip
    return pts


def finish(tmp_path: Path, name: str, points: list[PointIn], **kw: Any) -> Any:
    from af.map.finish import finish_map

    return finish_map(
        points,
        chart="synthetic",
        scheme=SCHEME,
        window=Window("all", None),
        title="Synthetic by group",
        frame_size=14.0,
        must_show_since=SINCE,
        vaccine_labels={"ag100": "Te/25-cell", "ag110": "Te/25-egg"},
        out_pdf=tmp_path / name / "figure.pdf",
        created=CREATED,
        provenance=PROVENANCE,
        **kw,
    )


def _overlap(a: tuple[float, float, float, float], b: Box) -> bool:
    return a[0] < b.right and a[2] > b.left and a[1] < b.bottom and a[3] > b.top


def test_the_note_is_drawn_recorded_and_kept_clear(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from af.map.finish import _page
    from af.map.render import DEFAULT_LOOK, note_box

    points = crowded_points()
    box = note_box([NOTE], DEFAULT_LOOK)
    assert 0.0 < box.top < 0.115 < box.bottom < 0.2 and 0.3 < box.right < 0.95
    # without the note, the frame this map takes puts some shown point where the note prints:
    # the test below would otherwise prove nothing
    bare = finish(tmp_path, "bare", points)
    page = bare.frame.frame.page(bare.scene.xy())
    shown = np.array([p.shown for p in bare.scene.points])
    assert hidden_mask(page[shown], (box,)).any()

    result = finish(tmp_path, "note", points, note=[NOTE])
    doc = json.loads(result.i7.read_text())
    assert doc["map"]["caption"] == [NOTE]
    assert result.recent_hidden == 0
    # The note is furniture, as the title is: no recent antigen's marker reaches into it, the
    # points it does cover (old ones, where the square is crowded) are counted as hidden, and
    # no label is placed on it.
    vaccines = {"ag100": "Te/25-cell", "ag110": "Te/25-egg"}
    pg = _page(points, SCHEME, Window("all", None), "Synthetic by group", SINCE, vaccines,
               DEFAULT_LOOK, None, (NOTE,))  # fmt: skip
    assert pg.furniture[-1] == box
    page = result.frame.frame.page(result.scene.xy())
    shown = np.array([p.shown for p in result.scene.points])
    under = hidden_mask(page, (box,), pg.radii) & shown
    recent = np.array([p.date is not None and p.date >= SINCE for p in result.scene.points])
    assert not (under & recent).any()
    assert result.frame.hidden["all antigens"] >= int(under.sum())
    assert result.labels and not any(_overlap(lab.box, box) for lab in result.labels.values())
    if shutil.which("pdftotext"):
        text = subprocess.run(
            ["pdftotext", str(result.pdf), "-"], capture_output=True, text=True, check=True
        ).stdout
        assert NOTE in " ".join(text.split())


def test_without_a_note_the_page_is_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No note: the same furniture as ever (legend, title), and the same figure bytes as a page
    given an empty note, so a round that sets no note draws exactly what it drew before."""
    pytest.importorskip("matplotlib")
    # the PDF's CreationDate is the wall clock unless pinned; two drawings a second apart differ
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1763164800")
    from af.map.finish import _page
    from af.map.render import DEFAULT_LOOK, legend_box, title_box

    pg = _page(synthetic_points(), SCHEME, Window("all", None), "t", SINCE, {}, DEFAULT_LOOK, None)
    assert pg.furniture == (legend_box(pg.scene, DEFAULT_LOOK), title_box(pg.scene, DEFAULT_LOOK))
    a = finish(tmp_path, "a", synthetic_points())
    b = finish(tmp_path, "b", synthetic_points(), note=())
    assert (
        hashlib.sha256(a.pdf.read_bytes()).digest() == hashlib.sha256(b.pdf.read_bytes()).digest()
    )
    doc_a, doc_b = json.loads(a.i7.read_text()), json.loads(b.i7.read_text())
    assert "caption" not in doc_a["map"]
    doc_a["figure"].pop("path", None), doc_b["figure"].pop("path", None)
    assert doc_a == doc_b


def test_a_note_and_a_caption_cannot_share_the_place(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="give one"):
        finish(tmp_path, "both", synthetic_points(), note=[NOTE], caption=["a serum"])


def test_a_note_needs_a_square_frame(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="square frame"):
        finish(tmp_path, "tall", synthetic_points(), note=[NOTE], frame=Frame(-7, 7, 14.0, 16.0))


def test_every_window_shares_a_frame_that_keeps_clear_of_the_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    import af.map.finish as finish_module
    from af.map.finish import shared_frame

    measured: list[tuple[str, ...]] = []
    real_note_box = finish_module.note_box

    def counting_note_box(lines: Any, look: Any) -> Box:
        measured.append(tuple(lines))
        return real_note_box(lines, look)

    monkeypatch.setattr(finish_module, "note_box", counting_note_box)
    windows = [(Window("all", None), "Synthetic"), (Window("12m", SINCE), "Synthetic (since)")]
    common: dict[str, Any] = dict(
        scheme=SCHEME, frame_size=14.0, must_show_since=SINCE, vaccine_labels={}
    )
    points = crowded_points()
    with_n = shared_frame(points, windows=windows, note=[NOTE], **common)
    assert measured == [(NOTE,)] * len(windows)  # the note is furniture on every window's page
    measured.clear()
    shared_frame(points, windows=windows, **common)
    assert measured == []  # and no note, no note box
    for window, title in windows:
        from af.map.finish import finish_map

        result = finish_map(
            points, chart="synthetic", window=window, title=title,
            out_pdf=tmp_path / window.name / "figure.pdf", created=CREATED,
            provenance=PROVENANCE, frame=with_n, note=[NOTE], **common,
        )  # fmt: skip
        assert result.frame.frame == with_n and result.recent_hidden == 0
        assert json.loads(result.i7.read_text())["map"]["caption"] == [NOTE]


def _ink(
    scene: Any, frame: Any, out: Path, caption: list[str], size: float | None, look: Any
) -> Any:
    """The pixels (600 dpi) a caption adds to a page: the same scene and frame, no labels, drawn
    with and without it, so the difference is the caption's ink and nothing else."""
    from matplotlib.image import imread

    from af.map.render import draw_pdf

    rasters = []
    for tag, lines in (("with", caption), ("without", [])):
        pdf = out / f"{tag}.pdf"
        draw_pdf(scene, frame, {}, pdf, look, caption=lines, caption_size=size)
        subprocess.run(
            ["pdftoppm", "-r", "600", "-png", "-singlefile", str(pdf), str(out / tag)], check=True
        )
        png = out / f"{tag}.png"
        assert png.stat().st_size > 0
        rasters.append(imread(png)[..., :3])
    ink = np.abs(rasters[0] - rasters[1]).max(axis=2) > 0
    assert ink.any()
    return ink


def test_the_note_prints_at_the_legend_size_inside_its_box(tmp_path: Path) -> None:
    """Q326: the note is printed at the legend entries' text size, and its furniture box,
    measured at that size, holds all of its ink. A bigger legend means a bigger note (one fact,
    not two); the circle addenda's caption keeps its own small size."""
    pytest.importorskip("matplotlib")
    if not shutil.which("pdftoppm"):
        pytest.skip("pdftoppm not installed")
    import dataclasses

    from af.map.render import DEFAULT_LOOK, note_box, note_size

    assert note_size(DEFAULT_LOOK) == DEFAULT_LOOK.legend_text_size != DEFAULT_LOOK.caption_size
    big = dataclasses.replace(DEFAULT_LOOK, legend_text_size=16.0)
    base = finish(tmp_path, "base", synthetic_points())
    heights: dict[str, int] = {}
    for name, look in (("default", DEFAULT_LOOK), ("big", big)):
        (tmp_path / name).mkdir(exist_ok=True)
        ink = _ink(base.scene, base.frame.frame, tmp_path / name, [NOTE], note_size(look), look)
        rows, cols = np.nonzero(ink)
        side = ink.shape[1]  # a square page
        box = note_box([NOTE], look)
        assert box.top * side <= rows.min() and rows.max() < box.bottom * side
        assert box.left * side <= cols.min() and cols.max() < box.right * side
        heights[name] = int(rows.max() - rows.min() + 1)
    assert heights["big"] / heights["default"] == pytest.approx(16.0 / 12.23, rel=0.03)
    (tmp_path / "caption").mkdir()
    ink = _ink(base.scene, base.frame.frame, tmp_path / "caption", [NOTE], None, DEFAULT_LOOK)
    rows, _ = np.nonzero(ink)
    small = (rows.max() - rows.min() + 1) / heights["default"]
    assert small == pytest.approx(7.0 / 12.23, rel=0.03)


def test_a_note_too_wide_for_the_page_is_refused() -> None:
    from af.map.render import DEFAULT_LOOK, note_box

    with pytest.raises(ValueError, match="too wide"):
        note_box(["W" * 120], DEFAULT_LOOK)
