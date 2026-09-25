"""Finish one map window: style, frame, label, draw, describe. The order is the point.

Styling comes first because the legend's size (and so the space the frame must leave free)
depends on the colour scheme. The frame comes before labels, because labels are placed on the
page. The PDF comes before the I7, because the I7 carries the PDF's hash.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from af.map.labels import Placed, place_labels
from af.map.render import (
    DEFAULT_LOOK,
    Look,
    check_provenance_inputs,
    draw_pdf,
    i7_document,
    legend_box,
    recent_hidden,
    title_box,
    write_i7,
)
from af.map.style import ColourScheme, PointIn, Scene, Window, style_points
from af.map.viewport import Frame, FrameChoice, Priority, choose_frame


@dataclass(frozen=True)
class FinishedMap:
    scene: Scene
    frame: FrameChoice
    labels: dict[str, Placed]
    pdf: Path
    i7: Path
    recent_hidden: int


def finish_map(
    points: Sequence[PointIn],
    *,
    chart: str,
    scheme: ColourScheme,
    window: Window,
    title: str,
    frame_size: float,
    must_show_since: dt.date,
    vaccine_labels: dict[str, str],
    out_pdf: Path,
    created: dt.datetime,
    provenance: dict[str, Any],
    orientation: dict[str, object] | None = None,
    look: Look = DEFAULT_LOOK,
) -> FinishedMap:
    """Produce ``out_pdf`` and its I7 for one chart and window.

    ``must_show_since``: antigens isolated on or after this date must all be visible (not off the
    page, not under the legend or title) in *every* window, including "all", where nothing is
    greyed and older antigens may fall off the edge if the frame is too small for them. Raises
    ``FrameError`` rather than writing a map that hides one. Vaccines come next: shown if at all
    possible, but an old vaccine far from today's viruses may not fit (the I7 records
    ``in_viewport`` for each), then antigens in the window, sera, and everything else."""
    # fail before drawing anything
    if created.tzinfo is None:
        raise ValueError("I7 'created' needs a time zone")
    check_provenance_inputs(provenance.get("inputs"))
    scene = style_points(points, scheme, window, title=title, vaccines=vaccine_labels)
    furniture = (legend_box(scene, look), title_box(scene, look))
    xy = scene.xy()
    shown = np.array([p.shown for p in scene.points])
    antigen = np.array([p.kind == "antigen" for p in scene.points])
    recent = (
        shown
        & antigen
        & np.array([p.date is not None and p.date >= must_show_since for p in scene.points])
    )
    vaccine = np.array([p.vaccine is not None for p in scene.points])
    in_window = shown & antigen & ~np.array([p.greyed for p in scene.points])
    priorities = (
        Priority("recent antigens", recent),
        Priority("vaccines", vaccine & shown),
        Priority("antigens in window", in_window),
        Priority("sera", shown & ~antigen),
        Priority("all antigens", shown & antigen),
    )
    choice = choose_frame(
        np.where(shown[:, None], xy, np.nan), priorities, size=frame_size, furniture=furniture
    )
    frame: Frame = choice.frame
    page = frame.page(xy)
    ids = [p.id for p in scene.points if p.vaccine is not None and p.shown]
    rows = [i for i, p in enumerate(scene.points) if p.vaccine is not None and p.shown]
    placed = place_labels(
        page[rows],
        [scene.points[i].vaccine or "" for i in rows],
        page[shown],
        furniture,
    )
    labels = dict(zip(ids, placed, strict=True))
    draw_pdf(scene, frame, labels, out_pdf, look)
    doc = i7_document(
        scene,
        frame,
        labels,
        chart=chart,
        pdf=out_pdf,
        created=created,
        provenance=provenance,
        orientation=orientation,
    )
    i7 = write_i7(doc, out_pdf)
    return FinishedMap(
        scene, choice, labels, out_pdf, i7, recent_hidden(scene, frame, furniture, must_show_since)
    )
