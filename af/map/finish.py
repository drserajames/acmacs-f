"""Finish one map window: style, frame, label, draw, describe. The order is the point.

Styling comes first because the legend's size (and so the space the frame must leave free)
depends on the colour scheme. The frame comes before labels, because labels are placed on the
page. The PDF comes before the I7, because the I7 carries the PDF's hash.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from af.map.hand import HandLabels
from af.map.hand import resolve as resolve_hand
from af.map.i7 import check_provenance_inputs, i7_document, write_i7
from af.map.labels import Placed, place_labels
from af.map.render import (
    DEFAULT_LOOK,
    CircleMark,
    Look,
    PointStyle,
    draw_pdf,
    legend_box,
    marker_extent,
    note_box,
    note_size,
    recent_hidden,
    text_size,
    title_box,
)
from af.map.style import ColourScheme, PointIn, Scene, Window, style_points
from af.map.viewport import (
    Box,
    Frame,
    FrameChoice,
    Priority,
    View,
    choose_frame,
    hidden_counts,
)


@dataclass(frozen=True)
class FinishedMap:
    scene: Scene
    frame: FrameChoice
    labels: dict[str, Placed]
    pdf: Path
    i7: Path
    recent_hidden: int


@dataclass(frozen=True)
class _Page:
    """What one window's page asks of its frame."""

    scene: Scene
    xy: np.ndarray  # displayed coordinates, NaN for points not shown
    priorities: tuple[Priority, ...]
    furniture: tuple[Box, ...]
    radii: np.ndarray


def _page(
    points: Sequence[PointIn],
    scheme: ColourScheme,
    window: Window,
    title: str,
    must_show_since: dt.date,
    vaccine_labels: dict[str, str],
    look: Look,
    styles: Mapping[str, PointStyle] | None,
    note: Sequence[str] = (),
) -> _Page:
    scene = style_points(points, scheme, window, title=title, vaccines=vaccine_labels)
    furniture: tuple[Box, ...] = (legend_box(scene, look), title_box(scene, look))
    if note:  # a map note is furniture too (Q323); without one the page is as it always was
        furniture = (*furniture, note_box(note, look))
    xy = scene.xy()
    shown = np.array([p.shown for p in scene.points])
    antigen = np.array([p.kind == "antigen" for p in scene.points])
    recent = (
        shown
        & antigen
        & np.array([p.date is not None and p.date >= must_show_since for p in scene.points])
    )
    vaccine = np.array([p.mark is not None for p in scene.points])
    in_window = shown & antigen & ~np.array([p.greyed for p in scene.points])
    priorities = (
        Priority("recent antigens", recent),
        Priority("vaccines", vaccine & shown),
        Priority("antigens in window", in_window),
        Priority("sera", shown & ~antigen),
        Priority("all antigens", shown & antigen),
    )
    own = styles or {}
    radii = np.array([marker_extent(p, look, own.get(p.id)) for p in scene.points])
    return _Page(scene, np.where(shown[:, None], xy, np.nan), priorities, furniture, radii)


@dataclass(frozen=True)
class EditorRequest:
    """Write a map's label editor (af.map.editor) to ``path``; Save offers ``save_as``."""

    path: Path
    map_name: str
    chain: str
    save_as: str


def _write_editor(
    editor: EditorRequest, scene: Scene, frame: Frame, page: np.ndarray, shown: np.ndarray,
    all_rows: list[int], labels: dict[str, Placed], hand: HandLabels | None,
    furniture: tuple[Box, ...], look: Look, styles: Mapping[str, PointStyle] | None,
) -> None:  # fmt: skip
    """The page's editor. A label the hand file hides is offered where the placer would put it
    among all the page's labels, so restoring it in the editor shows a sensible place."""
    from af.map.editor import write_map_editor

    keys = {scene.points[i].id: _hand_key(scene.points[i]) for i in all_rows}
    hidden_ids = [scene.points[i].id for i in all_rows if scene.points[i].id not in labels]
    hidden: list[tuple[str, Placed]] = []
    if hidden_ids:
        texts = [scene.points[i].mark or "" for i in all_rows]
        sizes = [text_size(t, look.label_size) for t in texts]
        free = place_labels(
            page[all_rows], texts, page[shown], furniture,
            sizes=[(w / look.page_points, h / look.page_points) for w, h in sizes],
            mark_radius=look.antigen_radius * look.mark_scale,
            gap=look.label_gap / look.page_points,
        )  # fmt: skip
        where = {scene.points[i].id: p for i, p in zip(all_rows, free, strict=True)}
        hidden = [(pid, where[pid]) for pid in hidden_ids]
    write_map_editor(
        editor.path, map_name=editor.map_name, chain=editor.chain, save_as=editor.save_as,
        scene=scene, frame=frame, labels=labels, keys=keys, hidden=hidden, hand=hand,
        furniture=furniture, look=look, styles=styles,
    )  # fmt: skip


def _hand_key(p: Any) -> str:
    """How a hand labels row names a point (af.map.hand): name + passage, as vaccine rows bind."""
    return f"{p.name} {p.passage or ''}".strip()


def _ordinal(day: dt.date | None) -> int:
    return day.toordinal() if day is not None else 0


def shared_frame(
    points: Sequence[PointIn],
    *,
    scheme: ColourScheme,
    windows: Sequence[tuple[Window, str]],
    frame_size: float,
    must_show_since: dt.date,
    vaccine_labels: dict[str, str],
    look: Look = DEFAULT_LOOK,
    styles: Mapping[str, PointStyle] | None = None,
    note: Sequence[str] = (),
) -> Frame:
    """One frame for all of a map's windows (each a window and its title), as the shipped maps
    use one viewport on the all, 12m and 6m pages: a reader turning the pages sees the same
    square. It is chosen as :func:`finish_map` chooses one window's frame, over every window's
    page at once (:class:`af.map.viewport.View`): each window's recent antigens visible, then
    the pages in turn, the most recent window's first and the all page last. The newest
    window's page is framed as it would be alone, and the older pages take what is left; the
    all page, where every antigen is in the window, no longer pulls the frame off the recent
    viruses (h1-cnic, R6 review). Positions and marker sizes are the same on every page:
    windows grey points, they do not move, hide or resize them. ``note``: the map's note
    (:func:`finish_map`), furniture on every page."""
    if not windows:
        raise ValueError("shared_frame needs at least one window")
    # the most recent window first (stable among equal dates); a window of every antigen last
    by_recency = sorted(windows, key=lambda wt: (wt[0].since is None, -_ordinal(wt[0].since)))
    pages = [
        _page(points, scheme, w, t, must_show_since, vaccine_labels, look, styles, note)
        for w, t in by_recency
    ]
    first, *rest = pages
    choice = choose_frame(
        first.xy,
        first.priorities,
        size=frame_size,
        furniture=first.furniture,
        radii=first.radii,
        also=tuple(View(p.priorities, p.furniture) for p in rest),
    )
    return choice.frame


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
    flags: Sequence[str] = (),
    look: Look = DEFAULT_LOOK,
    styles: Mapping[str, PointStyle] | None = None,
    frame: Frame | None = None,
    circles: Sequence[CircleMark] = (),
    caption: Sequence[str] = (),
    addendum: dict[str, Any] | None = None,
    require_shown: bool = True,
    hand: HandLabels | None = None,
    editor: EditorRequest | None = None,
    note: Sequence[str] = (),
) -> FinishedMap:
    """Produce ``out_pdf`` and its I7 for one chart and window.

    ``must_show_since``: antigens isolated on or after this date must all be visible (not off the
    page, not under the legend or title) in *every* window, including "all", where nothing is
    greyed and older antigens may fall off the edge if the frame is too small for them; among
    the frames that show them, one that keeps their markers whole wins. Raises
    ``FrameError`` rather than writing a map that hides one. Vaccines come next: shown if at all
    possible, but an old vaccine far from today's viruses may not fit (the I7 records
    ``in_viewport`` for each), then antigens in the window, sera, and everything else.

    ``frame``: draw in this frame (a map's :func:`shared_frame`) instead of choosing one for
    this window alone; it is still checked to show the recent antigens.

    ``circles``: drawn round their points (serum circles, :mod:`af.map.circles`) and recorded
    in the I7. They do not move the frame: a circle may run off the page, as ae's did.
    ``caption``: lines printed under the title (the circled sera), recorded in the I7.
    ``addendum``: what an addendum figure adds (af.map.addenda), recorded in the I7 as is.
    ``require_shown`` False draws in the given ``frame`` even where this page's furniture covers
    a recent antigen (an addendum figure keeps its map's zoom), and records in the I7 what each
    priority has hidden.
    ``hand``: the map's labels placed or hidden by hand (af.map.hand), on every window page.
    ``editor``: also write the map's label editor (af.map.editor) from this page.
    ``note``: a map's short note (Q323), printed where ``caption`` is, at the legend's text size
    (Q326), and recorded in the I7 as
    its caption, but furniture: the frame and the labels keep clear of it. Not with a caption,
    and only on a square page (the box is measured for one)."""
    # fail before drawing anything
    if created.tzinfo is None:
        raise ValueError("I7 'created' needs a time zone")
    check_provenance_inputs(provenance.get("inputs"))
    if note and caption:
        raise ValueError("a map note and a caption print in the same place: give one")
    if note and frame is not None and frame.aspect != 1.0:
        raise ValueError(f"a map note needs a square frame, got aspect {frame.aspect}")
    pg = _page(points, scheme, window, title, must_show_since, vaccine_labels, look, styles, note)
    scene, xy, furniture = pg.scene, pg.scene.xy(), pg.furniture
    shown = np.array([p.shown for p in scene.points])
    if frame is None:
        choice = choose_frame(
            pg.xy, pg.priorities, size=frame_size, furniture=furniture, radii=pg.radii
        )
    else:
        choice = hidden_counts(
            frame, pg.xy, pg.priorities, furniture, pg.radii, strict=require_shown
        )
    frame = choice.frame
    page = frame.page(xy)
    rows = [i for i, p in enumerate(scene.points) if p.mark is not None and p.shown]
    all_rows = list(rows)
    pins: dict[str, tuple[float, float]] = {}
    if hand is not None:
        labelled = [(scene.points[i].id, _hand_key(scene.points[i]), scene.points[i].mark or "")
                    for i in rows]  # fmt: skip
        pins, hidden_by_hand = resolve_hand(hand, labelled)
        rows = [i for i in rows if scene.points[i].id not in hidden_by_hand]
    ids = [scene.points[i].id for i in rows]
    texts = [scene.points[i].mark or "" for i in rows]
    sizes = [text_size(t, look.label_size) for t in texts]  # in the figure font
    placed = place_labels(
        page[rows],
        texts,
        page[shown],
        furniture,
        sizes=[(w / look.page_points, h / look.page_points) for w, h in sizes],
        mark_radius=look.antigen_radius * look.mark_scale,
        gap=look.label_gap / look.page_points,
        pinned={k: pins[pid] for k, pid in enumerate(ids) if pid in pins},
    )
    labels = dict(zip(ids, placed, strict=True))
    size = note_size(look) if note else None
    caption = tuple(note) or caption
    draw_pdf(scene, frame, labels, out_pdf, look, styles, circles, caption, size)
    if editor is not None:
        _write_editor(editor, scene, frame, page, shown, all_rows, labels, hand, furniture, look,
                      styles)  # fmt: skip
    doc = i7_document(
        scene,
        frame,
        labels,
        chart=chart,
        pdf=out_pdf,
        created=created,
        provenance=provenance,
        orientation=orientation,
        flags=flags,
        look=look,
        styles=styles,
        circles=circles,
        caption=caption,
        addendum=addendum,
        hidden=None if require_shown else choice.hidden,
    )
    i7 = write_i7(doc, out_pdf)
    return FinishedMap(
        scene,
        choice,
        labels,
        out_pdf,
        i7,
        recent_hidden(scene, frame, furniture, must_show_since),
    )
