"""Addenda drawn from a round's maps: one pass after af.map.build has built every map.

The serum-circle addenda (and the serum-coverage ones, 10-serology) choose their sera across
all of an addendum's maps at once: a section that matches no serum is an error for the whole
addendum, not for one lab's map (af.chart.serum_selection). So the build keeps each map's
:class:`MapPage` (its chart, drawn projection, points, frame) and, given a round's
``addenda.toml``, runs every block in it after the maps: a block kind (``[[circles]]``,
``[[coverage]]``) names its handler. Without ``--addenda`` nothing here runs and the maps are
built exactly as before.

``addenda.toml`` is round data, never code (it names the round's strains)::

    [addenda]
    meeting = 2026-09-21     # sera titrated in the recency_months before the meeting's month
    recency_months = 12

    [[circles]]
    addendum = 5             # the cover number; figures go under addenda/addendum-5-h3/
    subtype = "h3"
    title = "Addendum 5 (H3N2 multiple serum circles, cell sera)"
    window = "all"
    fold = 2.0
    radius = "theoretical"
    passage_classes = ["cell"]
    maps = [{ chart = "h3-hi-guinea-pig-cdc" }, { empty = true }, ...]   # page order, 6 a page
    sections = [{ label = "K", strains = ["..."], reason = "..." }, ...]

Each handler writes its figures, then the block's ``addendum.json`` (the page plan 09's frame
reads) and ``shown.json`` (every shown serum, with its section, reason and circle record).
"""

from __future__ import annotations

import datetime as dt
import json
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from af.chart.model import Chart, Projection
from af.map.render import CircleMark, Look, PointStyle
from af.map.style import ColourScheme, PointIn, Scene, Window, style_points
from af.map.viewport import Frame
from af.tables.labs import Lab

KINDS = ("circles", "coverage")
PAGE_CELLS = 6
PAGE_KINDS = ("plain", "circles", "names")  # a circle addendum's pages, in the shipped order


class AddendaError(ValueError):
    pass


@dataclass(frozen=True)
class MapPage:
    """What an addendum draws from one built map: everything af.map.build had when it finished
    the map's pages, so an addendum figure is the map's own figure with more on it."""

    folder: str
    centre: str  # the collaborating centre: the chain dataset's first segment
    chain_ref: dict[str, Any]  # the chain version the map was drawn from
    chart: Chart
    projection: Projection  # as drawn: the map's column-base overrides applied
    points: tuple[PointIn, ...]  # displayed coordinates, every point (hidden ones too)
    scheme: ColourScheme
    frame: Frame  # the map's one frame (all its windows)
    look: Look
    windows: tuple[tuple[Window, str], ...]  # each window and its page title
    must_show_since: dt.date
    provenance: dict[str, Any]
    vaccine_labels: dict[str, str] = field(default_factory=dict)  # point id -> the map's label

    def layout(self) -> np.ndarray:
        """The drawn layout in chart order (antigens then sera), NaN where the chart has no
        coordinates. Hidden points keep theirs: radii and coverage count every titrated antigen."""
        out = np.full((self.chart.n_points, 2), np.nan)
        for p in self.points:
            k = int(p.id[2:]) + (self.chart.n_antigens if p.kind == "serum" else 0)
            if p.xy is not None:
                out[k] = p.xy
        return out

    def window(self, name: str) -> tuple[Window, str]:
        found = [w for w in self.windows if w[0].name == name]
        if not found:
            names = [w.name for w, _ in self.windows]
            raise AddendaError(f"{self.folder}: no window {name!r} (has {names})")
        return found[0]

    def scene(self, window: str, *, vaccines: bool = False) -> Scene:
        """The map's styled scene in ``window``, as its page draws it, with its vaccine marks
        or (the circle addenda, as ae drew them) without."""
        w, title = self.window(window)
        marks = self.vaccine_labels if vaccines else {}
        return style_points(list(self.points), self.scheme, w, title=title, vaccines=dict(marks))


@dataclass(frozen=True)
class AddendaContext:
    colours: Any  # af.map.colouring.StoreColours: serum strains, for selection
    since: dt.date  # sera titrated before it are left out (the round's recency)
    out_root: Path
    created: dt.datetime
    labs: Mapping[str, Lab]  # by lab code: how serum ids print


@dataclass(frozen=True)
class AddendumResult:
    directory: Path
    manifest: dict[str, Any]
    figures: tuple[Path, ...]
    selection: dict[str, Any]
    shown: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AddendaConfig:
    meeting: dt.date
    recency_months: int
    blocks: dict[str, list[dict[str, Any]]]

    @property
    def since(self) -> dt.date:
        """The first of the month ``recency_months`` before the meeting's month (ae's
        older_12_months_date): never derived from today."""
        months = self.meeting.year * 12 + self.meeting.month - 1 - self.recency_months
        return dt.date(months // 12, months % 12 + 1, 1)


def load_addenda(path: Path) -> AddendaConfig:
    """Read a round's addenda.toml. An unknown block kind, or no block at all, is an error."""
    data = tomllib.loads(path.read_text())
    head = data.pop("addenda", None)
    if not isinstance(head, dict) or "meeting" not in head:
        raise AddendaError(f"{path}: needs [addenda] meeting = <date>")
    if unknown := sorted(set(data) - set(KINDS)):
        raise AddendaError(f"{path}: unknown addendum kinds {unknown} (known: {list(KINDS)})")
    if not any(data.values()):
        raise AddendaError(f"{path}: no addendum blocks")
    meeting = head["meeting"]
    if not isinstance(meeting, dt.date):
        raise AddendaError(f"{path}: [addenda] meeting must be a date, got {meeting!r}")
    return AddendaConfig(meeting, int(head.get("recency_months", 12)), dict(data))


def directory_name(block: Mapping[str, Any]) -> str:
    return f"addendum-{int(block['addendum'])}-{block['subtype']}"


def block_maps(block: Mapping[str, Any]) -> list[str | None]:
    """The block's slots in page order: a map folder, or None for an empty slot."""
    out: list[str | None] = []
    for m in block.get("maps", ()):
        if m.get("empty"):
            out.append(None)
        elif "chart" in m:
            out.append(str(m["chart"]))
        else:
            raise AddendaError(f"{directory_name(block)}: a map slot needs chart or empty: {m}")
    if not any(out):
        raise AddendaError(f"{directory_name(block)}: no maps")
    return out


Handler = Callable[[Mapping[str, Any], Mapping[str, MapPage], AddendaContext], AddendumResult]


def handlers() -> dict[str, Handler]:
    """Each block kind's handler, looked up by the block's array name."""
    from af.serology.coverage import run_coverage  # 10-serology's; lazy: it imports this module

    return {"circles": run_circles, "coverage": run_coverage}


def run_addenda(
    config: AddendaConfig, pages: Mapping[str, MapPage], ctx: AddendaContext
) -> list[AddendumResult]:
    """Run every block, after checking that every map a block names was built."""
    known = handlers()
    for kind, blocks in config.blocks.items():
        if kind not in known:
            raise AddendaError(f"no handler for [[{kind}]] blocks yet")
        for block in blocks:
            if missing := [m for m in block_maps(block) if m is not None and m not in pages]:
                raise AddendaError(f"{directory_name(block)}: maps not built: {missing}")
    results = []
    for kind, blocks in config.blocks.items():
        for block in blocks:
            result = known[kind](block, pages, ctx)
            result.directory.mkdir(parents=True, exist_ok=True)
            (result.directory / "addendum.json").write_text(json.dumps(result.manifest, indent=1))
            (result.directory / "shown.json").write_text(json.dumps(result.shown, indent=1))
            results.append(result)
    return results


def draw_addendum_figure(
    page: MapPage,
    out_pdf: Path,
    *,
    window: str,
    created: dt.datetime,
    styles: Mapping[str, PointStyle] | None = None,
    circles: Sequence[CircleMark] = (),
    caption: Sequence[str] = (),
    vaccine_labels: dict[str, str] | None = None,
    addendum: dict[str, Any] | None = None,
    title: str | None = None,
) -> Path:
    """One addendum figure: the map's page in ``window`` and its frame, with the addendum's
    serum styles, circles and caption (and ``title`` for the page's own). Returns its I7."""
    from af.map.finish import finish_map

    w, page_title = page.window(window)
    title = title if title is not None else page_title
    result = finish_map(
        list(page.points),
        chart=page.folder,
        scheme=page.scheme,
        window=w,
        title=title,
        frame_size=page.frame.size,
        frame=page.frame,
        must_show_since=page.must_show_since,
        vaccine_labels=vaccine_labels or {},
        out_pdf=out_pdf,
        created=created,
        provenance=page.provenance,
        look=page.look,
        styles=styles,
        circles=circles,
        caption=caption,
        addendum=addendum,
    )
    return result.i7


def render_coverage(
    page: MapPage,
    record: Any,
    *,
    radius: str,
    clade_line: str,
    out_pdf: Path,
    created: dt.datetime,
    addendum: dict[str, Any],
    lab: Lab,
    window: str = "all",
) -> Path:
    """One serum-coverage figure (10-serology's [[coverage]] blocks), as ae's sc-* maps: the map
    pale with its vaccine labels and legend, the serum's titrated antigens outlined within or
    outside fold, the serum and its circle (:func:`af.map.circles.coverage_marks`), and a
    three-line title: the map's, the serum (its id as maps print it), ``clade_line``.
    Returns the figure's I7."""
    from af.map.circles import coverage_marks
    from af.map.vaccines import display_serum_id

    marks, styles = coverage_marks(record, page.scene(window, vaccines=True), radius)
    map_title = page.window("all")[1] if window == "all" else page.window(window)[1]
    serum = " ".join(x for x in (record.name, display_serum_id(record.serum_id, lab)) if x)
    return draw_addendum_figure(
        page,
        out_pdf,
        window=window,
        created=created,
        styles=styles,
        circles=marks,
        vaccine_labels=page.vaccine_labels,
        addendum=addendum,
        title="\n".join([map_title, serum, clade_line]),
    )


# ---------------------------------------------------------------- serum circles


def _serum_index(chart: Chart) -> dict[tuple[str, ...], int]:
    """Each serum's selection key (af.chart.serum_selection.candidates) -> its chart index."""
    out: dict[tuple[str, ...], int] = {}
    for sr, s in enumerate(chart.sera):
        key = (s.name, s.serum_id, s.passage, s.reassortant, *s.annotations)
        if key in out:
            raise AddendaError(f"two sera share one identity: {key}")
        out[key] = sr
    return out


def _passage_overrides(
    name: str, block: Mapping[str, Any], cands: list[Any]
) -> tuple[list[Any], list[dict[str, str]]]:
    """A lab that gives no serum passage (its sera have no passage class) can have one set per
    serum, as hand curation: ``passage_class_overrides = [{ map, serum, passage_class, reason }]``
    with ``serum`` a designation substring (Sarah, Sep 2026: such a serum is judged by the strain
    it was raised against). Each must match exactly one serum of its map (design rule 1)."""
    from dataclasses import replace

    out = list(cands)
    done = []
    for o in block.get("passage_class_overrides", ()):
        missing = {"map", "serum", "passage_class", "reason"} - set(o)
        if missing:
            raise AddendaError(f"{name}: passage class override {o} lacks {sorted(missing)}")
        hits = [
            i
            for i, c in enumerate(out)
            if c.chart == o["map"] and str(o["serum"]).upper() in c.designation.upper()
        ]
        if len(hits) != 1:
            raise AddendaError(f"{name}: passage class override {o['serum']!r} on {o['map']} "
                               f"matches {len(hits)} sera (needs 1)")  # fmt: skip
        c = out[hits[0]]
        out[hits[0]] = replace(c, passage_class=str(o["passage_class"]))
        done.append({"map": o["map"], "serum": c.designation, "was": str(c.passage_class),
                     "now": str(o["passage_class"]), "reason": str(o["reason"])})  # fmt: skip
    return out, done


def run_circles(
    block: Mapping[str, Any], pages: Mapping[str, MapPage], ctx: AddendaContext
) -> AddendumResult:
    """A serum-circle addendum: per map a plain page (the circled sera marked), a circles page
    and a names page (the circled sera listed), in the block's map order, six to a page."""
    from af.chart.serum_circles import serum_circles
    from af.chart.serum_selection import Section, candidates, select_sera
    from af.map.circles import CircleLook, names_lines, serum_circle_marks

    name = directory_name(block)
    out_dir = ctx.out_root / "addenda" / name
    window = str(block.get("window", "all"))
    fold = float(block.get("fold", 2.0))
    look = CircleLook(radius=str(block.get("radius", "theoretical")))
    sections = [Section.from_config(s) for s in block.get("sections", ())]
    if not sections:
        raise AddendaError(f"{name}: no sections")
    residues = {int(r[:-1]) for s in sections for r in s.aa}
    slots = block_maps(block)
    folders = [f for f in slots if f is not None]
    cands = []
    for f in folders:
        page = pages[f]
        strains = ctx.colours.for_sera(page.chart, residues)
        cands += candidates(page.chart, strains, chart_name=f, centre=page.centre)
    cands, overridden = _passage_overrides(name, block, cands)
    classes = block.get("passage_classes")
    selection = select_sera(
        cands,
        sections,
        since=ctx.since,
        passage_classes=frozenset(classes) if classes is not None else None,
    )
    figures: dict[str, dict[str, Path]] = {}
    shown: list[dict[str, Any]] = []
    for f in folders:
        page = pages[f]
        index = _serum_index(page.chart)
        chosen = [
            (label, c) for label, cs in selection.sections.items() for c in cs if c.serum.chart == f
        ]
        if not chosen:
            raise AddendaError(f"{name}: no serum chosen on {f}; take the map out of the block")
        records = serum_circles(page.chart, page.layout(), fold, projection=page.projection)
        picked = [records[index[c.serum.key[1:]]] for _, c in chosen]
        scene = page.scene(window)
        painted = {p.id: p.colour for p in scene.points if p.kind == "antigen"}
        marks, styles = serum_circle_marks(picked, painted, look)
        lab = ctx.labs.get(str(page.chart.info.get("l", "")))
        if lab is None:
            raise AddendaError(f"{f}: lab {page.chart.info.get('l')!r} not in the labs table")
        sera = [
            {"serum": r.designation, "section": label, "reason": c.reason}
            for r, (label, c) in zip(picked, chosen, strict=True)
        ]
        figures[f] = {}
        for kind in PAGE_KINDS:
            figures[f][kind] = draw_addendum_figure(
                page,
                out_dir / f / kind / "v1" / "figure.pdf",
                window=window,
                created=ctx.created,
                styles=styles,
                circles=marks if kind != "plain" else (),
                caption=names_lines(picked, lab) if kind == "names" else (),
                addendum={"name": name, "kind": "circles", "page": kind, "sera": sera},
            )
        shown += [
            {**s, "addendum": name, "map": f, "chain": page.chain_ref, "record": r.to_json()}
            for s, r in zip(sera, picked, strict=True)
        ]
    plan = []
    for kind in PAGE_KINDS:
        for start in range(0, len(slots), PAGE_CELLS):
            cells = slots[start : start + PAGE_CELLS]
            cells += [None] * (PAGE_CELLS - len(cells))
            plan.append(
                {
                    "caption": None,
                    "cells": [
                        str(figures[c][kind].relative_to(out_dir)) if c is not None else None
                        for c in cells
                    ],
                }
            )
    manifest = {
        "kind": "circles",
        "addendum": int(block["addendum"]),
        "subtype": block["subtype"],
        "title": block.get("title", ""),
        "pages": plan,
        "figures": [str(p.relative_to(out_dir)) for fs in figures.values() for p in fs.values()],
        "declarations": [],
        "selection": {**selection.report(), "passage_class_overrides": overridden},
    }
    return AddendumResult(
        out_dir,
        manifest,
        tuple(p for fs in figures.values() for p in fs.values()),
        selection.report(),
        shown,
    )
