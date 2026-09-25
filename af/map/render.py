"""Draw a styled map to PDF and describe it in an I7 JSON (interface I7, workstream 11).

Why both at once: the report builder never looks inside a PDF. It embeds the PDF the I7 names (by
sha256), and the comparison reads only the I7. So the I7 is written from the same :class:`Scene`
the PDF is drawn from, and only after the PDF exists.

Look: similar to today's report maps (DECISIONS: "similar look, a little flexibility"): square
page, light 1-unit grid, sera as open squares, egg antigens as egg shapes, greyed antigens drawn
first and small-looking, legend box bottom-left with counts, bold title top-left, vaccines
enlarged with labels.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.map.labels import Placed
from af.map.style import Scene, ScenePoint
from af.map.viewport import Box, Frame

GREY = "#c8c8c8"
SERUM_OUTLINE = "#9a9a9a"


@dataclass(frozen=True)
class Look:
    """Sizes as fractions of the page side (so the look does not depend on the PDF size)."""

    page_points: float = 800.0
    antigen_radius: float = 0.0125
    vaccine_scale: float = 2.0
    serum_half_side: float = 0.0125
    title_size: float = 25.0
    label_size: float = 22.0
    legend_row_height: float = 0.0275
    legend_char_width: float = 0.0105
    legend_pad: float = 0.0125


DEFAULT_LOOK = Look()


def legend_box(scene: Scene, look: Look) -> Box:
    """Where the legend will be drawn, bottom-left; the frame search and label placer avoid it."""
    rows = len(scene.legend)
    text = max((len(t) for t, _, _ in scene.legend), default=0)
    count = max((len(str(n)) for _, _, n in scene.legend), default=0)
    width = (
        look.legend_pad * 4 + look.legend_row_height + (text + count + 2) * look.legend_char_width
    )
    height = rows * look.legend_row_height + 2 * look.legend_pad
    return Box(
        "legend",
        look.legend_pad,
        1 - look.legend_pad - height,
        look.legend_pad + width,
        1 - look.legend_pad,
    )


def title_box(scene: Scene, look: Look) -> Box:
    width = min(0.95, 0.024 + len(scene.title) * look.title_size * 0.6 / look.page_points)
    return Box("title", 0.0, 0.0, width, 0.06)


def draw_pdf(
    scene: Scene,
    frame: Frame,
    labels: Mapping[str, Placed],
    out_pdf: Path,
    look: Look = DEFAULT_LOOK,
) -> None:
    """Draw ``scene`` inside ``frame`` to ``out_pdf`` (one page)."""
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse, Rectangle

    side = look.page_points / 72.0
    fig = plt.figure(figsize=(side, side), dpi=72)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # y grows downward, as in the map frame
    ax.axis("off")
    for k in range(int(frame.size) + 1):
        g = k / frame.size
        ax.plot([g, g], [0, 1], color="#dddddd", lw=0.8, zorder=0)
        ax.plot([0, 1], [g, g], color="#dddddd", lw=0.8, zorder=0)
    page = frame.page(scene.xy())

    def order(p: ScenePoint) -> int:
        if p.kind == "serum":
            return 0
        if p.vaccine:
            return 4
        return 1 if (p.greyed or p.colour is None) else (2 if p.reference else 3)

    for i in sorted(range(len(scene.points)), key=lambda i: order(scene.points[i])):
        p = scene.points[i]
        if not p.shown:
            continue
        x, y = page[i]
        z = order(p) + 1
        if p.kind == "serum":
            s = look.serum_half_side
            ax.add_patch(
                Rectangle((x - s, y - s), 2 * s, 2 * s, fc="none", ec=SERUM_OUTLINE, lw=1, zorder=z)
            )
            continue
        grey = p.greyed or p.colour is None
        fill = GREY if grey else p.colour
        edge = GREY if grey else "black"
        r = look.antigen_radius * (look.vaccine_scale if p.vaccine else 1.0)
        if p.vaccine:
            edge = "black"
        if p.passage_class == "egg":
            ax.add_patch(Ellipse((x, y), 1.7 * r, 2.3 * r, fc=fill, ec=edge, lw=0.8, zorder=z))
        else:
            ax.add_patch(Circle((x, y), r, fc=fill, ec=edge, lw=0.8, zorder=z))
    for lab in labels.values():
        ax.text(
            lab.box[0],
            lab.box[3],
            lab.text,
            fontsize=look.label_size,
            va="bottom",
            ha="left",
            zorder=8,
        )
    ax.text(0.024, 0.015, scene.title, fontsize=look.title_size, weight="bold", va="top", zorder=9)
    _draw_legend(ax, scene, look)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    if not out_pdf.is_file() or out_pdf.stat().st_size == 0:
        raise RuntimeError(f"map PDF not written: {out_pdf}")


def _draw_legend(ax: Any, scene: Scene, look: Look) -> None:
    from matplotlib.patches import Circle, Rectangle

    box = legend_box(scene, look)
    ax.add_patch(
        Rectangle(
            (box.left, box.top),
            box.right - box.left,
            box.bottom - box.top,
            fc="white",
            ec="black",
            lw=1,
            zorder=10,
        )
    )
    h = look.legend_row_height
    font = h * look.page_points * 0.62
    for k, (text, colour, count) in enumerate(scene.legend):
        y = box.top + look.legend_pad + h * (k + 0.5)
        ax.add_patch(
            Circle(
                (box.left + look.legend_pad + h / 2, y),
                h * 0.35,
                fc=colour,
                ec="black",
                lw=0.6,
                zorder=11,
            )
        )
        ax.text(box.left + 2 * look.legend_pad + h, y, text, fontsize=font, va="center", zorder=11)
        ax.text(
            box.right - look.legend_pad,
            y,
            str(count),
            fontsize=font,
            va="center",
            ha="right",
            zorder=11,
        )


def i7_document(
    scene: Scene,
    frame: Frame,
    labels: Mapping[str, Placed],
    *,
    chart: str,
    pdf: Path,
    created: dt.datetime,
    provenance: dict[str, Any],
    orientation: dict[str, object] | None = None,
) -> dict[str, Any]:
    """The I7 JSON for a rendered map (eu-23's I7 draft v1). ``created`` must carry a time zone."""
    if created.tzinfo is None:
        raise ValueError("I7 'created' needs a time zone")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    page = frame.page(scene.xy())
    inside = (page >= 0).all(axis=1) & (page <= 1).all(axis=1)

    def point(i: int, p: ScenePoint) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": p.id,
            "name": p.name,
            "passage_class": p.passage_class,
            "date": p.date.isoformat() if p.date else None,
            "xy": [round(p.xy[0], 4), round(p.xy[1], 4)] if p.xy is not None else None,
            "shown": p.shown,
            "in_viewport": bool(inside[i]) if p.xy is not None else False,
            "clade": p.legend,
            "colour": p.colour,
            "greyed": p.greyed,
            "vaccine": p.vaccine is not None,
            "reference": p.reference,
        }
        if p.hidden_reason:
            d["hidden_reason"] = p.hidden_reason
        if p.kind == "serum":
            d["serum_id"] = p.serum_id
        if p.id in labels:
            d["label"] = labels[p.id].text
        return d

    map_block: dict[str, Any] = {
        "chart": chart,
        "window": {
            "name": scene.window.name,
            "since": scene.window.since.isoformat() if scene.window.since else None,
        },
        "viewport": [frame.x, frame.y, frame.size, frame.size],
        "clade_scheme": scene.scheme,
        "legend": [{"clade": t, "count": n} for t, _, n in scene.legend],
        "antigens": [point(i, p) for i, p in enumerate(scene.points) if p.kind == "antigen"],
        "sera": [point(i, p) for i, p in enumerate(scene.points) if p.kind == "serum"],
    }
    if orientation is not None:
        map_block["orientation"] = orientation
    return {
        "i7_version": 1,
        "kind": "map",
        "title": scene.title,
        "placeholder": False,
        "figure": {"pdf": pdf.name, "sha256": digest, "pages": 1},
        "provenance": {"producer": "af.map.render", "created": created.isoformat(), **provenance},
        "map": map_block,
    }


def write_i7(document: dict[str, Any], out_pdf: Path) -> Path:
    """Write ``<pdf stem>.i7.json`` beside the PDF, after checking the PDF hash still matches."""
    digest = hashlib.sha256(out_pdf.read_bytes()).hexdigest()
    if document["figure"]["sha256"] != digest:
        raise RuntimeError(f"{out_pdf} changed after its I7 was built")
    target = out_pdf.with_name(out_pdf.stem + ".i7.json")
    target.write_text(json.dumps(document, indent=1, allow_nan=False) + "\n")
    return target


def recent_hidden(scene: Scene, frame: Frame, furniture: Sequence[Box]) -> int:
    """Shown, non-greyed antigens that end up off the page or under furniture (should be 0)."""
    from af.map.viewport import hidden_mask

    page = frame.page(scene.xy())
    hidden = hidden_mask(page, tuple(furniture))
    return int(
        sum(
            1
            for i, p in enumerate(scene.points)
            if p.kind == "antigen" and p.shown and not p.greyed and bool(hidden[i])
        )
    )
