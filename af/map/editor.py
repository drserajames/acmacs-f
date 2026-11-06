"""The map label editor: one self-contained HTML file beside a map (af.util.label_editor).

Sarah, 30 Oct 2026: "map labels need an interface to hand move them". The page is the shared label
editor with the map's adapter (label_editor.js): the map drawn without its labels, as a picture,
and each vaccine label over it where the build placed it. Labels are dragged or hidden there, the
build's refusals (af.map.labels.hand_clashes) rerun as they move, and Save writes the map's hand
labels file (af.map.hand), each row with why, by whom and when. One file per map, written from its
first window's page: the frame and the labels' places are the same on every window.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from af.map.hand import HandLabels
from af.map.labels import Placed
from af.map.render import Look, PointStyle, draw_scene, text_metrics
from af.map.style import Scene
from af.map.viewport import Box, Frame
from af.util import label_editor

_COMMENT = (
    "Written by af's label editor (af.map.editor),",
    "read back by af.map.hand; the round's maps.toml names this file (hand_labels). Each label is",
    "on the point named by name + passage, its box's top-left at the point + (dx, dy) in fractions",
    "of the page.",
)


def write_map_editor(
    path: Path,
    *,
    map_name: str,
    chain: str,
    save_as: str,
    scene: Scene,
    frame: Frame,
    labels: Mapping[str, Placed],
    keys: Mapping[str, str],
    hidden: Sequence[tuple[str, Placed]],
    hand: HandLabels | None,
    furniture: Sequence[Box],
    look: Look,
    styles: Mapping[str, PointStyle] | None = None,
) -> None:
    """Write the editor for one map page. ``labels``: the labels the page prints, by point id;
    ``hidden``: (point id, the place it would take) for each label the hand file hides; ``keys``:
    point id -> name + passage, how a row names the point."""
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont

    width = look.page_points
    height = width * frame.aspect
    fig = plt.figure(figsize=(width / 72, height / 72), dpi=72)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    draw_scene(ax, scene, frame, {}, look, styles=styles)  # the map without its labels
    rows_of = {p.point: p for p in hand.label_pins} if hand else {}
    hides_of = {h.point: h for h in hand.hide_labels} if hand else {}

    def row(k: int, pid: str, lab: Placed, is_hidden: bool) -> dict[str, Any]:
        key = keys[pid]
        pin = rows_of.get(key)
        hide = hides_of.get(key)
        return {
            "id": k,
            "text": lab.text,
            "lines": [lab.text],
            "anchor": [lab.anchor[0] * width, lab.anchor[1] * height],
            "box": [
                lab.box[0] * width,
                lab.box[1] * height,
                lab.box[2] * width,
                lab.box[3] * height,
            ],
            "keys": {"point": key},
            "descent": text_metrics(lab.text, look.label_size)[2],
            "hidden": is_hidden,
            "pin": {"hand": True, "reason": pin.reason, "by": pin.by, "date": pin.date}
            if pin and lab.hand
            else None,
            "hide": {"reason": hide.reason, "by": hide.by, "date": hide.date}
            if hide and is_hidden
            else None,
        }

    shown = [row(k, pid, lab, False) for k, (pid, lab) in enumerate(labels.items())]
    shown += [row(len(shown) + k, pid, lab, True) for k, (pid, lab) in enumerate(hidden)]
    page = frame.page(scene.xy())
    drawn = [p.shown for p in scene.points]
    points = [[float(x * width), float(y * height)] for (x, y), d in zip(page, drawn, strict=True)
              if d and np.isfinite([x, y]).all()]  # fmt: skip
    data = {
        "save_as": save_as,
        "title": f"{map_name}: labels",
        "subtitle": chain,
        "saves": "the file the round's maps.toml names (hand_labels)",
        "saved": "Put it beside the round's maps.toml, replacing the old one, and rebuild.",
        "names": "point",
        "comment": [f"Map labels placed by hand: {map_name}. {_COMMENT[0]}", *_COMMENT[1:]],
        "header": {"map": map_name, "chain": chain},
        "page": {"width": width, "height": height},
        "font": label_editor.embed_font(Path(findfont(FontProperties()))),
        "labels": shown,
        "adapter": {
            "furniture": [
                {
                    "name": b.name,
                    "box": [b.left * width, b.top * height, b.right * width, b.bottom * height],
                }
                for b in furniture
            ],  # fmt: skip
            "mark_radius": look.antigen_radius * look.mark_scale * width,
            "gap": look.label_gap,
            "font_size": look.label_size,
            "halo": look.label_halo,
            "points": points,
        },
    }
    adapter = resources.files("af.map").joinpath("label_editor.js").read_text()
    label_editor.write_editor(path, fig, data, adapter)
    plt.close(fig)


def editor_path(out_root: Path, folder: str) -> Path:
    """Where a map's label editor goes: beside its window folders."""
    return out_root / "map" / folder / "labels-editor.html"


def save_name(hand_labels: Path | None, folder: str) -> str:
    """The file name Save offers: the one the maps config names, or the conventional one."""
    if hand_labels is not None:
        return hand_labels.name
    safe = re.sub(r"[^\w.-]", "_", folder)
    return f"labels-{safe}.toml"
