"""The tree label editor: one self-contained HTML file beside a figure (DECISIONS Q232 a).

It is written from the same drawing as the figure (render.render), so what it shows is what
prints: the figure with its substitution labels taken out, as a picture, and each label drawn
over it in the figure's font at the box the renderer placed it in, with its leader. Labels are
dragged or hidden there, the checks pin clearing and the draw report make are rerun as they move,
and Save writes the rows af.tree.draw.hand reads back, each with why and by whom. No server and
nothing to install: the file opens in a browser.
"""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from matplotlib import font_manager

from .aa_labels import AALabel, LabelHide

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class PinRow(Protocol):
    """What the editor keeps of the pin holding a label (figure.LabelPin)."""

    @property
    def first(self) -> str: ...
    @property
    def last(self) -> str: ...
    @property
    def reason(self) -> str: ...
    @property
    def hand(self) -> bool: ...


@dataclass
class EditorRequest:
    path: Path  # the HTML file
    save_as: str = "labels.toml"  # the hand labels file name Save offers (af.tree.draw.hand)
    tree: str = ""  # the tree the figure draws (store dataset@version), shown and saved
    pins: Mapping[int, PinRow] = field(default_factory=dict)  # label index -> its pin
    hidden: list[tuple[AALabel, LabelHide]] = field(default_factory=list)  # hidden by hand
    dpi: int = 216  # the background picture: 3 pixels a point, sharp at 300% zoom


def write_editor(
    editor: EditorRequest,
    fig: Figure,
    labels: list[dict],
    page: dict[str, Any],
    hidden: list[dict],
) -> None:
    """Write the editor for a drawn figure whose label artists are already hidden.

    ``labels`` and ``hidden`` are the renderer's label rows (render._editor_layout,
    _editor_hidden); ``page`` the page, label style, occupancy and check settings
    (render._editor_page)."""
    # A branch on a big tree is far thinner than a pixel, and a PNG fades it to nothing where a
    # PDF viewer or a printer draws it a dot wide: so the picture draws every line at least a
    # pixel wide, as the page looks. The PDF is written before this and is not touched.
    pixel = 72 / editor.dpi
    for ax in fig.axes:
        for collection in ax.collections:
            width = np.asarray(collection.get_linewidth(), dtype=float)
            collection.set_linewidth(_at_least(width, pixel).tolist())
        for line in ax.lines:
            line.set_linewidth(float(_at_least(np.asarray(line.get_linewidth()), pixel)))
        for patch in ax.patches:
            patch.set_linewidth(float(_at_least(np.asarray(patch.get_linewidth()), pixel)))
    picture = io.BytesIO()
    fig.savefig(picture, format="png", dpi=editor.dpi)
    page = dict(page)
    ink: np.ndarray = page.pop("ink")
    font_file = Path(page.pop("font_file"))
    data = {
        "tree": editor.tree,
        "save_as": editor.save_as,
        "page": page,
        "ink": {
            "rows": int(ink.shape[0]),
            "cols": int(ink.shape[1]),
            "bits": base64.b64encode(np.packbits(ink, axis=None).tobytes()).decode(),
        },
        "font": _font(font_file),
        "labels": labels,
        "hidden": hidden,
    }
    editor.path.parent.mkdir(parents=True, exist_ok=True)
    editor.path.write_text(_html(data, picture.getvalue()))


def _at_least(width: np.ndarray, pixel: float) -> np.ndarray:
    """Line widths no thinner than a pixel; a line with no width stays undrawn."""
    return np.where(width > 0, np.maximum(width, pixel), 0.0)


def _font(path: Path) -> dict[str, str | None]:
    """The label font for the browser: embedded when it is a single font file, else by family
    name (a collection such as macOS Courier.ttc is installed where the editor is opened)."""
    family = font_manager.get_font(path).family_name
    fmt = {".ttf": "truetype", ".otf": "opentype"}.get(path.suffix.lower())
    data = base64.b64encode(path.read_bytes()).decode() if fmt else None
    return {"family": family, "format": fmt, "data": data}


def _asset(name: str) -> str:
    return resources.files("af.tree.draw").joinpath(name).read_text()


def _html(data: dict[str, Any], png: bytes) -> str:
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    picture = base64.b64encode(png).decode()
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Tree labels</title>"
        f"<style>{_asset('label_editor.css')}</style></head><body>"
        f"<script type='application/json' id='data'>{payload}</script>"
        f"<script>const PICTURE = 'data:image/png;base64,{picture}';</script>"
        f"<script>{_asset('label_editor.js')}</script></body></html>"
    )
