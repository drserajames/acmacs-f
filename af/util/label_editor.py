"""A label editor: one self-contained HTML file beside a figure (DECISIONS Q232 a).

The page shows the figure as a picture with its movable labels taken out, and draws each label
over it where the build placed it. Labels are dragged or hidden there, and the build's own checks
are rerun as they move. Save writes the hand rows the figure's build reads back, each with why,
by whom and when. No server and nothing to install: the file opens in a browser.

What the page does is common (label_editor.js): selecting, dragging, hiding, the panel, the
"keep the rest where they are" freeze, and the refusals on Save. What a label is, where its text
and leader go and which positions the build refuses belong to the figure, so each figure brings
an adapter script defining ``ADAPTER``:

- ``offset(label)`` -> ``{dx, dy}``: the row a label's current box saves as;
- ``refused(labels, willBeHand)`` -> ``[{ids, text}]``: what the build will stop on (red);
- ``warned(labels)`` -> ``[{ids, text}]``: what it will draw but reads badly (amber);
- ``draw(label)`` -> ``{leader: [[x, y], [x, y]] | null, lines: [{x, y, text, anchor}]}``;
- ``textStyle(label)`` -> the SVG attributes of the label's text.

They are called on the shown labels only. ``data`` (the page's ``#data``), in points with y down:

- ``save_as``: the file name Save offers; ``tables``: ``{pin, hide}``, the TOML tables it writes;
- ``title``, ``subtitle``, ``saves`` and ``saved``: the panel's heading, the line under it, what
  is said of the file before and after saving;
- ``comment``: the saved file's opening comment lines; ``header``: its top-level keys, in order;
- ``page``: ``{width, height}``; ``font``: :func:`embed_font`;
- ``labels``: ``[{id, text, lines, anchor, box, keys, pin, hidden, hide}]``, ``id`` the label's
  index; ``keys`` the fields naming it in a saved row; ``pin`` ``{hand, reason, by, date, keys?}``
  or null; ``hide`` the hand row that hid it, ``{reason, by, date, keys?}``, or null. Any other
  field is the adapter's;
- ``adapter``: anything else the adapter needs.

A saved row is the label's keys (a pin's or hide's own, when it has them), its text, ``dx`` and
``dy`` for a pin, then reason, by and date.
"""

from __future__ import annotations

import base64
import io
import json
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import font_manager

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.figure import Figure


def write_editor(
    path: Path, fig: Figure, data: Mapping[str, Any], adapter_js: str, dpi: int = 216
) -> None:
    """Write the editor for a drawn figure whose movable labels are already hidden.

    ``dpi`` is the picture's: 216 is 3 pixels a point, sharp at 300% zoom."""
    # A line on a big figure is far thinner than a pixel, and a PNG fades it to nothing where a
    # PDF viewer or a printer draws it a dot wide: so the picture draws every line at least a
    # pixel wide, as the page looks. The figure's own files are written before this.
    pixel = 72 / dpi
    for ax in fig.axes:
        for collection in ax.collections:
            width = np.asarray(collection.get_linewidth(), dtype=float)
            collection.set_linewidth(_at_least(width, pixel).tolist())
        for line in ax.lines:
            line.set_linewidth(float(_at_least(np.asarray(line.get_linewidth()), pixel)))
        for patch in ax.patches:
            patch.set_linewidth(float(_at_least(np.asarray(patch.get_linewidth()), pixel)))
    picture = io.BytesIO()
    fig.savefig(picture, format="png", dpi=dpi)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_html(data, picture.getvalue(), adapter_js))


def embed_font(path: Path) -> dict[str, str | None]:
    """The label font for the browser: embedded when it is a single font file, else by family
    name (a collection such as macOS Courier.ttc is installed where the editor is opened)."""
    family = font_manager.get_font(path).family_name
    fmt = {".ttf": "truetype", ".otf": "opentype"}.get(path.suffix.lower())
    data = base64.b64encode(path.read_bytes()).decode() if fmt else None
    return {"family": family, "format": fmt, "data": data}


def _at_least(width: np.ndarray, pixel: float) -> np.ndarray:
    """Line widths no thinner than a pixel; a line with no width stays undrawn."""
    return np.where(width > 0, np.maximum(width, pixel), 0.0)


def _asset(name: str) -> str:
    return resources.files("af.util").joinpath(name).read_text()


def _html(data: Mapping[str, Any], png: bytes, adapter_js: str) -> str:
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    picture = base64.b64encode(png).decode()
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_text(data.get('title', 'Labels'))}</title>"
        f"<style>{_asset('label_editor.css')}</style></head><body>"
        f"<script type='application/json' id='data'>{payload}</script>"
        f"<script>const PICTURE = 'data:image/png;base64,{picture}';</script>"
        f"<script>{_asset('label_editor.js')}</script>"
        f"<script>{adapter_js}</script>"
        "<script>startLabelEditor(ADAPTER);</script></body></html>"
    )


def _text(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
