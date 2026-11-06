"""The tree label editor: one self-contained HTML file beside a figure (DECISIONS Q232 a).

It is written from the same drawing as the figure (render.render), so what it shows is what
prints: the figure with its substitution labels taken out, as a picture, and each label drawn
over it in the figure's font at the box the renderer placed it in, with its leader. The page is
af.util.label_editor's; this module gives it the tree's labels and its adapter (label_editor.js
here), which reruns the checks pin clearing and the draw report make as labels move. Save writes
the rows af.tree.draw.hand reads back, each with why and by whom.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from af.util import label_editor

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
    page = dict(page)
    ink: np.ndarray = page.pop("ink")
    font_file = Path(page.pop("font_file"))
    figure = page["figure"]
    rows = [_shown(k, lab) for k, lab in enumerate(labels)]
    rows += [_hidden(len(rows) + k, lab) for k, lab in enumerate(hidden)]
    data = {
        "save_as": editor.save_as,
        "title": f"{figure or 'Tree'}: labels",
        "subtitle": editor.tree,
        "saves": "the file the round's trees.toml names",
        "saved": "Put it beside the round's trees.toml, replacing the old one, and rebuild.",
        "names": "node of",
        "comment": [f"Tree labels placed by hand: {figure}. {_COMMENT[0]}", *_COMMENT[1:]],
        "header": {"figure": figure, "tree": editor.tree},
        "page": {"width": page["width"], "height": page["height"]},
        "font": label_editor.embed_font(font_file),
        "labels": rows,
        "adapter": {
            **page,
            "ink": {
                "rows": int(ink.shape[0]),
                "cols": int(ink.shape[1]),
                "bits": base64.b64encode(np.packbits(ink, axis=None).tobytes()).decode(),
            },
        },
    }
    adapter = resources.files("af.tree.draw").joinpath("label_editor.js").read_text()
    label_editor.write_editor(editor.path, fig, data, adapter, dpi=editor.dpi)


# the saved file's opening comment, after its first words
_COMMENT = (
    "Written by af's label editor (af.tree.draw.editor),",
    "read back by af.tree.draw.hand; the round's trees.toml names this file. Each label is on the",
    "common ancestor of leaves first and last, its box's top-left at the node + (dx, dy) x the page"
    " height.",
)
_DONE_BY = re.compile(r"^(.*) \(([^,()]+), (\d{4}-\d{2}-\d{2})\)$", re.S)


def _row(reason: str) -> dict[str, str]:
    """A reason as hand.py and the config give it, "why (by, date)" for a hand row, split."""
    found = _DONE_BY.match(reason)
    if found is None:
        return {"reason": reason, "by": "", "date": ""}
    return {"reason": found[1], "by": found[2], "date": found[3]}


def _label(k: int, lab: dict, hidden: bool) -> dict:
    return {
        "id": k,
        "text": lab["text"],
        "lines": lab["text"].split(" "),
        "anchor": lab["node"],
        "box": lab["box"],
        "keys": {"first": lab["first"], "last": lab["last"], "leaves": lab["leaves"]},
        "glyph_w": lab["glyph_w"],
        "line_w": lab["line_w"],
        "tether": lab.get("tether", lab["node"]),  # where its leader ends
        "hidden": hidden,
    }


def _shown(k: int, lab: dict) -> dict:
    pin = lab["pin"]
    if pin is not None:
        keys = {"first": pin["first"], "last": pin["last"], "leaves": lab["leaves"]}
        pin = {"hand": pin["hand"], **_row(pin["reason"]), "keys": keys}
    return {**_label(k, lab, False), "pin": pin, "hide": None}


def _hidden(k: int, lab: dict) -> dict:
    return {**_label(k, lab, True), "pin": None, "hide": _row(lab["reason"])}
