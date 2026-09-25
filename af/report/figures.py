"""Find the figure for each report slot: the latest one, or a pinned version.

PROVISIONAL LAYOUT until the store layout (interface I9) lands in :mod:`af.store`::

    <figures_root>/<slot>/<version>/figure.i7.json   (+ the PDF it names)

"Latest" is decided by the parsed ``provenance.created`` timestamp in each I7, never by
sorting directory names (design rule 7). Two versions with the same timestamp are an error,
not a coin toss. A broken newer version is an error too: it must not quietly fall back to
an older figure (the stale-geo trap in today's report, B-report-layer §4.1 item 3).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from af.report.i7 import I7_NAME, Figure, I7Error, load_figure


class FigureError(RuntimeError):
    """A slot has no usable figure."""


def created(fig: Figure) -> dt.datetime:
    """The figure's creation time; it must carry a time zone so times from two hosts compare."""
    raw = fig.doc["provenance"].get("created")
    if not raw:
        raise FigureError(f"{fig.i7_path}: provenance.created missing")
    when = dt.datetime.fromisoformat(raw)
    if when.tzinfo is None:
        raise FigureError(f"{fig.i7_path}: provenance.created {raw!r} has no time zone")
    return when


@dataclass(frozen=True)
class Resolved:
    slot: str
    version: str
    pinned: bool
    figure: Figure

    @property
    def created(self) -> dt.datetime:
        return created(self.figure)


def resolve(root: Path, slot: str, pin: str | None = None) -> Resolved:
    """The figure to use for ``slot``: the pinned version if given, else the latest."""
    slot_dir = root / slot
    if not slot_dir.is_dir():
        raise FigureError(f"slot {slot}: no figures at {slot_dir}")
    try:
        if pin is not None:
            path = slot_dir / pin / I7_NAME
            if not path.is_file():
                raise FigureError(f"slot {slot}: pinned version {pin!r} not found at {path}")
            return Resolved(slot, pin, True, load_figure(path))
        candidates = [
            (created(fig), path.parent.name, fig)
            for path in sorted(slot_dir.glob(f"*/{I7_NAME}"))
            for fig in [load_figure(path)]
        ]
    except I7Error as err:
        raise FigureError(f"slot {slot}: {err}") from err
    if not candidates:
        raise FigureError(f"slot {slot}: no {I7_NAME} under {slot_dir}")
    candidates.sort(key=lambda c: c[0])
    if len(candidates) > 1 and candidates[-1][0] == candidates[-2][0]:
        raise FigureError(
            f"slot {slot}: versions {candidates[-2][1]} and {candidates[-1][1]} "
            "have the same created time; pin one"
        )
    _, version, fig = candidates[-1]
    return Resolved(slot, version, False, fig)
