"""Map labels placed by hand: one file per map, the rows its label editor writes, read back.

Sarah, 30 Oct 2026: "map labels need an interface to hand move them". The same model as the tree
labels (af.tree.draw.hand; agreed with 09, 31 Oct 2026): one file per figure, the one editable copy
of what was done by hand on it (design rule 6), named in the round's maps config and kept beside
it by whoever moves the labels. Every row says why, by whom and when (rule 9). A file that is
named must exist (rule 4); a file of comments only has no rows.

    map = "lab-x-hi"                          # the map folder, for the reader
    chain = "lab/x-hi/merged@0123456789abcdef"

    # dx, dy: the label box's top-left = the point's page position + (dx, dy), in fractions of
    # the page WIDTH (af.map.labels.pinned_box), on every window page of the map.
    [[label_pins]]
    point = "B/EXAMPLETOWN/2/2019 E3"         # name + passage: exactly one point of the map
    text = "Ex/19-egg"                        # the label as printed; a relabelled point refuses
    dx = 0.012
    dy = -0.034
    reason = "clear of the cell preparation's label"
    by = "AB"
    date = "2026-10-31"

    [[hide_labels]]                           # not printed
    point = "B/EXAMPLETOWN/2/2019 MDCK1"
    text = "Ex/19-cell"
    reason = "the egg label beside it says it"
    by = "AB"
    date = "2026-10-31"
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.util.config import load_config


class HandLabelsError(ValueError):
    """A hand labels file that cannot apply: a row without a reason, a name or a date, a point
    the map does not have (or has twice), a label text that is not the point's, or a label
    both pinned and hidden."""


@dataclass(frozen=True)
class HandPin:
    point: str
    text: str
    dx: float
    dy: float
    reason: str
    by: str
    date: str


@dataclass(frozen=True)
class HandHide:
    point: str
    text: str
    reason: str
    by: str
    date: str


@dataclass(frozen=True)
class HandLabels:
    map: str = ""
    chain: str = ""
    label_pins: list[HandPin] = field(default_factory=list)
    hide_labels: list[HandHide] = field(default_factory=list)

    def record(self, path: Path) -> dict[str, Any]:
        """What the figure records of the file: its rows (design rule 9: counted, with why)."""
        return {
            "file": path.name,
            "pins": [vars(p) | {"hand": True} for p in self.label_pins],
            "hides": [vars(h) | {"hand": True} for h in self.hide_labels],
        }


def load_hand_labels(path: Path) -> HandLabels:
    """The file's rows. A row missing its reason or who made it, or with a date that is not
    YYYY-MM-DD, refuses the whole file; so does a label both pinned and hidden."""
    rows = load_config(path, HandLabels)
    problems = [
        f"{kind} {r.point} {r.text!r}: {problem}"
        for kind, group in (("pin", rows.label_pins), ("hide", rows.hide_labels))
        for r in group
        for problem in _problems(r)
    ]
    pinned = {(p.point, p.text) for p in rows.label_pins}
    problems += [
        f"{h.point} {h.text!r}: both pinned and hidden" for h in rows.hide_labels
        if (h.point, h.text) in pinned
    ]  # fmt: skip
    for kind, group in (("pin", rows.label_pins), ("hide", rows.hide_labels)):
        seen: set[str] = set()
        for r in group:
            if r.point in seen:
                problems.append(f"{kind} {r.point}: two rows for one point")
            seen.add(r.point)
    if problems:
        listed = "\n  ".join(problems)
        raise HandLabelsError(f"{path}: {len(problems)} row(s) refused:\n  {listed}")
    return rows


def _problems(row: HandPin | HandHide) -> list[str]:
    out = [f"no {name}" for name in ("reason", "by") if not getattr(row, name).strip()]
    try:
        datetime.date.fromisoformat(row.date)
    except ValueError:
        out.append(f"date {row.date!r} is not YYYY-MM-DD")
    return out


def resolve(
    rows: HandLabels, labelled: Sequence[tuple[str, str, str]]
) -> tuple[dict[str, tuple[float, float]], set[str]]:
    """The pins (point id -> dx, dy) and hidden labels (point ids) of one page. ``labelled``:
    (point id, the point's name + passage, the label text it prints) for every label the page
    would print. A row must match exactly one of them, by point and text (design rule 1)."""
    problems: list[str] = []

    def find(point: str, text: str) -> str | None:
        by_point = [(pid, t) for pid, key, t in labelled if key == point]
        if len(by_point) != 1:
            problems.append(f"{point}: {len(by_point)} labelled points (one needed)")
            return None
        pid, printed = by_point[0]
        if printed != text:
            problems.append(f"{point}: the label prints {printed!r}, the row says {text!r}")
            return None
        return pid

    pins = {}
    for p in rows.label_pins:
        pid = find(p.point, p.text)
        if pid is not None:
            pins[pid] = (p.dx, p.dy)
    hides = {pid for h in rows.hide_labels if (pid := find(h.point, h.text)) is not None}
    if problems:
        listed = "\n  ".join(problems)
        raise HandLabelsError(f"hand labels of {rows.map or 'a map'}:\n  {listed}")
    return pins, hides
