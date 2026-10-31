"""Tree labels placed by hand: the file the label editor writes (af.tree.draw.editor), read back.

The file is the one editable copy of what was done by hand on a tree (design rule 6), kept
beside the round's tree config by whoever does it (DECISIONS Q232 a; 00, 30 Oct 2026: owned by
the person moving the labels, so no one sits between them and their own edits). Every row says
why and by whom (rule 9): a position with no reason cannot be judged when the tree changes.

    figure = "A(H1N1)"                      # what the editor was opened on, for the reader
    tree = "h1/report@0123456789abcdef"

    [[label_pins]]                          # the label on the MRCA of first and last, its
    first = "EPI_ISL_..."                   # box's top-left at the node + (dx, dy) x the
    last = "EPI_ISL_..."                    # page height (figure.LabelPin)
    text = "K2N A6S"
    dx = -0.0123
    dy = 0.0045
    reason = "clear of the vaccine strain name"
    by = "AB"
    date = "2026-10-31"

    [[hide_labels]]                         # not printed (aa_labels.LabelHide)
    first = "EPI_ISL_..."
    last = "EPI_ISL_..."
    text = "L7F"
    reason = "the clade bracket already says it"
    by = "AB"
    date = "2026-10-31"
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path

from af.util.config import load_config

from .aa_labels import LabelHide
from .figure import LabelPin


class HandLabelsError(ValueError):
    """A hand labels row without a reason, a name or a date (design rule 9)."""


@dataclass(frozen=True)
class HandPin:
    first: str
    last: str
    text: str
    dx: float
    dy: float
    reason: str
    by: str
    date: str


@dataclass(frozen=True)
class HandHide:
    first: str
    last: str
    text: str
    reason: str
    by: str
    date: str


@dataclass(frozen=True)
class HandLabels:
    figure: str = ""
    tree: str = ""
    label_pins: list[HandPin] = field(default_factory=list)
    hide_labels: list[HandHide] = field(default_factory=list)


def load_hand_labels(path: Path) -> tuple[list[LabelPin], list[LabelHide]]:
    """The file's pins and hides, each marked as done by hand, its reason ending with who and
    when. A row missing any of them, or with a date that is not a date, refuses the file."""
    rows = load_config(path, HandLabels)
    problems = [
        f"{kind} {r.text}: {problem}"
        for kind, group in (("pin", rows.label_pins), ("hide", rows.hide_labels))
        for r in group
        for problem in _problems(r)
    ]
    if problems:
        listed = "\n  ".join(problems)
        raise HandLabelsError(f"{path}: {len(problems)} row(s) refused:\n  {listed}")
    pins = [
        LabelPin(p.first, p.last, p.text, p.dx, p.dy, _reason(p), hand=True)
        for p in rows.label_pins
    ]
    hides = [LabelHide(h.first, h.last, h.text, _reason(h), hand=True) for h in rows.hide_labels]
    return pins, hides


def _problems(row: HandPin | HandHide) -> list[str]:
    out = [f"no {name}" for name in ("reason", "by") if not getattr(row, name).strip()]
    try:
        datetime.date.fromisoformat(row.date)
    except ValueError:
        out.append(f"date {row.date!r} is not YYYY-MM-DD")
    return out


def _reason(row: HandPin | HandHide) -> str:
    return f"{row.reason.strip()} ({row.by.strip()}, {row.date})"
