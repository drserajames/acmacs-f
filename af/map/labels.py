"""Vaccine labels: the text, and where to put it.

Why: every lab folder carries a hand table of label offsets bound to designations, and on the Sep
2026 round 17 vaccine-tagged antigens had no row (so ae placed their labels itself) while 8 rows
matched nothing. Here the text follows one rule and the placement is computed. A hand offset can
still be given as named data where Sarah wants one.

Coordinates are page fractions: (0, 0) top-left, (1, 1) bottom-right.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from af.map.vaccines import VaccineMark, strain_name
from af.map.viewport import Box

Array = NDArray[np.float64]


def label_text(
    name: str, passage_class: str, reassortant: str, abbreviations: Mapping[str, str]
) -> str:
    """Short label ``<place>/<yy>-<class>``, as on the shipped maps (ae's rule).

    The place abbreviation comes from ``abbreviations`` (curated data, keyed by the upper-case
    location: :class:`LabelAbbreviations`, CDC's codes for US places). Without an entry:
    initials of a multi-word place, else its first two letters. The suffix is the passage word
    the caller gives; a reassortant given as such is labelled by its reassortant name. (The
    round's maps label a reassortant vaccine by its egg passage: af.map.build decides that.)
    """
    parts = strain_name(name).split("/")
    if len(parts) < 3:
        raise ValueError(f"cannot make a label from strain name {name!r}")
    place, year = parts[0], parts[-1]
    abbr = abbreviations.get(place)
    if abbr is None:
        words = [w for w in re.split(r"[ _-]+", place) if w]
        abbr = "".join(w[0] for w in words).upper() if len(words) > 1 else place[:2].title()
    suffix = (
        reassortant.strip()
        if passage_class == "reassortant" and reassortant.strip()
        else passage_class
    )
    return f"{abbr}/{year[-2:]}-{suffix}"


@dataclass(frozen=True)
class LabelAbbreviations:
    """A round's place abbreviations for vaccine labels: a TSV with ``location`` and
    ``abbreviation`` columns (``#`` lines are its header notes), named in maps.toml. Its hash
    goes into each figure's provenance (rule 5)."""

    path: Path
    sha256: str
    table: dict[str, str]

    @classmethod
    def read(cls, path: Path) -> LabelAbbreviations:
        data = path.read_bytes()
        lines = [ln for ln in data.decode().splitlines() if ln.strip() and not ln.startswith("#")]
        if not lines or lines[0].split("\t") != ["location", "abbreviation"]:
            raise ValueError(f"{path}: expected a header line 'location<TAB>abbreviation'")
        table: dict[str, str] = {}
        for n, line in enumerate(lines[1:], 2):
            cells = line.split("\t")
            if len(cells) != 2 or not all(c.strip() == c and c for c in cells):
                raise ValueError(f"{path}: row {n}: expected two non-empty cells, got {cells}")
            if cells[0] in table:
                raise ValueError(f"{path}: row {n}: {cells[0]!r} has two rows")
            table[cells[0]] = cells[1]
        return cls(path, hashlib.sha256(data).hexdigest(), table)

    def input(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class LabelRule:
    """A vaccine to label: strain name and passage class (``any`` = every class)."""

    name: str
    passage: str  # "cell", "egg", "reassortant" or "any"


def vaccines_to_label(
    marks: Sequence[VaccineMark],
    *,
    subtype_list: Sequence[LabelRule] | None,
    lab_list: Sequence[LabelRule] | None,
) -> list[VaccineMark]:
    """Which marked vaccines get a label (Sarah, 25 Sep 2026): all of them by default; a
    per-subtype list replaces that; a per lab+subtype list replaces both. The most specific list
    given wins outright (lists are not merged). Unlabelled vaccines are still drawn as vaccines.
    Every rule in the list that wins must match a marked vaccine, or it is an error.
    """
    rules = lab_list if lab_list is not None else subtype_list
    if rules is None:
        return list(marks)
    chosen: list[VaccineMark] = []
    for rule in rules:
        hits = [
            m
            for m in marks
            if strain_name(m.row.name) == strain_name(rule.name)
            and rule.passage in ("any", m.passage_class)
        ]
        if not hits:
            raise ValueError(f"vaccine label rule {rule} matches no marked vaccine")
        chosen.extend(h for h in hits if h not in chosen)
    return chosen


@dataclass(frozen=True)
class Placed:
    text: str
    anchor: tuple[float, float]
    box: tuple[float, float, float, float]  # left, top, right, bottom
    overlaps: int  # other labels or furniture overlapped (0 wanted)
    covered_points: int  # drawn points under the label


def place_labels(
    anchors: Array,
    texts: Sequence[str],
    drawn: Array,
    furniture: Sequence[Box],
    *,
    sizes: Sequence[tuple[float, float]] | None = None,
    mark_radius: float | None = None,
    char_width: float = 0.0155,
    height: float = 0.03,
    radii: Sequence[float] = (0.0275, 0.04, 0.056),
) -> list[Placed]:
    """Greedy placement, in order: try positions around each anchor at increasing distance and
    keep the one that (1) stays on the page, (2) overlaps the furniture (legend, title) least,
    (3) overlaps fewest earlier labels and marks, (4) is nearest, (5) covers fewest drawn
    points. Deterministic: no randomness, fixed order. Furniture comes first because it is drawn
    over the labels: a label half under the legend cannot be read (style 08-23).

    Nearest before fewest covered, as the shipped maps place them: a label sits just clear of
    its mark (the first radius is a marked point's 0.025 plus a little) even over other points,
    which its white halo keeps readable; ranking covered points first floated labels 30-45 u
    away (style 08-17). ``sizes`` are each label's measured (width, height); without them the
    size is estimated from ``char_width`` and ``height``. With ``mark_radius``, every anchor's
    mark (a circle of that radius) counts like furniture: a label over another vaccine's mark
    hides it, which no halo mends.
    """
    placed: list[Placed] = []
    boxes: list[tuple[float, float, float, float]] = [
        (b.left, b.top, b.right, b.bottom) for b in furniture
    ]
    n_furniture = len(boxes)
    if sizes is None:
        sizes = [(char_width * len(t), height) for t in texts]
    for (x, y), text, (w, h) in zip(anchors, texts, sizes, strict=True):
        best: tuple[tuple[float, ...], tuple[float, float, float, float], int, int] | None = None
        for r in radii:
            for step in range(12):
                a = math.radians(30 * step)
                dx, dy = r * math.cos(a), r * math.sin(a)
                left = x + dx - (w if dx < -1e-9 else 0.0 if dx > 1e-9 else w / 2)
                top = y + dy - (h if dy < -1e-9 else 0.0 if dy > 1e-9 else h / 2)
                box = (left, top, left + w, top + h)
                off = box[0] < 0 or box[1] < 0 or box[2] > 1 or box[3] > 1
                on_furniture = sum(_overlap(box, other) for other in boxes[:n_furniture])
                hits = sum(_overlap(box, other) for other in boxes[n_furniture:])
                if mark_radius is not None:
                    hits += sum(_on_circle(box, m, mark_radius) for m in anchors)
                covered = int(_inside(drawn, box).sum())
                cost = (float(off), float(on_furniture), float(hits), r, float(covered))
                if best is None or cost < best[0]:
                    best = (cost, box, on_furniture + hits, covered)
        assert best is not None
        _, box, hits, covered = best
        boxes.append(box)
        placed.append(Placed(text, (float(x), float(y)), box, hits, covered))
    return placed


def _on_circle(box: tuple[float, float, float, float], centre: Array, radius: float) -> bool:
    """Whether ``box`` reaches into the circle (nearest point of the box inside it)."""
    x, y = float(centre[0]), float(centre[1])
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return dx * dx + dy * dy < radius * radius - 1e-12


def _overlap(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _inside(points: Array, box: tuple[float, float, float, float]) -> NDArray[np.bool_]:
    if len(points) == 0:
        return np.zeros(0, bool)
    with np.errstate(invalid="ignore"):
        return (
            (points[:, 0] > box[0])
            & (points[:, 0] < box[2])
            & (points[:, 1] > box[1])
            & (points[:, 1] < box[3])
        )
