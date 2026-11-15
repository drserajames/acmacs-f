"""Place a scene's labels as ae does: candidates, pair matrices, the seeded search; then report.

The report is ae's (residual conflicts and its metrics) computed with the same predicates the
search avoided, so what was optimised is what is counted.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .candidates import grids, make_candidates
from .measure import metrics, residuals, warning
from .pairs import build_pairs
from .predicates import Placed, consts
from .scene import Cands, Options, Scene
from .search import search_with_report


@dataclass
class Result:
    placed: list[Placed]  # per scene label: (label, text box, attach point)
    base: list[float]  # each label's own cost at its chosen candidate
    tiers: list[int]
    report: dict[str, Any] = field(default_factory=dict)


def place(scene: Scene, opts: Options) -> Result:
    t0 = time.perf_counter()
    cands = make_candidates(scene, opts)
    t1 = time.perf_counter()
    pairs = build_pairs(scene, cands, opts)
    t2 = time.perf_counter()
    choice, searched = search_with_report(scene.labels, cands, pairs, opts)
    t3 = time.perf_counter()
    placed = [_placed(scene, cands, i, int(c)) for i, c in enumerate(choice)]
    k = consts(scene, opts)
    tiers = [int(cands[i].tier[c]) for i, c in enumerate(choice)]
    conflicts = residuals(placed, k, tiers=tiers)
    both = grids(scene, opts)[2]  # tree and obstacles: what a leader's cells are counted on

    def cells(a: tuple[float, float], b: tuple[float, float]) -> int:
        return both.leader_ink_cells(a[0], a[1], b[0], b[1])

    report: dict[str, Any] = {
        "metrics": metrics(placed, scene, k, leader_cells=cells, tiers=tiers),
        "residual_conflicts": [str(c) for c in conflicts],
        "warning": warning(conflicts),
        "search": asdict(searched),
        "candidates": sum(len(c) for c in cands),
        "seconds": {
            "candidates": round(t1 - t0, 2),
            "pairs": round(t2 - t1, 2),
            "search": round(t3 - t2, 2),
        },
    }
    base = [float(cands[i].base[c]) for i, c in enumerate(choice)]
    return Result(placed, base, tiers, report)


def _placed(scene: Scene, cands: list[Cands], i: int, c: int) -> Placed:
    return (scene.labels[i], cands[i].box(c), (float(cands[i].cx[c]), float(cands[i].cy[c])))
