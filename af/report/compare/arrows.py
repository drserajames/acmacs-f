"""Procrustes arrows: how far each point of a map moved between two builds of it.

Sarah, 30 Oct 2026: "Please make a variant of the side by side with procrustes arrows (this has been
implemented previously in ae)". ae and AD drew the primary map with an arrow from each point to the
secondary map's position after a Procrustes fit, leaving out arrows shorter than 0.3 units
(ae.map_draw export_procrustes; AD make_pc). Here, per map and window:

1. the two figures side by side, at their own sizes, with the pixel measure
   (af.report.compare.pixels);
2. the new figure as the report draws it, with an arrow for every point drawn in both that moved
   more than :data:`THRESHOLD`: from where the reference puts it, after the reference is fitted
   onto the new map, to where the new map draws it; red above :data:`FAR`;
3. the numbers.

Points are matched and fitted exactly as the report comparison does it (af.report.compare.maps:
identity matching; rotation, reflection and translation, no scaling). The fit here moves the
reference onto the new map, where COMPARISON fits the new map onto the reference: a rigid fit and
its inverse leave every point the same distance from its partner, so the arrows' lengths and RMSD
are COMPARISON's (checked on every pair). The orientation printed is COMPARISON's own, the bulk's.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.report.compare import maps

THRESHOLD = 0.3  # units: shorter moves draw no arrow (ae/AD's procrustes threshold)
FAR = 1.0  # units: longer moves are drawn red and counted
LARGEST = 20  # moves listed on the numbers page


class ArrowsError(ValueError):
    """Two figures cannot be compared."""


@dataclass(frozen=True)
class MapPair:
    """One map slot in two builds. ``ref_pdf`` is None where the reference has I7s only."""

    slot: str
    new_i7: Path
    ref_i7: Path
    new_pdf: Path
    ref_pdf: Path | None


@dataclass(frozen=True)
class Move:
    """One point drawn in both maps: where the new map draws it, and where the reference puts it
    after the fit (both in the new map's displayed coordinates)."""

    label: str
    kind: str  # "antigen" or "serum"
    vaccine: bool
    new_xy: tuple[float, float]
    ref_xy: tuple[float, float]
    distance: float


@dataclass(frozen=True)
class Arrows:
    slot: str
    moves: tuple[Move, ...]
    fit: maps.Fit
    only_ref: int  # antigens and sera drawn in the reference only
    only_new: int
    comparison: dict[
        str, Any
    ]  # COMPARISON's procrustes block for this pair (af.report.compare.maps)

    @property
    def antigens(self) -> int:
        return sum(1 for m in self.moves if m.kind == "antigen")

    def count(self, above: float, kind: str | None = None) -> int:
        return sum(1 for m in self.moves if m.distance > above and kind in (None, m.kind))

    def quantile(self, q: float) -> float:
        d = sorted(m.distance for m in self.moves)
        return d[min(len(d) - 1, int(round(q * (len(d) - 1))))]


def _label(point: maps.Point) -> str:
    extra = point.get("passage_class") or point.get("serum_id") or ""
    return f"{point['name']} {extra}".strip()


def measure(ref: dict[str, Any], new: dict[str, Any], slot: str, how: str = "identity") -> Arrows:
    """Match the two maps' points and fit the reference onto the new map."""
    antigens, sera = maps.matched(ref, new, how)
    pairs = [(r, n, "antigen") for _, r, n in antigens["_pairs"]]
    pairs += [(r, n, "serum") for _, r, n in sera["_pairs"]]
    if len(pairs) < 3:
        raise ArrowsError(f"{slot}: {len(pairs)} points drawn in both maps; a fit needs 3")
    fit = maps.procrustes(
        [tuple(n["xy"]) for _, n, _ in pairs], [tuple(r["xy"]) for r, _, _ in pairs]
    )
    moves = tuple(
        Move(
            _label(n),
            kind,
            bool(n.get("vaccine") or r.get("vaccine")),
            (float(n["xy"][0]), float(n["xy"][1])),
            fitted,
            d,
        )  # fmt: skip
        for (r, n, kind), fitted, d in zip(pairs, fit.fitted, fit.distances, strict=True)
    )
    only_ref = antigens["only_ref"] + sera["only_ref"]
    only_new = antigens["only_new"] + sera["only_new"]
    comparison = maps.compare(ref, new, how)["procrustes"]
    if not math.isclose(comparison["rmsd"], fit.rmsd, rel_tol=1e-9, abs_tol=1e-12):
        raise ArrowsError(
            f"{slot}: RMSD {fit.rmsd!r} here but {comparison['rmsd']!r} in the comparison"
        )
    return Arrows(slot, moves, fit, only_ref, only_new, comparison)


def overlay(arrows: Arrows, viewport: list[float], size: tuple[float, float], out_pdf: Path) -> int:
    """A transparent page of ``size`` (pt) with the arrows, to lay over the new figure: map
    coordinates go to the page exactly as the figure placed them (af.map.viewport.Frame.page,
    from the I7's viewport). Returns the number of arrows drawn."""
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import numpy as np

    from af.map.viewport import Frame

    x, y, w, h = viewport
    frame = Frame(x, y, w, None if math.isclose(w, h) else h)
    shown = [m for m in arrows.moves if m.distance > THRESHOLD]
    fig = plt.figure(figsize=(size[0] / 72, size[1] / 72), dpi=72)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)  # the page grows downward
    ax.axis("off")
    if shown:
        heads = frame.page(np.array([m.new_xy for m in shown]))
        tails = frame.page(np.array([m.ref_xy for m in shown]))
        for m, head, tail in zip(shown, heads, tails, strict=True):
            colour = "#d62728" if m.distance > FAR else "black"
            ax.annotate(
                "", xy=tuple(head), xytext=tuple(tail), annotation_clip=False,
                arrowprops={"arrowstyle": "-|>,head_length=0.6,head_width=0.3", "lw": 1.0,
                            "color": colour, "shrinkA": 0, "shrinkB": 0},
            )  # fmt: skip
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, transparent=True)
    plt.close(fig)
    return len(shown)


def headline(a: Arrows) -> str:
    """The comparison in one sentence, for the numbers page and the document's first page."""
    return (
        f"{len(a.moves):,} points drawn in both maps ({a.antigens:,} antigens, "
        f"{len(a.moves) - a.antigens:,} sera) sit {a.fit.rmsd:.3f} units apart (RMSD) after the "
        f"fit; {a.count(FAR):,} moved more than {FAR:g} unit and {a.count(THRESHOLD):,} more than "
        f"{THRESHOLD:g}"
    )


def orientation(a: Arrows) -> str:
    """The orientation as COMPARISON reports it: the bulk's (points within 1 unit), of the new
    map against the reference, or why there is none."""
    p = a.comparison
    none = maps.orientation_text(p)
    if none:
        return none
    reflection = "with reflection" if p["reflected"] else "no reflection"
    return f"bulk orientation {p['rotation_deg']:+.1f} degrees, {reflection}"


def summary(a: Arrows) -> dict[str, Any]:
    """The numbers, as the document prints them and its JSON records them."""
    largest = sorted(a.moves, key=lambda m: -m.distance)[:LARGEST]
    return {
        "slot": a.slot,
        "matched": len(a.moves), "antigens": a.antigens, "sera": len(a.moves) - a.antigens,
        "only_ref": a.only_ref, "only_new": a.only_new,
        "rmsd": a.fit.rmsd, "median": a.quantile(0.5), "p95": a.quantile(0.95),
        "max": max(m.distance for m in a.moves),
        "orientation": orientation(a),
        "rotation_deg": a.comparison["rotation_deg"], "reflected": a.comparison["reflected"],
        f"moved_gt_{THRESHOLD:g}": a.count(THRESHOLD),
        f"moved_gt_{FAR:g}": {"antigens": a.count(FAR, "antigen"), "sera": a.count(FAR, "serum")},
        "vaccines": [{"label": m.label, "distance": m.distance} for m in a.moves if m.vaccine],
        "largest": [{"label": m.label, "kind": m.kind, "distance": m.distance} for m in largest],
    }  # fmt: skip


def read_pair(pair: MapPair, how: str = "identity") -> Arrows:
    ref = json.loads(pair.ref_i7.read_text())
    new = json.loads(pair.new_i7.read_text())
    if new.get("kind") != "map" or ref.get("kind") != "map":
        raise ArrowsError(f"{pair.slot}: not a map figure on both sides")
    return measure(ref, new, pair.slot, how)
