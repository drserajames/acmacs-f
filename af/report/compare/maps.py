"""Same-science comparison of two antigenic-map figures, from their I7 documents.

"Same science" (DECISIONS, 24 Sep 2026): the same viruses and sera per map, the same clade
groupings, and the same antigenic relationships. Each is measured separately, so a failure
says which one it is:

- points shown: Jaccard of the points in the map (shown, with coordinates). Points inside the
  frame on one side only are listed separately: the frame is presentation, not content;
- clade grouping: adjusted Rand index, which ignores what the clades are called;
- geometry: per-point displacement after a Procrustes fit (rotation, reflection and
  translation, no scaling, because map units are log2 fold). The rotation and reflection are
  reported separately, since a reader sees a turned map as different. Orientation comes from
  the bulk (points moved more than 1 unit left out; see :func:`bulk_orientation`);
- relationships: the change in clade-to-clade centroid distances. This catches a whole clade
  moving, which a point percentile misses when the clade is small.

Points are matched by designation (``id``) or, when the two sides spell passages differently,
by name + passage class (sera: name + serum id). Keys that occur twice on one side are dropped
and counted, never merged.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Point = dict[str, Any]
XY = tuple[float, float]


MATCH_MODES = ("id", "name", "loose")


def spelling_key(name: str) -> str:
    """Name with punctuation and spacing removed, upper case; letters of every script kept.

    af keeps each lab's spelling of a location (DECISIONS 24 Sep: no locdb rewrites), where ae
    rewrote names, so the same virus is COTE D'IVOIRE on one side and COTE DIVOIRE on the other,
    or LASERENA and LA SERENA. Letters and digits of any script stay: some CNIC places are known
    only by their Chinese name, and deleting those characters would merge distinct places.
    Two different strains that differ only in punctuation or spacing would collide; they are
    then dropped as ambiguous and counted, never merged.
    """
    return "".join(ch for ch in name.upper() if ch.isalnum())


def point_key(point: Point, how: str) -> str:
    """Matching key: ``id``; ``name`` (name + passage class, or + serum id); ``loose`` (``name``
    with the name spelling-normalised by :func:`spelling_key`)."""
    if how not in MATCH_MODES:
        raise ValueError(f"match mode {how!r} not one of {MATCH_MODES}")
    if how == "id":
        return str(point["id"])
    name = spelling_key(point["name"]) if how == "loose" else point["name"]
    if point.get("serum_id"):  # a serum id identifies the serum; passage class may be unset
        return f"{name}|{point['serum_id']}"
    # A null passage class (not a passage: specimen ids, blanks) keys as "none" on both sides.
    return f"{name}|{point['passage_class'] or 'none'}"


def _label(point: Point) -> str:
    """How a point is listed for a reader: its own spelling, whatever key matched it."""
    return point_key(point, "name")


def normalise_key(key: str, how: str) -> str:
    """A ``NAME|rest`` key written by hand (excused-point files), keyed as ``point_key`` would."""
    if how != "loose":
        return key
    name, _, rest = key.rpartition("|")
    return f"{spelling_key(name)}|{rest}"


def drawn(point: Point) -> bool:
    """In the map: shown and has coordinates. Whether it falls inside the frame is presentation,
    compared separately, so a frame that cuts a point off is not reported as a missing virus."""
    return bool(point["shown"] and point["xy"])


def index_points(points: Sequence[Point], how: str) -> tuple[dict[str, Point], int]:
    """Key -> point for keys that occur once; also the number of points dropped as ambiguous."""
    counts = Counter(point_key(p, how) for p in points)
    unique = {point_key(p, how): p for p in points if counts[point_key(p, how)] == 1}
    return unique, sum(n for n in counts.values() if n > 1)


@dataclass(frozen=True)
class Fit:
    distances: list[float]
    rmsd: float
    rotation_deg: float
    reflected: bool


def _centre(points: Sequence[XY]) -> tuple[list[XY], XY]:
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return [(x - cx, y - cy) for x, y in points], (cx, cy)


def _rotation_fit(a: Sequence[XY], b: Sequence[XY]) -> tuple[float, list[float]]:
    """Best rotation of centred ``b`` onto centred ``a`` (closed form in 2-D) and distances."""
    s_cross = sum(bx * ay - by * ax for (ax, ay), (bx, by) in zip(a, b, strict=True))
    s_dot = sum(bx * ax + by * ay for (ax, ay), (bx, by) in zip(a, b, strict=True))
    theta = math.atan2(s_cross, s_dot)
    c, s = math.cos(theta), math.sin(theta)
    dist = [
        math.hypot(bx * c - by * s - ax, bx * s + by * c - ay)
        for (ax, ay), (bx, by) in zip(a, b, strict=True)
    ]
    return theta, dist


def procrustes(a: Sequence[XY], b: Sequence[XY]) -> Fit:
    """Fit ``b`` onto ``a`` by rotation, optional reflection (y -> -y) and translation."""
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("procrustes needs two equal lists of at least 3 points")
    a0, _ = _centre(a)
    b0, _ = _centre(b)
    theta, dist = _rotation_fit(a0, b0)
    theta_r, dist_r = _rotation_fit(a0, [(x, -y) for x, y in b0])
    reflected = sum(d * d for d in dist_r) < sum(d * d for d in dist) - 1e-12
    if reflected:
        theta, dist = theta_r, dist_r
    rmsd = math.sqrt(sum(d * d for d in dist) / len(dist))
    return Fit(dist, rmsd, math.degrees(theta), reflected)


# Largest single displacement (units) below which two layouts count as the same up to rotation and
# translation. Measured on the Sep 2026 stand-ins: coordinates written to ~4 decimals leave
# max 7.7e-5; a genuinely recomputed layout moved at least one point 9.8e-3. 1e-3 sits between.
IDENTICAL_MAX = 1e-3
BULK_MOVE_MAX = 1.0  # units: points moved further than this are left out of the orientation fit


def bulk_orientation(a: Sequence[XY], b: Sequence[XY], fit: Fit) -> Fit:
    """Orientation of the bulk of the map: refit without the points the full fit moved > 1 unit.

    A least-squares fit over every point rotates to compromise with the points that genuinely
    moved (a refused guard, a dropped hand move), so a map with 9% of its points moved can read as
    turned 4.7 degrees when its bulk is not turned at all. The reader sees the bulk. If fewer than
    half the points are within the cut, the full fit is used (there is no stable bulk to orient by).
    """
    keep = [i for i, dist in enumerate(fit.distances) if dist <= BULK_MOVE_MAX]
    if len(keep) < max(3, len(a) // 2):
        return fit
    return procrustes([a[i] for i in keep], [b[i] for i in keep])


def adjusted_rand(x: Sequence[str], y: Sequence[str]) -> float:
    """Adjusted Rand index of two labelings: 1 = the same grouping, whatever the labels."""
    n = len(x)
    if n < 2:
        return float("nan")

    def pairs(k: int) -> float:
        return k * (k - 1) / 2

    s_ij = sum(pairs(v) for v in Counter(zip(x, y, strict=True)).values())
    s_a = sum(pairs(v) for v in Counter(x).values())
    s_b = sum(pairs(v) for v in Counter(y).values())
    expected = s_a * s_b / pairs(n)
    top = (s_a + s_b) / 2
    return 1.0 if top == expected else (s_ij - expected) / (top - expected)


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), ``q`` in 0..100."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _coloured(point: Point) -> bool:
    """Coloured by clade: has a clade label and is not greyed out as outside the window."""
    return bool(point["clade"]) and not point.get("greyed")


def _group(ref: list[Point], new: list[Point], how: str, clades: bool) -> dict[str, Any]:
    ri, rdup = index_points([p for p in ref if drawn(p)], how)
    ni, ndup = index_points([p for p in new if drawn(p)], how)
    common = sorted(ri.keys() & ni.keys())
    in_frame = {k: (bool(ri[k]["in_viewport"]), bool(ni[k]["in_viewport"])) for k in common}
    out: dict[str, Any] = {
        "ref_shown": len(ri), "new_shown": len(ni), "common": len(common),
        "only_ref": len(ri.keys() - ni.keys()), "only_new": len(ni.keys() - ri.keys()),
        "ambiguous_dropped": {"ref": rdup, "new": ndup},
        "jaccard": len(common) / max(1, len(ri.keys() | ni.keys())),
        # Full lists: every one-sided point must be listed somewhere a person reads.
        "only_ref_keys": sorted(_label(ri[k]) for k in ri.keys() - ni.keys()),
        "only_new_keys": sorted(_label(ni[k]) for k in ni.keys() - ri.keys()),
        # Frame: drawn on both sides, inside the frame on one side only.
        "in_frame_only_ref_keys": sorted(
            _label(ri[k]) for k, (r, n) in in_frame.items() if r and not n
        ),
        "in_frame_only_new_keys": sorted(
            _label(ni[k]) for k, (r, n) in in_frame.items() if n and not r
        ),
    }  # fmt: skip
    if clades:
        coloured = [k for k in common if _coloured(ri[k]) and _coloured(ni[k])]
        lr = [str(ri[k]["clade"]) for k in coloured]
        ln = [str(ni[k]["clade"]) for k in coloured]
        disagree = Counter((a, b) for a, b in zip(lr, ln, strict=True) if a != b)
        out["clade"] = {
            "compared": len(coloured),
            "label_agreement": (
                sum(a == b for a, b in zip(lr, ln, strict=True)) / len(coloured)
                if coloured else float("nan")
            ),
            "adjusted_rand": adjusted_rand(lr, ln),
            "top_disagreements": [f"{a} -> {b}: {n}" for (a, b), n in disagree.most_common(5)],
        }  # fmt: skip
        grey = [bool(ri[k].get("greyed")) == bool(ni[k].get("greyed")) for k in common]
        out["greyed_agreement"] = sum(grey) / len(grey) if grey else float("nan")
    out["_pairs"] = [(k, ri[k], ni[k]) for k in common]
    return out


def clade_centroids(pairs: list[tuple[str, Point, Point]], min_points: int = 10) -> dict[str, Any]:
    """Change in clade-to-clade centroid distances, using the reference's clade labels.

    Using one side's labels for both means relabelling cannot hide a move. Clades with fewer
    than ``min_points`` common coloured points are left out and counted.
    """
    groups: dict[str, list[tuple[Point, Point]]] = {}
    for _, r, n in pairs:
        if r["clade"] and not r.get("greyed"):
            groups.setdefault(str(r["clade"]), []).append((r, n))
    names = sorted(c for c, members in groups.items() if len(members) >= min_points)
    result: dict[str, Any] = {"clades": len(names), "left_out": len(groups) - len(names)}
    if len(names) < 2:
        return result

    def centroid(members: list[tuple[Point, Point]], side: int) -> XY:
        xs = [m[side]["xy"] for m in members]
        return (sum(p[0] for p in xs) / len(xs), sum(p[1] for p in xs) / len(xs))

    cr = {c: centroid(groups[c], 0) for c in names}
    cn = {c: centroid(groups[c], 1) for c in names}
    worst = (0.0, "", 0.0, 0.0)
    diffs = []
    for i, c1 in enumerate(names):
        for c2 in names[i + 1 :]:
            dr = math.dist(cr[c1], cr[c2])
            dn = math.dist(cn[c1], cn[c2])
            diffs.append(abs(dr - dn))
            if abs(dr - dn) > worst[0]:
                worst = (abs(dr - dn), f"{c1} / {c2}", dr, dn)
    result.update(
        max_abs_diff=worst[0], mean_abs_diff=sum(diffs) / len(diffs), worst_pair=worst[1],
        worst_pair_ref_new=[worst[2], worst[3]],
    )  # fmt: skip
    return result


def compare(ref: dict[str, Any], new: dict[str, Any], how: str = "name") -> dict[str, Any]:
    """Compare two map I7 documents; see the module docstring for what each number means."""
    rm, nm = ref["map"], new["map"]
    antigens = _group(rm["antigens"], nm["antigens"], how, clades=True)
    sera = _group(rm["sera"], nm["sera"], how, clades=False)
    ag_pairs, sr_pairs = antigens.pop("_pairs"), sera.pop("_pairs")
    out: dict[str, Any] = {
        "ref": ref["title"], "new": new["title"], "match": how,
        "antigens": antigens, "sera": sera,
    }  # fmt: skip
    pairs = ag_pairs + sr_pairs
    if len(pairs) >= 3:
        a_xy = [tuple(p[1]["xy"]) for p in pairs]
        b_xy = [tuple(p[2]["xy"]) for p in pairs]
        fit = procrustes(a_xy, b_xy)
        bulk = bulk_orientation(a_xy, b_xy, fit)
        d, n_ag = fit.distances, len(ag_pairs)
        out["procrustes"] = {
            "points": len(d), "rmsd": fit.rmsd,
            "rotation_deg": bulk.rotation_deg, "reflected": bulk.reflected,
            "rotation_deg_all_points": fit.rotation_deg,
            "median": _percentile(d, 50), "p95": _percentile(d, 95), "max": max(d),
            "frac_gt_1": sum(x > 1 for x in d) / len(d),
            "frac_gt_2": sum(x > 2 for x in d) / len(d),
            "moved_gt_2": [
                {"key": pairs[i][0], "distance": d[i]}
                for i in sorted(range(len(d)), key=lambda i: -d[i]) if d[i] > 2
            ][:50],
            "rmsd_antigens": math.sqrt(sum(x * x for x in d[:n_ag]) / n_ag) if n_ag else None,
            # The same layout on both sides (e.g. a stand-in built from the reference's own chart):
            # displacement then compares a map with itself and tests nothing.
            "identical_layout": max(d) < IDENTICAL_MAX,
        }  # fmt: skip
        out["clade_centroids"] = clade_centroids(ag_pairs)
    return out
