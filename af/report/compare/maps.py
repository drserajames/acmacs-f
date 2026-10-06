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

Points are matched by designation (``id``); by name + passage class (sera: name + serum id),
exactly (``name``) or with the name's punctuation and spacing dropped (``loose``); or by
**identity** (``identity``, the default): isolate number, year, passage class and isolation
date (sera: isolate, year, serum id), with the location left out. af keeps each lab's spelling
of a place (DECISIONS 24 Sep) where ae rewrote it, so one virus is KIEV on one side and KYIV
on the other, RAS AL KHAIMAH CITY and RAK; the isolate, year, passage and date agree. A point
missing one of those, or whose identity is shared by two points on either side, is matched by
its ``loose`` key instead, on both sides, and counted. Keys that occur twice on one side are
dropped and counted, never merged.

Serum ids are compared without a leading token that is the map's own lab (``CRICK F01/99``
and ``F01/99`` are one serum): af's merged maps prefix every serum id with the lab, where the
ae round keeps the lab's own id, which for some labs already begins with the lab's name. Only
that one token is dropped, on both sides, so a serum id of another lab's form is untouched; the
counts dropped are reported (``sera.serum_id_lab_dropped``) so a matched pair is not read as
two identical ids.

Then, by Sarah's rulings on serum ids (DECISIONS, 6 Oct), also on both sides and also counted
(``sera.serum_id_normalised``): letter case is ignored; a trailing bleed-day suffix (``-14D``) is
ignored, the id itself keeping it ("Keep the suffix, but merge without the suffix"); and an id
that only says it is unknown (``UNKNOWN-<passage>``) is no id, so the serum is matched as one
with none. Qualifier words (a source or kind written in the id) are part of the identity and are
NOT ignored: "Not all NIB/CDC sera will be the same".
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Point = dict[str, Any]
XY = tuple[float, float]


MATCH_MODES = ("id", "name", "loose", "identity")


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


def identity_key(point: Point) -> str | None:
    """Isolate/year + passage class + isolation date (sera: isolate/year + serum id), or None.

    The location is left out (see the module docstring); the type prefix and anything after
    the year (a lab's reassortant suffix) too, so the date or the serum id must tell two
    preparations of one isolate apart. None when the name is not TYPE/.../ISOLATE/YEAR or the
    date (antigens) or serum id (sera) is missing: such a point is matched by its name.
    """
    parts = str(point["name"]).split("/")
    if len(parts) < 4:
        return None
    isolate, year = spelling_key(parts[-2]), parts[-1].strip()[:4]
    if not isolate or not year.isdigit():
        return None
    if isolate.isdigit():
        isolate = isolate.lstrip("0") or "0"  # 01 and 1 are one isolate number
    if point.get("serum_id"):
        return f"#{isolate}/{year}|{point['serum_id']}"
    if not point.get("date"):
        return None
    return f"#{isolate}/{year}|{point['passage_class'] or 'none'}|{point['date']}"


def point_key(point: Point, how: str) -> str:
    """Matching key: ``id``; ``name`` (name + passage class, or + serum id); ``loose`` (``name``
    with the name spelling-normalised by :func:`spelling_key`). ``identity`` depends on the
    other points of both sides, so it is not a per-point key: see :func:`keyed_points`."""
    if how not in MATCH_MODES or how == "identity":
        raise ValueError(f"match mode {how!r} not one of {MATCH_MODES[:3]}")
    if how == "id":
        return str(point["id"])
    name = spelling_key(point["name"]) if how == "loose" else point["name"]
    if point.get("serum_id"):  # a serum id identifies the serum; passage class may be unset
        return f"{name}|{point['serum_id']}"
    # A null passage class (not a passage: specimen ids, blanks) keys as "none" on both sides.
    return f"{name}|{point['passage_class'] or 'none'}"


def _label(point: Point) -> str:
    """How a point is listed for a reader: its own spelling, whatever key matched it."""
    if DRAWN_SERUM_ID in point:
        return f"{point['name']}|{point[DRAWN_SERUM_ID]}"
    return point_key(point, "name")


DRAWN_SERUM_ID = "_serum_id_as_drawn"  # the id before the lab token was dropped, for listing


def map_lab(doc: dict[str, Any]) -> str | None:
    """The lab a map figure was built for: the first part of its chain dataset, upper case.

    af's figures record the chain they were drawn from (``provenance.inputs.chain.dataset``,
    e.g. ``<lab>/<table group>/merged``); a figure that records none has no lab to drop.
    """
    chain = doc.get("provenance", {}).get("inputs", {}).get("chain")
    dataset = chain.get("dataset") if isinstance(chain, dict) else None
    return str(dataset).split("/")[0].upper() if dataset else None


def drop_lab_token(points: Sequence[Point], lab: str) -> tuple[list[Point], int]:
    """Points with a leading ``"<lab> "`` dropped from their serum ids; also how many had it."""
    prefix, out, dropped = f"{lab} ", [], 0
    for point in points:
        serum_id = point.get("serum_id")
        if serum_id and serum_id.startswith(prefix) and serum_id[len(prefix) :].strip():
            point = {**point, "serum_id": serum_id[len(prefix) :], DRAWN_SERUM_ID: serum_id}
            dropped += 1
        out.append(point)
    return out, dropped


BLEED_DAY = re.compile(r"-\d{1,3}D$")
UNKNOWN_ID = re.compile(r"^UNKNOWN-[A-Z&/]+$")


def match_serum_ids(points: Sequence[Point]) -> tuple[list[Point], dict[str, int]]:
    """Serum ids as matching reads them (see the module docstring); counts of each change."""
    counts = {"case": 0, "bleed_day_suffix": 0, "unknown_as_none": 0}
    out = []
    for point in points:
        drawn = point.get("serum_id")
        if not drawn:
            out.append(point)
            continue
        serum_id: str | None = drawn.upper()
        counts["case"] += serum_id != drawn
        if serum_id and BLEED_DAY.search(serum_id):
            serum_id = BLEED_DAY.sub("", serum_id)
            counts["bleed_day_suffix"] += 1
        if serum_id and UNKNOWN_ID.match(serum_id):
            serum_id = None
            counts["unknown_as_none"] += 1
        if serum_id != drawn:
            point = {**point, "serum_id": serum_id,
                     DRAWN_SERUM_ID: point.get(DRAWN_SERUM_ID, drawn)}  # fmt: skip
        out.append(point)
    return out, counts


SERUM_RULE_WORDS = {"case": "letter case", "bleed_day_suffix": "a bleed-day suffix",
                    "unknown_as_none": "an UNKNOWN id"}  # fmt: skip


def serum_rules_text(sera: dict[str, Any]) -> str:
    """What serum-id matching ignored, with counts per side, or "" when it ignored nothing."""
    rules = sera.get("serum_id_normalised", {})
    parts = [
        f"{words} ({rules['new'][k]} af, {rules['ref'][k]} reference)"
        for k, words in SERUM_RULE_WORDS.items()
        if rules and (rules["new"][k] or rules["ref"][k])
    ]
    return "ignoring " + ", ".join(parts) if parts else ""


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
    return _unique([(point_key(p, how), p) for p in points])


def _unique(keyed: Sequence[tuple[str, Point]]) -> tuple[dict[str, Point], int]:
    counts = Counter(k for k, _ in keyed)
    return {k: p for k, p in keyed if counts[k] == 1}, sum(n for n in counts.values() if n > 1)


def keyed_points(
    ref: Sequence[Point], new: Sequence[Point], how: str
) -> tuple[list[tuple[str, Point]], list[tuple[str, Point]], dict[str, int]]:
    """Both sides' (key, point), and for ``identity`` how many points fell back to ``loose``.

    Identity matching is two passes. First by identity; then every point identity did not
    pair (no identity, an identity shared by two points on EITHER side, or no partner with the
    same identity) is keyed by its ``loose`` name, on both sides, so the name can still pair
    it. A point is never keyed one way on one side and another way on the other, so a shared
    identity cannot pair the wrong partners, and identity never loses a pair the name finds:
    the same name with another date on the two sides is still one virus to the name pass.
    """
    if how != "identity":
        return [(point_key(p, how), p) for p in ref], [(point_key(p, how), p) for p in new], {}
    ids = {"ref": [identity_key(p) for p in ref], "new": [identity_key(p) for p in new]}
    counts = {side: Counter(k for k in keys if k is not None) for side, keys in ids.items()}
    shared = {k for c in counts.values() for k, n in c.items() if n > 1}
    paired = (counts["ref"].keys() & counts["new"].keys()) - shared
    fallback = {"no_identity": 0, "shared_identity": 0, "unpaired_identity": 0}

    def key(point: Point, ident: str | None) -> str:
        if ident in paired:
            return str(ident)
        reason = "no_identity" if ident is None else (
            "shared_identity" if ident in shared else "unpaired_identity")  # fmt: skip
        fallback[reason] += 1
        return point_key(point, "loose")

    return (
        [(key(p, k), p) for p, k in zip(ref, ids["ref"], strict=True)],
        [(key(p, k), p) for p, k in zip(new, ids["new"], strict=True)],
        fallback,
    )


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
    rk, nk, fallback = keyed_points([p for p in ref if drawn(p)], [p for p in new if drawn(p)], how)
    (ri, rdup), (ni, ndup) = _unique(rk), _unique(nk)
    common = sorted(ri.keys() & ni.keys())
    in_frame = {k: (bool(ri[k]["in_viewport"]), bool(ni[k]["in_viewport"])) for k in common}
    out: dict[str, Any] = {
        "ref_shown": len(ri), "new_shown": len(ni), "common": len(common),
        "only_ref": len(ri.keys() - ni.keys()), "only_new": len(ni.keys() - ri.keys()),
        "ambiguous_dropped": {"ref": rdup, "new": ndup},
        "identity_fallback": fallback,
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
        # Colour coverage: drawn on both sides, in a clade colour on one only, and not greyed by the
        # time window on either (that is greyed_agreement's). "Lost" is Sarah's Q81 switch measure
        # for a map changing its colour source: the reference coloured it, af leaves it uncoloured.
        in_window = [k for k in common if not ri[k].get("greyed") and not ni[k].get("greyed")]
        lost = [k for k in in_window if ri[k]["clade"] and not ni[k]["clade"]]
        gained = [k for k in in_window if ni[k]["clade"] and not ri[k]["clade"]]
        ref_coloured = sum(1 for k in in_window if ri[k]["clade"])
        out["colour"] = {
            "ref_coloured": ref_coloured, "lost": len(lost), "gained": len(gained),
            "lost_frac": len(lost) / ref_coloured if ref_coloured else float("nan"),
        }  # fmt: skip
        out["colour_only_ref_keys"] = sorted(_label(ri[k]) for k in lost)
        out["colour_only_new_keys"] = sorted(_label(ni[k]) for k in gained)
        grey = [bool(ri[k].get("greyed")) == bool(ni[k].get("greyed")) for k in common]
        out["greyed_agreement"] = sum(grey) / len(grey) if grey else float("nan")
        # Vaccine marks: an antigen drawn on both sides but marked as a vaccine on one only (e.g.
        # the wrong preparation of a vaccine strain marked). Listed, like one-sided points.
        out["vaccine_only_ref_keys"] = sorted(
            _label(ri[k]) for k in common if ri[k].get("vaccine") and not ni[k].get("vaccine")
        )
        out["vaccine_only_new_keys"] = sorted(
            _label(ni[k]) for k in common if ni[k].get("vaccine") and not ri[k].get("vaccine")
        )
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
    if rm.get("y_axis") != nm.get("y_axis"):
        # A y-down map against a y-up one would fit through the mirror and report every map as
        # reflected and its rotation mirrored: refuse, so the reference is re-extracted instead.
        raise ValueError(
            f"{ref.get('title')!r} vs {new.get('title')!r}: y axes differ "
            f"(reference {rm.get('y_axis') or 'unstated'}, new {nm.get('y_axis') or 'unstated'})"
        )
    antigens = _group(rm["antigens"], nm["antigens"], how, clades=True)
    ref_sera, new_sera, lab = rm["sera"], nm["sera"], map_lab(new)
    if lab:
        ref_sera, ref_dropped = drop_lab_token(ref_sera, lab)
        new_sera, new_dropped = drop_lab_token(new_sera, lab)
    ref_sera, ref_rules = match_serum_ids(ref_sera)
    new_sera, new_rules = match_serum_ids(new_sera)
    sera = _group(ref_sera, new_sera, how, clades=False)
    if lab:
        sera["serum_id_lab_dropped"] = {"lab": lab, "ref": ref_dropped, "new": new_dropped}
    sera["serum_id_normalised"] = {"ref": ref_rules, "new": new_rules}
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
                {"key": _label(pairs[i][1]), "distance": d[i]}
                for i in sorted(range(len(d)), key=lambda i: -d[i]) if d[i] > 2
            ][:50],
            "rmsd_antigens": math.sqrt(sum(x * x for x in d[:n_ag]) / n_ag) if n_ag else None,
            # The same layout on both sides (e.g. a stand-in built from the reference's own chart):
            # displacement then compares a map with itself and tests nothing.
            "identical_layout": max(d) < IDENTICAL_MAX,
        }  # fmt: skip
        out["clade_centroids"] = clade_centroids(ag_pairs)
    return out
