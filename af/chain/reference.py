"""A finished map measured against a reference map of the same data (e.g. a previous round's).

Sarah, 1 Oct 2026: maps are built from scratch and their differences from the reference are
DECLARED, not engineered away. Two questions with different answers, reported together:

composition  which points the two maps share: staged identity matching (exact; place
             punctuation; place respelled; passage date dropped; name only), Jaccard over
             antigens and sera.
basin        did the search find the best layout? Both stresses are measured on THIS map's
             chart: its own (published) stress, and the best of N starts relaxed from the
             reference's coordinates for every point that pairs (the rest at random). A map
             above that "seeded" stress is not the lowest-stress layout found, and the points
             placed most differently say where (search, not data: both fit the same titres).

The seeded relax is a measurement only: nothing derived from the reference enters the map.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from af.chart.model import Chart
from af.chart.procrustes import procrustes

LISTED = 50  # one-side points listed by name (all are counted)
MOVERS = 20  # points listed by how far apart they sit in the two layouts
LABS = r"^(NIID|VIDRL|CDC|CNIC|CRICK)\s+"  # af keeps the lab's serum-id prefix; others may not


def _bare(name: str) -> str:
    """The place without punctuation: 'NIIGATA C' and 'NIIGATA-C' are one place."""
    p = name.split("/")
    return "/".join([p[0], re.sub(r"[^A-Z0-9]", "", p[1].upper()), *p[2:]]) if len(p) > 2 else name


def _respelled(name: str) -> tuple[str, str]:
    """Type and everything after the place: the place itself respelled (SAINT/ST)."""
    p = name.split("/")
    return (p[0], "/".join(p[2:]))


def _undated(passage: str) -> str:
    return re.sub(r" \((\d{4}-\d\d-\d\d|\d\d/\d\d/\d{4})\)", "", passage)


def _sid(serum_id: str) -> str:
    return re.sub(LABS, "", serum_id.upper()).replace(" ", "")


def _ag(name_of: Callable[[str], Any], passage_of: Callable[[str], str] | None) -> Callable:
    """An antigen identity key: the name as `name_of` reduces it, then the passage (or not)."""

    def key(a: Any) -> tuple:
        base = (name_of(a.name), a.reassortant, tuple(a.annotations))
        return base if passage_of is None else (*base, passage_of(a.passage))

    return key


def _same(x: str) -> str:
    return x


ANTIGEN_STAGES: tuple[tuple[str, Callable[[Any], tuple]], ...] = (
    ("exact", _ag(_same, _same)),
    ("place punctuation", _ag(_bare, _same)),
    ("place respelled", _ag(_respelled, _same)),
    ("passage date", _ag(_same, _undated)),
    ("name only", _ag(_same, None)),
    ("place punctuation, name only", _ag(_bare, None)),
)
SERUM_STAGES: tuple[tuple[str, Callable[[Any], tuple]], ...] = (
    ("serum id", lambda s: (s.serum_id,)),
    ("serum id without lab", lambda s: (s.name, _sid(s.serum_id))),
    ("place punctuation", lambda s: (_bare(s.name), _sid(s.serum_id))),
    ("name only", lambda s: (s.name, s.reassortant, tuple(s.annotations))),
)


def staged_pairs(
    ours: Sequence[Any], theirs: Sequence[Any], stages: Sequence[tuple[str, Callable]]
) -> tuple[dict[int, int], dict[str, int]]:
    """ours index -> theirs index; at each stage only keys unique on BOTH sides pair."""
    pairs: dict[int, int] = {}
    by_stage: dict[str, int] = {}
    for label, key in stages:
        used = set(pairs.values())
        lo = [i for i in range(len(ours)) if i not in pairs]
        lr = [j for j in range(len(theirs)) if j not in used]
        ca, cb = Counter(key(ours[i]) for i in lo), Counter(key(theirs[j]) for j in lr)
        a = {key(ours[i]): i for i in lo if ca[key(ours[i])] == 1}
        b = {key(theirs[j]): j for j in lr if cb[key(theirs[j])] == 1}
        new = {a[k]: b[k] for k in a.keys() & b.keys()}
        pairs.update(new)
        by_stage[label] = len(new)
    return pairs, by_stage


def composition(chart: Chart, ref: Chart) -> dict[str, Any]:
    ag, ag_stages = staged_pairs(chart.antigens, ref.antigens, ANTIGEN_STAGES)
    sr, sr_stages = staged_pairs(chart.sera, ref.sera, SERUM_STAGES)
    matched = len(ag) + len(sr)
    union = chart.n_points + ref.n_points - matched
    ag_ref, sr_ref = set(ag.values()), set(sr.values())

    def side(points: Sequence[Any], keep: Callable[[int], bool]) -> dict[str, Any]:
        names = [p.designation() for i, p in enumerate(points) if keep(i)]
        return {"count": len(names), "listed": names[:LISTED]}

    return {
        "antigens": {"map": chart.n_antigens, "reference": ref.n_antigens, "matched": len(ag)},
        "sera": {"map": chart.n_sera, "reference": ref.n_sera, "matched": len(sr)},
        "jaccard": matched / union if union else 1.0,
        "matched_by_stage": {"antigens": ag_stages, "sera": sr_stages},
        "map_only": {
            "antigens": side(chart.antigens, lambda i: i not in ag),
            "sera": side(chart.sera, lambda i: i not in sr),
        },
        "reference_only": {
            "antigens": side(ref.antigens, lambda j: j not in ag_ref),
            "sera": side(ref.sera, lambda j: j not in sr_ref),
        },
        "_pairs": (ag, sr),  # for the basin measurement; dropped before recording
    }


def basin(
    chart: Chart,
    ref: Chart,
    pairs: tuple[dict[int, int], dict[int, int]],
    *,
    starts: int,
    seed: int,
    threads: int = 0,
) -> dict[str, Any]:
    from af.map.optimise import MapProblem, relax, stress

    proj = chart.best()
    arrays = chart.optimiser_arrays(projection=proj)
    problem = MapProblem(
        titre_value=arrays["titre_value"],
        titre_type=arrays["titre_type"],
        column_bases=np.asarray(arrays["column_bases"], dtype=float),
        disconnected=arrays["disconnected"],
        dodgy_is_regular=bool(arrays.get("dodgy_is_regular", False)),
        avidity_adjust=arrays.get("avidity_adjust"),
        unmovable=arrays.get("unmovable"),
    )
    own = float(stress(problem, np.asarray(proj.layout)))
    ref_layout = np.asarray(ref.projections[0].layout)
    start = np.full((chart.n_points, ref_layout.shape[1]), np.nan)
    ag, sr = pairs
    for i, j in ag.items():
        start[i] = ref_layout[j]
    for i, j in sr.items():
        start[chart.n_antigens + i] = ref_layout[ref.n_antigens + j]
    seeded_points = int(np.isfinite(start).all(axis=1).sum())
    result = relax(problem, n_starts=starts, seed=seed, start_layout=start, threads=threads)
    best = result.projections[0]
    fit = procrustes(np.asarray(proj.layout), np.asarray(best.layout))
    dist = np.nan_to_num(fit.distances)
    order = np.argsort(-dist)[:MOVERS]
    names = [
        chart.antigens[p].designation()
        if p < chart.n_antigens
        else "SR " + chart.sera[p - chart.n_antigens].designation()
        for p in order
    ]
    seeded = float(best.stress)
    return {
        "starts": starts,
        "seeded_points": seeded_points,
        "points": chart.n_points,
        "map_stress": own,
        "seeded_stress": seeded,
        "seeded_minus_map": seeded - own,
        "relative": (seeded - own) / own if own else 0.0,
        "rmsd": float(fit.rmsd),
        "apart_over_0_5": int((dist > 0.5).sum()),
        "apart_over_1": int((dist > 1.0).sum()),
        "movers": [
            {"point": n, "distance": float(dist[p])} for n, p in zip(names, order, strict=True)
        ],
    }


def reference_check(
    chart: Chart, ref: Chart, *, starts: int, seed: int, threads: int = 0
) -> dict[str, Any]:
    comp = composition(chart, ref)
    pairs = comp.pop("_pairs")
    return {
        "composition": comp,
        "basin": basin(chart, ref, pairs, starts=starts, seed=seed, threads=threads),
    }


def sentences(check: dict[str, Any], label: str) -> list[str]:
    """What the review says, in words, so no figure can be read without the others."""
    c, b = check["composition"], check["basin"]
    a, s = c["antigens"], c["sera"]
    here, there = c["map_only"]["antigens"]["count"], c["reference_only"]["antigens"]["count"]
    out = [
        f"Against {label}: antigens {a['map']} here, {a['reference']} there, {a['matched']} shared"
        f" ({here} only here, {there} only there); sera {s['map']} / {s['reference']},"
        f" {s['matched']} shared; Jaccard {c['jaccard']:.3f}. Jaccard says which points the maps"
        " share, not whether they sit alike."
    ]
    d, rel = b["seeded_minus_map"], b["relative"]
    where = (
        f"RMSD {b['rmsd']:.2f}; {b['apart_over_0_5']} points more than 0.5 apart,"
        f" {b['apart_over_1']} more than 1"
    )
    if d < -1e-6 * abs(b["map_stress"]):
        movers = ", ".join(f"{m['point']} {m['distance']:.2f}" for m in b["movers"][:8])
        out.append(
            f"Basin: this map is NOT the lowest-stress layout found. Relaxed from the reference's"
            f" positions ({b['seeded_points']} of {b['points']} points, best of {b['starts']}),"
            f" the same titres reach {b['seeded_stress']:.3f}, {-d:.3f} ({-rel:.2%}) below this"
            f" map's {b['map_stress']:.3f}. {where}. Placed most differently: {movers}. The"
            f" cause is the search (which basin the starts reached), not the data."
        )
    else:
        out.append(
            f"Basin: this map is at or below the layout relaxed from the reference's positions"
            f" ({b['seeded_points']} of {b['points']} points, best of {b['starts']}): seeded"
            f" {b['seeded_stress']:.3f} vs {b['map_stress']:.3f} here ({d:+.3f}); {where}."
        )
    return out
