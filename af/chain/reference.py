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
GROUP_FROM = 1.0  # points further apart than this are grouped (map units)
GROUP_LINK = 1.0  # two such points are linked when this close in the map AND displaced this alike
GROUPS = 8  # groups described, largest first (all are counted)
FAR = 5.0  # every point further apart than this is listed with its group
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
    names = _point_names(chart, [int(p) for p in order])
    every = np.arange(chart.n_points)
    layout = np.asarray(proj.layout)
    grouped = moved_groups(chart, every, layout, fit.apply(np.asarray(best.layout)))
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
        "moved": grouped,
    }


def _point_names(chart: Chart, points: Sequence[int]) -> list[str]:
    return [
        chart.antigens[p].designation()
        if p < chart.n_antigens
        else "SR " + chart.sera[p - chart.n_antigens].designation()
        for p in points
    ]


def _year(chart: Chart, p: int) -> str:
    """An antigen's year from its recorded date (not its name); sera and undated points: '?'."""
    if p < chart.n_antigens:
        date = chart.antigens[p].date
        if len(date) >= 4 and date[:4].isdigit():
            return date[:4]
    return "?"


def moved_groups(
    chart: Chart, points: np.ndarray, layout: np.ndarray, fitted: np.ndarray
) -> dict[str, Any]:
    """Where two layouts of the same points differ, and whether the differences are GROUPS.

    `points` are chart indices, `layout` this map's rows for them and `fitted` the other layout's,
    already Procrustes-fitted onto it. Points further apart than GROUP_FROM are linked when they
    are within GROUP_LINK of each other in this map AND moved alike (displacements within
    GROUP_LINK); connected points form a group. A group's spread is the mean distance of its
    members' displacements from the group's mean: small means a block carried elsewhere, large a
    loose chain of neighbours.
    """
    ok = np.isfinite(layout).all(axis=1) & np.isfinite(fitted).all(axis=1)
    dist = np.where(ok, np.linalg.norm(np.nan_to_num(layout - fitted), axis=1), 0.0)
    disp = np.nan_to_num(layout - fitted)
    far = np.nonzero(dist > GROUP_FROM)[0]
    parent = list(range(len(far)))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if len(far):
        pos, dis = layout[far], disp[far]
        for x in range(len(far)):
            near = np.nonzero(
                (np.linalg.norm(pos[x + 1 :] - pos[x], axis=1) <= GROUP_LINK)
                & (np.linalg.norm(dis[x + 1 :] - dis[x], axis=1) <= GROUP_LINK)
            )[0]
            for y in near + x + 1:
                parent[root(x)] = root(int(y))
    members: dict[int, list[int]] = {}
    for x in range(len(far)):
        members.setdefault(root(x), []).append(int(far[x]))
    ordered = sorted(members.values(), key=lambda g: (-len(g), g[0]))
    group_of = {k: n for n, g in enumerate(ordered, start=1) for k in g}

    def describe(n: int, g: list[int]) -> dict[str, Any]:
        mean = disp[g].mean(axis=0)
        years = Counter(_year(chart, int(points[k])) for k in g)
        return {
            "group": n,
            "size": len(g),
            "mean_shift": float(np.linalg.norm(mean)),
            "spread": float(np.linalg.norm(disp[g] - mean, axis=1).mean()),
            "years": dict(sorted(years.items())),
            "points": _point_names(chart, [int(points[k]) for k in g]),
        }

    far_list = sorted((int(k) for k in far if dist[k] > FAR), key=lambda k: -dist[k])
    return {
        "compared": int(ok.sum()),
        "apart_over_0_5": int((dist > 0.5).sum()),
        "apart_over_1": int((dist > 1.0).sum()),
        "apart_over_2": int((dist > 2.0).sum()),
        "apart_over_5": int((dist > FAR).sum()),
        "n_groups": len(ordered),
        "group_sizes": [len(g) for g in ordered],
        "groups": [describe(n, g) for n, g in enumerate(ordered[:GROUPS], start=1)],
        "far": [
            {"point": name, "distance": float(dist[k]), "group": group_of[k]}
            for name, k in zip(
                _point_names(chart, [int(points[k]) for k in far_list]), far_list, strict=True
            )
        ],
    }


def shipped(chart: Chart, ref: Chart, pairs: tuple[dict[int, int], dict[int, int]]) -> dict:
    """This map against the reference's OWN layout, over the points that pair, no relaxation:
    what a reader comparing the two maps sees."""
    ag, sr = pairs
    mine = [*ag, *(chart.n_antigens + i for i in sr)]
    theirs = [*ag.values(), *(ref.n_antigens + j for j in sr.values())]
    a = np.asarray(chart.best().layout)[mine]
    b = np.asarray(ref.projections[0].layout)[theirs]
    fit = procrustes(a, b)
    return {"rmsd": float(fit.rmsd), **moved_groups(chart, np.asarray(mine), a, fit.apply(b))}


def reference_check(
    chart: Chart, ref: Chart, *, starts: int, seed: int, threads: int = 0
) -> dict[str, Any]:
    comp = composition(chart, ref)
    pairs = comp.pop("_pairs")
    return {
        "composition": comp,
        "shipped": shipped(chart, ref, pairs),
        "basin": basin(chart, ref, pairs, starts=starts, seed=seed, threads=threads),
    }


def _groups_sentence(m: dict[str, Any]) -> str:
    if not m["n_groups"]:
        return "no point is more than 1 apart."
    head = "; ".join(
        f"{g['size']} moved {g['mean_shift']:.2f} (spread {g['spread']:.2f})"
        for g in m["groups"][:4]
    )
    return (
        f"the {m['apart_over_1']} points more than 1 apart form {m['n_groups']} group(s), largest:"
        f" {head}; {m['apart_over_5']} points are more than {FAR:g} apart."
    )


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
    if (sh := check.get("shipped")) is not None:
        out.append(
            f"As drawn: against {label}'s own layout ({sh['compared']} shared points fitted, no"
            f" relaxation) the RMSD is {sh['rmsd']:.2f}; " + _groups_sentence(sh)
        )
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
        if "moved" in b:
            out[-1] += " In groups: " + _groups_sentence(b["moved"])
    else:
        out.append(
            f"Basin: this map is at or below the layout relaxed from the reference's positions"
            f" ({b['seeded_points']} of {b['points']} points, best of {b['starts']}): seeded"
            f" {b['seeded_stress']:.3f} vs {b['map_stress']:.3f} here ({d:+.3f}); {where}."
        )
    return out
