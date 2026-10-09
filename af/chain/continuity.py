"""A rebuilt map against the version it replaces: what moved, declared beside the store.

Maps are rebuilt from scratch and their differences are DECLARED, not engineered away (Sarah,
1 Oct 2026). Month to month that means a record per rebuilt map (notes/chains/MONTHLY-UPDATE.md
§4): stress before and after, a Procrustes fit of the previous layout onto the new one over the
points both maps share, how many points sit further apart than each reporting cut-off, every
point beyond the movers cut-off by name, and which points the new tables added or dropped.

Points pair by kind and designation, never by index: a rebuild appends points, but tables that
arrive late for an earlier date land mid-chart. A designation that occurs more than once on
either side pairs with nothing and is counted. The two stresses are only comparable when both
versions fitted the same problem (same tables inputs, same chain parameters); the record says
whether they did and, if not, which parameters differ.

The cut-offs are data (a thresholds TOML, each with its reason), not literals:

    [[apart]]
    distance = 0.5
    reason = "..."
    [movers]
    distance = 1.0
    reason = "..."

    python -m af.chain.continuity <chains root> <lab/group/variant> <version> <thresholds> <records>
        [--note TEXT]

writes ``<records>/<lab/group/variant>/<version>.continuity.json``; it refuses to overwrite.
``--note`` says why the record exists when it is not a month-to-month one (e.g. a one-off rebuild
on the same tables); it is recorded as ``context`` and read first.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from af.chart.ace import read_chart
from af.chart.model import Chart
from af.chart.procrustes import procrustes
from af.util.artefacts import sha256_path
from af.util.config import ConfigError, load_config

FIELDS = (
    "context: why this record exists when it is not month to month (--note), else null. "
    "stress: each map's own best stress; comparable only when same_problem is true. "
    "points: paired = same kind and designation once on each side; filled = the same point "
    "with its passage (antigen) or serum id (serum) filled in or emptied (same kind, name, "
    "reassortant, annotations; one candidate a side; the field empty on one side), fitted like "
    "paired points; added/removed = only in the new/previous map; ambiguous = designations "
    "occurring more than once on a side (not paired). "
    "procrustes: the previous layout fitted onto the new one (rotation, reflection, translation, "
    "no scaling) over paired and filled points connected in both; apart = counts beyond each "
    "cut-off. "
    "movers: every paired or filled point further apart than thresholds.movers, furthest first."
)


@dataclass(frozen=True)
class Cutoff:
    distance: float  # map units
    reason: str  # why this cut-off: a rule without its reason is a defect


@dataclass(frozen=True)
class Thresholds:
    apart: list[Cutoff]  # each one's count is reported
    movers: Cutoff  # every point further apart than this is listed by name


def load_thresholds(path: Path) -> Thresholds:
    t = load_config(path, Thresholds)
    problems = [
        f"{where}: {problem}"
        for where, c in [*((f"apart[{i}]", c) for i, c in enumerate(t.apart)), ("movers", t.movers)]
        for problem in (["distance must be > 0"] if not c.distance > 0 else [])
        + (["reason is empty"] if not c.reason.strip() else [])
    ]
    if not t.apart:
        problems.append("apart: at least one cut-off is needed")
    if problems:
        raise ConfigError(path, problems)
    return t


def _keys(chart: Chart) -> list[tuple[str, str]]:
    return [("antigen", a.designation()) for a in chart.antigens] + [
        ("serum", s.designation()) for s in chart.sera
    ]


def _bases(chart: Chart) -> list[tuple[tuple[str, str, str, tuple[str, ...]], str]]:
    """Each point without the field a later rebuild may fill (an antigen's passage, a serum's
    id), and that field."""
    return [
        (("antigen", a.name, a.reassortant, a.annotations), a.passage) for a in chart.antigens
    ] + [(("serum", s.name, s.reassortant, s.annotations), s.serum_id) for s in chart.sera]


def _filled(
    new: Chart, previous: Chart, left_new: list[int], left_prev: list[int]
) -> list[tuple[int, int]]:
    """Leftovers that are one point with its passage or serum id filled in (or emptied): same
    kind, name, reassortant and annotations, exactly one candidate on each side, and the field
    empty on at least one side. A merge that pairs a blank passage by name (STRICT_THEN_NAME) or a
    table whose blank serum id was filled would otherwise show the point as removed plus added."""
    bn, bp = _bases(new), _bases(previous)
    on_new = Counter(bn[i][0] for i in left_new)
    on_prev = {bp[j][0]: j for j in left_prev}
    once_prev = Counter(bp[j][0] for j in left_prev)
    return [
        (i, on_prev[bn[i][0]])
        for i in left_new
        if on_new[bn[i][0]] == 1
        and once_prev[bn[i][0]] == 1
        and not (bn[i][1] and bp[on_prev[bn[i][0]]][1])
    ]


def _named(keys: list[tuple[str, str]], kind: str) -> list[str]:
    return sorted(name for k, name in keys if k == kind)


def _side(keys: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "antigens": len(_named(keys, "antigen")),
        "sera": len(_named(keys, "serum")),
        "listed": {"antigens": _named(keys, "antigen"), "sera": _named(keys, "serum")},
    }


def continuity(new: Chart, previous: Chart, thresholds: Thresholds) -> dict[str, Any]:
    """`new` against `previous`, each by its best projection."""
    kn, kp = _keys(new), _keys(previous)
    cn, cp = Counter(kn), Counter(kp)
    ambiguous = sorted({k for k, n in cn.items() if n > 1} | {k for k, n in cp.items() if n > 1})
    where = {k: i for i, k in enumerate(kp) if cp[k] == 1}
    pairs = [(i, where[k]) for i, k in enumerate(kn) if cn[k] == 1 and k in where]
    filled = _filled(
        new,
        previous,
        [i for i, k in enumerate(kn) if k not in cp],
        [j for j, k in enumerate(kp) if k not in cn],
    )
    gone_new, gone_prev = {i for i, _ in filled}, {j for _, j in filled}
    added = [k for i, k in enumerate(kn) if k not in cp and i not in gone_new]
    removed = [k for j, k in enumerate(kp) if k not in cn and j not in gone_prev]
    n_paired = len(pairs)
    pairs += filled

    pn, pp = new.best(), previous.best()
    a = np.asarray(pn.layout, dtype=float)[[i for i, _ in pairs]]
    b = np.asarray(pp.layout, dtype=float)[[j for _, j in pairs]]
    fit = procrustes(a, b)
    dist = fit.distances
    ok = ~np.isnan(dist)
    far = sorted(
        (
            (-float(d), i)  # furthest first; equal distances (stacked points) in chart order
            for (i, _), d in zip(pairs, dist, strict=True)
            if not np.isnan(d) and d > thresholds.movers.distance
        )
    )
    movers = [{"kind": kn[i][0], "point": kn[i][1], "distance": -d} for d, i in far]
    sn, sp = pn.stress_value, pp.stress_value
    return {
        "stress": {"new": sn, "previous": sp, "difference": sn - sp, "relative": (sn - sp) / sp},
        "points": {
            "new": len(kn),
            "previous": len(kp),
            "paired": n_paired,
            "filled": {
                "count": len(filled),
                "listed": [f"{kp[j][1]} -> {kn[i][1]}" for i, j in filled],
            },
            "added": _side(added),
            "removed": _side(removed),
            "ambiguous": {"count": len(ambiguous), "listed": [f"{k}: {n}" for k, n in ambiguous]},
        },
        "procrustes": {
            "compared": int(ok.sum()),
            "disconnected": int((~ok).sum()),
            "rmsd": fit.rmsd,
            "apart": [
                {"distance": c.distance, "count": int((dist[ok] > c.distance).sum())}
                for c in thresholds.apart
            ],
        },
        "movers": movers,
    }


# ---------------------------------------------------------------- the record beside the store


def _history(dataset_dir: Path, version: str) -> dict[str, Any]:
    entries = [
        json.loads(line) for line in (dataset_dir / "HISTORY.jsonl").read_text().splitlines()
    ]
    mine = [e for e in entries if e.get("version") == version and e.get("event") == "published"]
    if len(mine) != 1:
        raise ValueError(
            f"{dataset_dir}: version {version} is published {len(mine)} times in HISTORY.jsonl"
        )
    return mine[0]


def _chosen(version_dir: Path) -> Path:
    step = json.loads((version_dir / "chain.json").read_text())["steps"][-1]
    path = version_dir / step["directory"] / step["chosen_file"]
    if not path.is_file():
        raise FileNotFoundError(f"{version_dir}: chosen map {path} missing")
    return path


def _problem(version_dir: Path) -> dict[str, Any]:
    prov = json.loads((version_dir / "PROVENANCE.json").read_text())
    return {"inputs": prov["inputs"], "parameters": prov["parameters"]}


def record(
    chains: Path,
    dataset: str,
    version: str,
    thresholds_path: Path,
    records: Path,
    note: str | None = None,
) -> dict[str, Any]:
    """The continuity record of `version` against its parent (HISTORY.jsonl's `parent`); `note`
    is the context a reader needs first, when the pair is not last month's map and this one's."""
    if note is not None and not note.strip():
        raise ValueError("--note is empty: give the context or leave it out")
    root = chains / dataset
    parent = _history(root, version).get("parent")
    if parent is None:
        raise ValueError(
            f"{dataset} {version}: no previous version (parent is None); nothing to compare"
        )
    vn, vp = root / "versions" / version, root / "versions" / parent
    new_path, prev_path = _chosen(vn), _chosen(vp)
    thresholds = load_thresholds(thresholds_path)
    pn, pp = _problem(vn), _problem(vp)
    keys = sorted(set(pn["parameters"]) | set(pp["parameters"]))
    differ = [k for k in keys if pn["parameters"].get(k) != pp["parameters"].get(k)]
    same = pn["inputs"] == pp["inputs"] and not differ
    drift = records / dataset / f"{parent}.tables-drift.json"
    from af.run.runtime import release_info

    release = release_info()
    return {
        "dataset": dataset,
        "version": version,
        "context": note,
        "map_sha256": sha256_path(new_path),
        "previous": {"version": parent, "map_sha256": sha256_path(prev_path)},
        "same_problem": same,
        "inputs_differ": pn["inputs"] != pp["inputs"],
        "parameters_differ": differ,
        "retires": str(drift.relative_to(records)) if drift.is_file() else None,
        "thresholds": {
            "path": str(thresholds_path),
            "sha256": sha256_path(thresholds_path),
            **asdict(thresholds),
        },
        **continuity(read_chart(new_path), read_chart(prev_path), thresholds),
        "method_note": (
            "rebuilt map against the version it replaces (MONTHLY-UPDATE §4), pairing points by "
            "designation; measured afterwards, read-only"
        ),
        "measured": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "measured_by": {
            "script": "af.chain.continuity",
            "release": release["commit"] if release else f"not a release ({sys.executable})",
        },
        "fields": FIELDS,
    }


def sentences(rec: dict[str, Any], *, movers: int = 8) -> list[str]:
    """What the record says, in words; `movers` are listed by name, all are counted."""
    s, p, f = rec["stress"], rec["points"], rec["procrustes"]
    prev = rec["previous"]["version"]
    out = [rec["context"]] if rec.get("context") else []  # records before --note have no key
    out += [
        f"Against the previous version {prev}: {p['paired']} points shared,"
        f" {p['added']['antigens']} antigens and {p['added']['sera']} sera added,"
        f" {p['removed']['antigens']} antigens and {p['removed']['sera']} sera gone"
        + (
            f", {p['filled']['count']} the same point with its passage or serum id filled in"
            if p["filled"]["count"]
            else ""
        )
        + (
            f", {p['ambiguous']['count']} designation"
            f"{'s' if p['ambiguous']['count'] != 1 else ''} repeated (not paired)"
            if p["ambiguous"]["count"]
            else ""
        )
        + "."
    ]
    if rec["same_problem"]:
        out.append(
            f"Same tables and parameters: stress {s['previous']:.3f} -> {s['new']:.3f}"
            f" ({s['relative']:+.2%})."
        )
    else:
        why = ["the tables inputs"] if rec["inputs_differ"] else []
        why += (
            [f"parameters {', '.join(rec['parameters_differ'])}"]
            if rec["parameters_differ"]
            else []
        )
        out.append(
            f"Stress {s['previous']:.3f} -> {s['new']:.3f}, NOT comparable:"
            f" {' and '.join(why)} differ."
        )
    apart = ", ".join(f"{a['count']} more than {a['distance']:g}" for a in f["apart"])
    listed = ", ".join(f"{m['point']} {m['distance']:.2f}" for m in rec["movers"][:movers])
    cut = rec["thresholds"]["movers"]["distance"]
    if all(a["distance"] != cut for a in f["apart"]):  # else that count already says it
        apart += f"; {len(rec['movers'])} more than {cut:g}"
    out.append(
        f"Procrustes over {f['compared']} points: RMSD {f['rmsd']:.2f}; {apart}"
        + (f"; furthest: {listed}." if listed else ".")
    )
    if rec["retires"]:
        out.append(f"This rebuild retires the drift record {rec['retires']}.")
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m af.chain.continuity", description=__doc__.split("\n")[0]
    )
    parser.add_argument("chains", type=Path, help="the chains store root")
    parser.add_argument("dataset", help="lab/group/variant, e.g. labx/h3/merged")
    parser.add_argument("version", help="the rebuilt map's version")
    parser.add_argument("thresholds", type=Path, help="the cut-offs TOML")
    parser.add_argument("records", type=Path, help="records directory beside the store")
    parser.add_argument("--note", help="why this record exists, when not month to month")
    args = parser.parse_args(argv)
    out = args.records / args.dataset / f"{args.version}.continuity.json"
    if out.exists():
        raise FileExistsError(f"{out} exists; continuity records are never overwritten")
    rec = record(args.chains, args.dataset, args.version, args.thresholds, args.records, args.note)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(out)
    print(out)
    for line in sentences(rec):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
