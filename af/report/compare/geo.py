"""Same-science comparison of geographic figures: dots per month and location, and per clade.

Input is the I7 ``geo`` shape, the same as ae's ``geo/<st>-records.json`` (periods -> locations
-> coloured point counts), where a point may carry a ``clade``; without one, its colour stands in.
Only months present on both sides are compared; months on one side only are listed.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def tallies(doc: dict[str, Any]) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]]]:
    by_location: dict[str, Counter[str]] = {}
    by_clade: dict[str, Counter[str]] = {}
    for period in doc["periods"]:
        month = period["period"]
        loc = by_location.setdefault(month, Counter())
        cla = by_clade.setdefault(month, Counter())
        for location in period["locations"]:
            for point in location["points"]:
                loc[location["name"]] += point["count"]
                cla[point.get("clade") or point["color"].lower()] += point["count"]
    return by_location, by_clade


def _diff(a: Counter[str], b: Counter[str]) -> dict[str, Any]:
    keys = sorted(a.keys() | b.keys(), key=lambda k: -abs(a[k] - b[k]))  # largest change first
    moved = sum(abs(a[k] - b[k]) for k in keys)
    total = max(1, sum(a.values()) + sum(b.values()))
    return {
        "ref": sum(a.values()), "new": sum(b.values()),
        "keys_only_ref": len(a.keys() - b.keys()), "keys_only_new": len(b.keys() - a.keys()),
        "abs_diff": moved,
        "frac_diff": moved / total,  # 0 = identical, 1 = disjoint
        "largest": [f"{k}: {a[k]} -> {b[k]}" for k in keys[:3]],
    }  # fmt: skip


def compare(ref: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    rl, rc = tallies(ref)
    nl, nc = tallies(new)
    months = sorted(rl.keys() & nl.keys())
    return {
        "months_compared": months,
        "months_only_ref": sorted(rl.keys() - nl.keys()),
        "months_only_new": sorted(nl.keys() - rl.keys()),
        "per_month": {
            m: {"location": _diff(rl[m], nl[m]), "clade": _diff(rc[m], nc[m])} for m in months
        },
    }
