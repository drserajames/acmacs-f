"""Run the same-science comparison for a built report against a set of reference I7s.

Reads the report's manifest (which figures it used), finds the reference I7 for each slot
(``<reference>/<slot with / as .>.i7.json``), compares, applies the limits file, and writes
``COMPARISON.json`` and ``COMPARISON.md``. Every figure is compared, and the exit status is 1
if any gated check fails, so a pipeline can stop on it.

Known differences are data, not code (design rule 9): an ``[[expected]]`` entry in the limits
file names a slot, a check and the reason (e.g. a deliberate rotation, or a curated arrangement
not yet set as an override). A failing check covered by one is reported as "expected" with its
reason. An expectation whose check passes is **stale** and counts as a failure (design rule 1:
a rule that matches nothing is an error), so the list cannot silently outlive its reason.

Run: ``python -m af.report.compare.run MANIFEST REFERENCE_DIR --limits L.toml --out DIR``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.report.compare import maps
from af.util.config import load_config


@dataclass(frozen=True)
class MapLimits:
    antigens_jaccard_min: float | None = None
    sera_jaccard_min: float | None = None
    clade_ari_min: float | None = None
    p95_displacement_max: float | None = None
    frac_moved_gt_1_max: float | None = None
    centroid_diff_max: float | None = None
    rotation_deg_max: float | None = None


@dataclass(frozen=True)
class TreeLimits:
    rf_normalised_max: float | None = None
    clade_ari_min: float | None = None


@dataclass(frozen=True)
class GeoLimits:
    location_frac_diff_max: float | None = None
    clade_frac_diff_max: float | None = None


@dataclass(frozen=True)
class Expected:
    """A named, approved known difference. Who approved it and when are required: an
    expectation without them is how exception lists rot."""

    slot: str
    check: str
    reason: str
    approved_by: str
    decided: dt.date


@dataclass(frozen=True)
class Limits:
    map: MapLimits = field(default_factory=MapLimits)
    tree: TreeLimits = field(default_factory=TreeLimits)
    geo: GeoLimits = field(default_factory=GeoLimits)
    expected: list[Expected] = field(default_factory=list)


MAP_CHECKS = (
    "antigens jaccard", "sera jaccard", "clade ARI", "p95 displacement", "frac moved > 1",
    "clade centroid max diff", "rotation deg", "reflected", "RMSD (not gated)",
)  # fmt: skip


def _check(name: str, value: float, op: str, limit: float | None) -> dict[str, Any]:
    if limit is None or math.isnan(value):  # reported, not gated
        return {"check": name, "value": value, "limit": limit, "ok": None}
    ok = value >= limit if op == ">=" else value <= limit
    return {"check": name, "value": value, "limit": limit, "ok": ok}


def map_checks(res: dict[str, Any], lim: MapLimits) -> list[dict[str, Any]]:
    nan = float("nan")
    a, s = res["antigens"], res["sera"]
    p, c = res.get("procrustes", {}), res.get("clade_centroids", {})
    rotation = abs(p["rotation_deg"]) if "rotation_deg" in p else nan
    reflected = float(p["reflected"]) if "reflected" in p else nan
    return [
        _check("antigens jaccard", a["jaccard"], ">=", lim.antigens_jaccard_min),
        _check("sera jaccard", s["jaccard"], ">=", lim.sera_jaccard_min),
        _check("clade ARI", a["clade"]["adjusted_rand"], ">=", lim.clade_ari_min),
        _check("p95 displacement", p.get("p95", nan), "<=", lim.p95_displacement_max),
        _check("frac moved > 1", p.get("frac_gt_1", nan), "<=", lim.frac_moved_gt_1_max),
        _check("clade centroid max diff", c.get("max_abs_diff", nan), "<=", lim.centroid_diff_max),
        _check("rotation deg", rotation, "<=", lim.rotation_deg_max),
        # A mirrored map is always a difference when orientation is gated at all.
        _check("reflected", reflected, "<=", 0.0 if lim.rotation_deg_max is not None else None),
        _check("RMSD (not gated)", p.get("rmsd", nan), "<=", None),
    ]


def apply_expected(slot: str, checks: list[dict[str, Any]], expected: list[Expected]) -> None:
    """Mark checks covered by an expectation: "expected" if failing, "stale" if passing.

    The note says what was expected and what was found, so a stale entry reads as "this known
    difference has changed", not as a new bug.
    """
    for exp in expected:
        if exp.slot != slot:
            continue
        for check in checks:
            if check["check"] != exp.check:
                continue
            origin = f"{exp.reason} (approved by {exp.approved_by}, {exp.decided.isoformat()})"
            found = f"{check['value']:.3f} against limit {check['limit']}"
            if check["ok"] is False:
                check["ok"] = "expected"
                check["expected"] = f"expected difference: {origin}; found {found}"
            elif check["ok"] is True:
                check["ok"] = "stale"
                check["expected"] = (
                    f"STALE: expected this check to fail because {origin}, but found {found}, "
                    "within the limit. The known difference has changed; review the entry."
                )
            else:
                check["expected"] = f"{origin}; check is not gated, so nothing to compare"


def slot_status(checks: list[dict[str, Any]]) -> str:
    oks = [c["ok"] for c in checks]
    if False in oks or "stale" in oks:
        return "FAIL"
    return "ok (expected differences)" if "expected" in oks else "ok"


def compare_report(
    manifest: dict[str, Any], reference: Path, limits: Limits, how: str
) -> tuple[list[dict[str, Any]], int]:
    """Compare every figure in ``manifest``; return the rows and the number of failing slots."""
    known = {e.check for e in limits.expected}
    unknown = sorted(known - set(MAP_CHECKS))
    if unknown:
        raise ValueError(f"expected: unknown check names {unknown}; valid: {list(MAP_CHECKS)}")
    used = {f["slot"] for f in manifest["figures"]}
    orphans = sorted({e.slot for e in limits.expected} - used)
    if orphans:
        raise ValueError(f"expected: slots not in this report {orphans}")
    rows: list[dict[str, Any]] = []
    failed = 0
    for fig in manifest["figures"]:
        slot = fig["slot"]
        new = json.loads(Path(fig["i7"]).read_text())
        ref_path = reference / (slot.replace("/", ".") + ".i7.json")
        if new["kind"] == "placeholder":
            rows.append({"slot": slot, "status": "placeholder"})
        elif not ref_path.is_file():
            rows.append({"slot": slot, "status": "no reference"})
        elif new["kind"] == "map":
            res = maps.compare(json.loads(ref_path.read_text()), new, how)
            checks = map_checks(res, limits.map)
            apply_expected(slot, checks, limits.expected)
            status = slot_status(checks)
            failed += status == "FAIL"
            rows.append({"slot": slot, "status": status, "checks": checks, "detail": res})
        else:
            rows.append({"slot": slot, "status": f"{new['kind']}: not compared on I7 yet"})
    return rows, failed


def _cell(check: dict[str, Any]) -> str:
    mark = {False: " ✗", "expected": " (expected)", "stale": " STALE"}.get(check["ok"], "")
    return f"{check['value']:.3f}{mark}"


def markdown(
    manifest: dict[str, Any], rows: list[dict[str, Any]], limits_name: str, how: str
) -> str:
    lines = [
        f"# Same-science comparison: {manifest['report']}", "",
        f"Report built {manifest['built']}; limits `{limits_name}`; points matched by {how}.", "",
        "| Slot | Status | " + " | ".join(MAP_CHECKS) + " |",
        "|---|---|" + "---|" * len(MAP_CHECKS),
    ]  # fmt: skip
    notes = []
    for row in rows:
        if "checks" in row:
            lines.append(f"| {row['slot']} | {row['status']} | "
                         + " | ".join(_cell(c) for c in row["checks"]) + " |")  # fmt: skip
            notes += [f"- {row['slot']} / {c['check']}: {c['expected']}"
                      for c in row["checks"] if "expected" in c]  # fmt: skip
        else:
            lines.append(f"| {row['slot']} | {row['status']} |" + " |" * len(MAP_CHECKS))
    if notes:
        lines += ["", "Named differences:", *notes]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a built report with a reference.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--limits", type=Path, required=True)
    parser.add_argument("--match", choices=("id", "name"), default="name")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    limits = load_config(args.limits, Limits)
    manifest = json.loads(args.manifest.read_text())
    rows, failed = compare_report(manifest, args.reference, limits, args.match)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "COMPARISON.json").write_text(json.dumps(rows, indent=1))
    (args.out / "COMPARISON.md").write_text(markdown(manifest, rows, args.limits.name, args.match))
    missing = sum(r["status"] == "no reference" for r in rows)
    print(f"{len(rows)} figures, {failed} failing, {missing} without reference", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
