"""Run the same-science comparison for a built report against a set of reference I7s.

Reads the report's build record (``<id>.build.json``: which figures it used), finds the
reference I7 for each slot (``<reference>/<slot with / as .>.i7.json``), compares, applies the
limits file, and writes ``COMPARISON.json`` and ``COMPARISON.md``. Every figure is compared, and
the exit status is 1 if any gated check fails, so a pipeline can stop on it.

Known differences are data, not code (design rule 9): an ``[[expected]]`` entry in the limits
file names a slot, a check and the reason (e.g. a deliberate rotation, or a curated arrangement
not yet set as an override). A failing check covered by one is reported as "expected" with its
reason. An expectation whose check passes is **stale** and counts as a failure (design rule 1:
a rule that matches nothing is an error), so the list cannot silently outlive its reason.

Known sets of points can also be excused (``[[excused]]``): taken out of the comparison on
named slots, with the approved reason, so the rest of the check still bites.

Run: ``python -m af.report.compare.run BUILD_RECORD REFERENCE_DIR --limits L.toml --out DIR``.
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

from af.report.compare import geo, maps, trees
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
    vaccine_differences_max: float | None = None  # antigens marked as a vaccine on one side only


@dataclass(frozen=True)
class TreeLimits:
    rf_normalised_max: float | None = None  # tree store vs tree store (compare.trees.compare)
    clade_ari_min: float | None = None
    # tree figures (I7), reported and not gated until limits are agreed:
    leaves_jaccard_min: float | None = None
    order_spearman_min: float | None = None
    section_jaccard_min: float | None = None


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
class Amendment:
    """One change to an adopted set of limits: a limit added, changed or withdrawn."""

    date: dt.date
    by: str
    change: str


@dataclass(frozen=True)
class Adoption:
    """Who adopted this set of limits, when, and whether it is final.

    Required, and printed at the top of every comparison, so a provisional set cannot quietly
    become permanent: anyone reading a result sees its status without looking elsewhere.
    """

    status: str  # "provisional" or "final"
    adopted_by: str
    adopted: dt.date
    review: str = ""  # what re-tests a provisional set, and when
    amendments: list[Amendment] = field(default_factory=list)  # printed with the status


@dataclass(frozen=True)
class Excused:
    """Named points taken out of the comparison on some slots, with the approved reason.

    For a difference that is a known set of points (e.g. antigens af shows because a legacy hide
    rule was dropped), where excusing a whole check would also hide unrelated differences. The
    point keys (``name|passage_class`` or ``name|serum_id``, one per line) live in a private
    file beside the limits, never in code. Every listed point must still be a one-sided
    difference on every listed slot: one found on both sides, or on neither, makes the entry
    stale, and a stale entry fails.
    """

    slots: list[str]
    points: Path
    reason: str
    approved_by: str
    decided: dt.date


@dataclass(frozen=True)
class CladesConfig:
    """Where to read clade definitions for mapping tree-figure labels to canonical names.

    The PIN is not configured: it is taken from the af tree figure being compared
    (provenance.inputs.clade_set, "<repository>@<commit>"), so the comparison cannot use a
    different nomenclature from the tree it compares.
    """

    clones: Path  # the pinned upstream nomenclature clones
    local: Path  # acmacs-f-data clades/local.tsv (local sub-groups under their upstream parent)
    clades_json: Path  # acmacs-data clades.json: the local groups' signatures, until switch-over
    subtypes: dict[str, str]  # tree I7 subtype -> clade-set subtype, e.g. h3 = "A(H3N2)"


def clade_set_for(tree_doc: dict[str, Any], cfg: CladesConfig, cache: dict[str, Any]) -> Any:
    """The clade set a tree figure was labelled with: its recorded pin plus the local layer."""
    from af.clades.importer import clades_json_signatures
    from af.clades.local import extend_from_file
    from af.clades.nomenclature import Pin, load_clade_set

    subtype = tree_doc["tree"]["subtype"]
    if subtype not in cfg.subtypes:
        raise ValueError(f"clades config: no clade-set subtype for tree subtype {subtype!r}")
    recorded = tree_doc["provenance"].get("inputs", {}).get("clade_set")
    if not recorded or "@" not in recorded:
        raise ValueError(f"tree figure ({subtype}) records no clade_set '<repository>@<commit>'")
    pin_text = recorded.split("+local", 1)[0]  # the local layer is added here, from local.tsv
    if pin_text in cache:
        return cache[pin_text]
    name = cfg.subtypes[subtype]
    repository, commit = pin_text.rsplit("@", 1)
    clade_set = load_clade_set(name, cfg.clones, Pin(name, repository, commit))
    signatures = clades_json_signatures(cfg.clades_json).get(name, {})
    cache[pin_text] = extend_from_file(clade_set, cfg.local, signatures=signatures)
    return cache[pin_text]


@dataclass(frozen=True)
class Limits:
    adoption: Adoption
    map: MapLimits = field(default_factory=MapLimits)
    tree: TreeLimits = field(default_factory=TreeLimits)
    geo: GeoLimits = field(default_factory=GeoLimits)
    expected: list[Expected] = field(default_factory=list)
    excused: list[Excused] = field(default_factory=list)


ONE_SIDED_LISTED = 25  # per slot, group and side, in COMPARISON.md; the JSON has them all

MAP_CHECKS = (
    "antigens jaccard", "sera jaccard", "clade ARI", "p95 displacement", "frac moved > 1",
    "clade centroid max diff", "rotation deg", "reflected", "vaccine marks differ",
    "RMSD (not gated)",
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
    checks = [
        _check("antigens jaccard", a["jaccard"], ">=", lim.antigens_jaccard_min),
        _check("sera jaccard", s["jaccard"], ">=", lim.sera_jaccard_min),
        _check("clade ARI", a["clade"]["adjusted_rand"], ">=", lim.clade_ari_min),
        _check("p95 displacement", p.get("p95", nan), "<=", lim.p95_displacement_max),
        _check("frac moved > 1", p.get("frac_gt_1", nan), "<=", lim.frac_moved_gt_1_max),
        _check("clade centroid max diff", c.get("max_abs_diff", nan), "<=", lim.centroid_diff_max),
        _check("rotation deg", rotation, "<=", lim.rotation_deg_max),
        # A mirrored map is always a difference when orientation is gated at all.
        _check("reflected", reflected, "<=", 0.0 if lim.rotation_deg_max is not None else None),
        _check(
            "vaccine marks differ",
            float(
                len(a.get("vaccine_only_ref_keys", [])) + len(a.get("vaccine_only_new_keys", []))
            ),
            "<=",
            lim.vaccine_differences_max,
        ),  # fmt: skip
        _check("RMSD (not gated)", p.get("rmsd", nan), "<=", None),
    ]
    if p.get("identical_layout"):
        for check in checks:
            if check["check"] in DISPLACEMENT_CHECKS:
                check.update(ok=None, not_tested="identical layout on both sides")
    return checks


DISPLACEMENT_CHECKS = ("p95 displacement", "frac moved > 1", "clade centroid max diff")


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


def read_keys(path: Path) -> set[str]:
    """Point keys, one per line; blank lines and ``#`` comments ignored. Empty is an error."""
    keys = {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not keys:
        raise ValueError(f"{path}: no point keys")
    return keys


def excuse_points(
    slot: str, ref: dict[str, Any], new: dict[str, Any], excused: list[tuple[Excused, set[str]]],
    how: str,
) -> list[dict[str, Any]]:  # fmt: skip
    """Remove excused points from both maps (in place); return one note per entry that applies."""
    notes = []
    for entry, keys in excused:
        if slot not in entry.slots:
            continue
        seen: dict[str, set[str]] = {"ref": set(), "new": set()}
        for side, doc in (("ref", ref), ("new", new)):
            for group in ("antigens", "sera"):
                points = doc["map"][group]
                kept = []
                for point in points:
                    key = maps.point_key(point, how)
                    if key in keys and maps.drawn(point):
                        seen[side].add(key)
                    if key not in keys:
                        kept.append(point)
                doc["map"][group] = kept
        one_sided = seen["ref"] ^ seen["new"]
        stale = sorted(keys - one_sided)
        origin = f"{entry.reason} (approved by {entry.approved_by}, {entry.decided.isoformat()})"
        notes.append({
            "points": len(keys), "one_sided": len(one_sided & keys), "stale": len(stale),
            "note": (
                f"excused {len(keys)} points: {origin}" if not stale else
                f"STALE: {len(stale)} of {len(keys)} excused points are no longer a one-sided "
                f"difference ({origin}); review the list"
            ),
        })  # fmt: skip
    return notes


def compare_report(
    record: dict[str, Any], reference: Path, limits: Limits, how: str,
    clades: CladesConfig | None = None,
) -> tuple[list[dict[str, Any]], int]:  # fmt: skip
    """Compare every figure in a build ``record``; return the rows and the number failing."""
    manifest = record
    if limits.adoption.status not in ("provisional", "final"):
        raise ValueError(f"adoption.status: {limits.adoption.status!r} not provisional|final")
    if limits.adoption.status == "provisional" and not limits.adoption.review:
        raise ValueError("adoption.review: a provisional set must say what re-tests it")
    known = {e.check for e in limits.expected}
    unknown = sorted(known - set(MAP_CHECKS) - set(TREE_CHECKS) - set(GEO_CHECKS))
    if unknown:
        raise ValueError(
            f"expected: unknown check names {unknown}; "
            f"valid: {[*MAP_CHECKS, *TREE_CHECKS, *GEO_CHECKS]}"
        )
    used = {f["slot"] for f in manifest["figures"]}
    orphans = sorted(
        ({e.slot for e in limits.expected} | {s for e in limits.excused for s in e.slots}) - used
    )
    if orphans:
        raise ValueError(f"expected/excused: slots not in this report {orphans}")
    excused = [
        (entry, {maps.normalise_key(k, how) for k in read_keys(entry.points)})
        for entry in limits.excused
    ]
    rows: list[dict[str, Any]] = []
    failed = 0
    clade_cache: dict[str, Any] = {}
    for fig in manifest["figures"]:
        slot = fig["slot"]
        new = json.loads(Path(fig["i7"]).read_text())
        ref_path = reference / (slot.replace("/", ".") + ".i7.json")
        if new["kind"] == "placeholder":
            rows.append({"slot": slot, "status": "placeholder"})
        elif not ref_path.is_file():
            rows.append({"slot": slot, "status": "no reference"})
        elif new["kind"] == "map":
            ref = json.loads(ref_path.read_text())
            notes = excuse_points(slot, ref, new, excused, how)
            res = maps.compare(ref, new, how)
            checks = map_checks(res, limits.map)
            apply_expected(slot, checks, limits.expected)
            status = "FAIL" if any(n["stale"] for n in notes) else slot_status(checks)
            failed += status == "FAIL"
            rows.append({"slot": slot, "status": status, "checks": checks, "detail": res,
                         "excused": notes,
                         "flags": list(new["map"].get("flags", []))})  # fmt: skip
        elif new["kind"] == "geo":
            res = geo_month(json.loads(ref_path.read_text()), new)
            checks = geo_checks(res, limits.geo)
            apply_expected(slot, checks, limits.expected)
            status = slot_status(checks)
            failed += status == "FAIL"
            rows.append({"slot": slot, "status": status, "geo_checks": checks, "detail": res})
        elif new["kind"] == "tree":
            clade_set = clade_set_for(new, clades, clade_cache) if clades else None
            res = trees.compare_figures(json.loads(ref_path.read_text()), new, clade_set)
            checks = tree_checks(res, limits.tree)
            apply_expected(slot, checks, limits.expected)
            status = slot_status(checks)
            failed += status == "FAIL"
            rows.append({"slot": slot, "status": status, "tree_checks": checks, "detail": res})
        else:
            rows.append({"slot": slot, "status": f"{new['kind']}: not compared on I7 yet"})
    return rows, failed


GEO_CHECKS = ("dots by location, frac diff", "dots by clade, frac diff", "same month")


def geo_month(ref: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Compare two one-month geo I7s with compare.geo (which takes a periods list)."""

    def periods(doc: dict[str, Any]) -> dict[str, Any]:
        g = doc["geo"]
        return {"periods": [{"period": g["month"], "locations": g["locations"]}]}

    res = geo.compare(periods(ref), periods(new))
    month = ref["geo"]["month"]
    res["same_month"] = month == new["geo"]["month"]
    res["month"] = res["per_month"].get(month)
    return res


def geo_checks(res: dict[str, Any], lim: GeoLimits) -> list[dict[str, Any]]:
    m = res["month"] or {
        "location": {"frac_diff": float("nan")},
        "clade": {"frac_diff": float("nan")},
    }
    return [
        _check(
            "dots by location, frac diff",
            m["location"]["frac_diff"],
            "<=",
            lim.location_frac_diff_max,
        ),  # fmt: skip
        _check("dots by clade, frac diff", m["clade"]["frac_diff"], "<=", lim.clade_frac_diff_max),
        _check("same month", float(res["same_month"]), ">=", 1.0),  # a different month always fails
    ]


TREE_CHECKS = ("leaves jaccard", "order spearman", "clade ARI (tree)", "section min jaccard",
               "time series same", "sections resolved")  # fmt: skip


def tree_checks(res: dict[str, Any], lim: TreeLimits) -> list[dict[str, Any]]:
    return [
        _check("leaves jaccard", res["jaccard"], ">=", lim.leaves_jaccard_min),
        _check("order spearman", res["order_spearman"], ">=", lim.order_spearman_min),
        _check("clade ARI (tree)", res["clade"]["adjusted_rand"], ">=", lim.clade_ari_min),
        _check(
            "section min jaccard", res["sections"]["min_jaccard"], ">=", lim.section_jaccard_min
        ),  # fmt: skip
        # A different time-series window is always a difference worth failing on.
        _check("time series same", float(res["time_series"]["same"]), ">=", 1.0),
        # A section whose bounds are not drawn leaves is a producer bug: always a failure.
        _check(
            "sections resolved",
            float(
                not (res["sections"]["unresolved"]["ref"] or res["sections"]["unresolved"]["new"])
            ),
            ">=",
            1.0,
        ),  # fmt: skip
    ]


def _cell(check: dict[str, Any]) -> str:
    if check.get("not_tested"):
        return "n/t"
    mark = {False: " ✗", "expected": " (expected)", "stale": " STALE"}.get(check["ok"], "")
    return f"{check['value']:.3f}{mark}"


def markdown(
    manifest: dict[str, Any], rows: list[dict[str, Any]], limits_name: str, how: str,
    adoption: Adoption,
) -> str:  # fmt: skip
    status = (
        f"**Limits {adoption.status.upper()}**: adopted by {adoption.adopted_by} on "
        f"{adoption.adopted.isoformat()}"
        + (f"; review: {adoption.review}" if adoption.review else "")
    )
    changes = [f"- amended {a.date.isoformat()} by {a.by}: {a.change}" for a in adoption.amendments]
    lines = [
        f"# Same-science comparison: {manifest['report']}", "",
        f"Report built {manifest['built']}; limits `{limits_name}`; points matched by {how}.", "",
        status, *changes, "",
        "| Slot | Status | antigens only ref / only new | sera only ref / only new | "
        "antigens in frame only ref / only new | "
        + " | ".join(MAP_CHECKS) + " |",
        "|---|---|---|---|---|" + "---|" * len(MAP_CHECKS),
    ]  # fmt: skip
    notes: list[str] = []
    one_sided: list[str] = []
    flagged: list[str] = []
    tree_rows = [row for row in rows if "tree_checks" in row]
    geo_rows = [row for row in rows if "geo_checks" in row]
    for row in rows:
        if "tree_checks" in row or "geo_checks" in row:
            continue
        if "checks" in row:
            a, s = row["detail"]["antigens"], row["detail"]["sera"]
            counts = (
                f"{a['only_ref']} / {a['only_new']} | {s['only_ref']} / {s['only_new']} | "
                f"{len(a['in_frame_only_ref_keys'])} / {len(a['in_frame_only_new_keys'])}"
            )
            lines.append(f"| {row['slot']} | {row['status']} | {counts} | "
                         + " | ".join(_cell(c) for c in row["checks"]) + " |")  # fmt: skip
            one_sided += _one_sided(row["slot"], a, s)
            notes += [f"- {row['slot']} / {c['check']}: {c['expected']}"
                      for c in row["checks"] if "expected" in c]  # fmt: skip
            notes += [f"- {row['slot']}: {n['note']}" for n in row.get("excused", [])]
            flagged += [f"- {row['slot']}: {flag}" for flag in row.get("flags", [])]
        else:
            lines.append(f"| {row['slot']} | {row['status']} | | | |" + " |" * len(MAP_CHECKS))
    if tree_rows:
        header = "| Tree slot | Status | leaves only ref / only new | " + " | ".join(TREE_CHECKS)
        lines += ["", header + " |", "|---|---|---|" + "---|" * len(TREE_CHECKS)]
        for row in tree_rows:
            d = row["detail"]
            lines.append(f"| {row['slot']} | {row['status']} | {len(d['only_ref_keys'])} / "
                         f"{len(d['only_new_keys'])} | "
                         + " | ".join(_cell(c) for c in row["tree_checks"]) + " |")  # fmt: skip
            notes += [f"- {row['slot']} / {c['check']}: {c['expected']}"
                      for c in row["tree_checks"] if "expected" in c]  # fmt: skip
            sec = d["sections"]
            for side in ("ref", "new"):
                if sec["unresolved"][side]:
                    notes.append(f"- {row['slot']}: {side} sections whose bounds are not drawn "
                                 f"leaves: {sec['unresolved'][side]}")  # fmt: skip
            if sec["only_ref"] or sec["only_new"]:
                notes.append(f"- {row['slot']}: sections only in ref {sec['only_ref']}, "
                             f"only in new {sec['only_new']}")  # fmt: skip
    if geo_rows:
        lines += ["", "| Geo slot | Status | dots ref / new | " + " | ".join(GEO_CHECKS) + " |",
                  "|---|---|---|" + "---|" * len(GEO_CHECKS)]  # fmt: skip
        for row in geo_rows:
            m = row["detail"]["month"] or {"location": {"ref": 0, "new": 0}}
            lines.append(f"| {row['slot']} | {row['status']} | {m['location']['ref']} / "
                         f"{m['location']['new']} | "
                         + " | ".join(_cell(c) for c in row["geo_checks"]) + " |")  # fmt: skip
            notes += [f"- {row['slot']} / {c['check']}: {c['expected']}"
                      for c in row["geo_checks"] if "expected" in c]  # fmt: skip
    if notes:
        lines += ["", "Named differences:", *notes]
    untested = [
        r["slot"] for r in rows if r.get("detail", {}).get("procrustes", {}).get("identical_layout")
    ]
    if untested:
        note = (
            f"**Displacement not tested (n/t) on {len(untested)} map figure(s):** af's layout is "
            "the reference's own, the same up to rotation and translation, so p95, fraction "
            "moved and clade centroids would compare a map with itself. Points shown, clades, "
            "greying and orientation are still tested."
        )
        lines += ["", note]
    if flagged:
        lines += ["", "Flagged by the map step for review (the figure carries these):", *flagged]
    if one_sided:
        lines += ["", "Points on one side only (after excused points):", *one_sided]
    return "\n".join(lines) + "\n"


def _one_sided(slot: str, antigens: dict[str, Any], sera: dict[str, Any]) -> list[str]:
    """List one-sided points for a person to read; long lists are cut, with the count kept."""
    out = []
    for group, g in (("antigens", antigens), ("sera", sera)):
        for side in ("ref", "new"):
            kinds = (
                (f"only_{side}_keys", f"only in {side}"),
                (f"in_frame_only_{side}_keys", f"inside the frame only in {side}"),
            )
            if group == "antigens":
                kinds = (
                    *kinds,
                    (f"vaccine_only_{side}_keys", f"marked as a vaccine only in {side}"),
                )
            for field_name, what in kinds:
                keys = g.get(field_name, [])
                if keys:
                    listed = ", ".join(keys[:ONE_SIDED_LISTED])
                    more = (
                        f" … and {len(keys) - ONE_SIDED_LISTED} more"
                        if len(keys) > ONE_SIDED_LISTED
                        else ""
                    )
                    out.append(f"- {slot} {group} {what} ({len(keys)}): {listed}{more}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a built report with a reference.")
    parser.add_argument("record", type=Path, help="the build record, <report id>.build.json")
    parser.add_argument("reference", type=Path)
    parser.add_argument("--limits", type=Path, required=True)
    parser.add_argument(
        "--match", choices=maps.MATCH_MODES, default="loose",
        help="loose (default): name spelling-normalised + passage class; see maps.spelling_key",
    )  # fmt: skip
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--clades", type=Path,
        help="clades config (TOML): map tree clade labels to canonical names before comparing",
    )  # fmt: skip
    args = parser.parse_args(argv)
    limits = load_config(args.limits, Limits)
    clades = load_config(args.clades, CladesConfig) if args.clades else None
    manifest = json.loads(args.record.read_text())
    rows, failed = compare_report(manifest, args.reference, limits, args.match, clades)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "COMPARISON.json").write_text(json.dumps(rows, indent=1))
    report_md = markdown(manifest, rows, args.limits.name, args.match, limits.adoption)
    (args.out / "COMPARISON.md").write_text(report_md)
    missing = sum(r["status"] == "no reference" for r in rows)
    print(
        f"{len(rows)} figures, {failed} failing, {missing} without reference "
        f"(limits {limits.adoption.status})",
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
