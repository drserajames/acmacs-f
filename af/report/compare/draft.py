"""Draft the known-differences part of a limits file from a comparison, and say what goes stale.

When the figures change wholesale (a new way of building every map), each ``[[expected]]``
entry has to be re-derived: some checks now fail that did not, some entries no longer match a
failing check, and editing them by hand is where an entry lands on a passing check or the wrong
window. This writes, for a built report compared against its reference with the current limits:

- ``DRAFT-expected.toml``: one ``[[expected]]`` entry per failing check that no entry covers
  (with ``--all-failing``, also those an entry already covers, re-measured). Each carries its
  slot, check, the measured value and limit and the af commit, so a person adds only the
  justification and the approval. ``approved_by``, ``decided`` and the reason hold placeholders
  that :func:`af.report.compare.run.load_limits` refuses, so a draft can never load as approved.
- ``DRAFT.md``: the same, plus every existing entry that would go stale, each with **why** (the
  check now passes, is not tested, has no limit; the figure has no such check, is a placeholder,
  has no reference, or is no longer in the report); excused point lists that no longer hold; and
  gaps: figures failing or uncompared that no drafted entry can cover.

Run: ``python -m af.report.compare.draft BUILD_RECORD REFERENCE_DIR --limits L.toml --out DIR``
(``--clades`` and ``--match`` as for :mod:`af.report.compare.run`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.report.compare import maps
from af.report.compare.run import (
    DRAFT_APPROVER,
    DRAFT_DATE,
    DRAFT_REASON,
    Excused,
    Expected,
    Limits,
    Sides,
    add_side_arguments,
    compare_report,
    load_clades,
    load_limits,
    sides_from,
)

CHECK_LISTS = ("checks", "tree_checks", "geo_checks")


@dataclass
class Draft:
    new: list[dict[str, Any]] = field(default_factory=list)  # failing, no entry
    covered: list[dict[str, Any]] = field(default_factory=list)  # failing, entry applies
    stale: list[dict[str, Any]] = field(default_factory=list)  # entry, why it no longer applies
    excused: list[dict[str, Any]] = field(default_factory=list)  # excused lists that no longer hold
    gaps: list[dict[str, Any]] = field(default_factory=list)  # failing/uncompared, nothing to draft


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def _checks(row: dict[str, Any]) -> list[dict[str, Any]]:
    for name in CHECK_LISTS:
        if name in row:
            return list(row[name])
    return []


def _context(row: dict[str, Any], check: str) -> str:
    """Counts a reader needs beside the value, where the comparison has them."""
    d = row.get("detail", {})
    if check in ("antigens jaccard", "sera jaccard"):
        g = d.get(check.split()[0], {})
        only_new, only_ref = len(g.get("only_new_keys", [])), len(g.get("only_ref_keys", []))
        text = f"{only_new} af-only, {only_ref} reference-only"
        lab = g.get("serum_id_lab_dropped")
        if lab and (lab["new"] or lab["ref"]):
            text += (f"; matched after dropping a leading '{lab['lab']}' from serum ids "
                     f"({lab['new']} af, {lab['ref']} reference)")  # fmt: skip
        if rules := maps.serum_rules_text(g):
            text += f"; serum ids matched {rules}"
        return text
    if check in ("p95 displacement", "frac moved > 1", "rotation deg"):
        p = d.get("procrustes", {})
        if p:
            return (f"p95 {_fmt(p['p95'])} u, {_fmt(100 * p['frac_gt_1'])}% > 1 u, "
                    f"rotation {_fmt(p['rotation_deg'])} deg, RMSD {_fmt(p['rmsd'])}")  # fmt: skip
    if check == "clade ARI":
        c = d.get("antigens", {}).get("clade", {})
        return f"compared {c.get('compared')}, top changes {c.get('top_disagreements', [])[:3]}"
    if check == "colour loss":
        c = d.get("antigens", {}).get("colour", {})
        return f"{c.get('lost')} of {c.get('ref_coloured')} reference-coloured antigens uncoloured"
    return ""


def _measured(row: dict[str, Any], check: dict[str, Any], af: str) -> str:
    context = _context(row, check["check"])
    return (f"measured on af {af}: {check['check']} = {_fmt(check['value'])} "
            f"(limit {_fmt(check['limit'])})" + (f"; {context}" if context else ""))  # fmt: skip


def _why_stale(row: dict[str, Any] | None, entry: Expected) -> str | None:
    """Why ``entry`` no longer covers a failing check on its figure; None if it still does."""
    if row is None:
        return "the figure is no longer in the report"
    if row["status"] == "placeholder":
        return "the figure is a placeholder"
    if row["status"] == "no reference":
        return "the figure has no reference to compare with"
    match = [c for c in _checks(row) if c["check"] == entry.check]
    if not match:
        return f"this figure has no {entry.check!r} check"
    check = match[0]
    if check["ok"] == "expected":
        return None
    if check.get("not_tested"):
        return f"the check is not tested ({check['not_tested']})"
    if check["limit"] is None:
        return f"the check has no limit (value {_fmt(check['value'])})"
    if isinstance(check["value"], float) and math.isnan(check["value"]):
        return "the check has no value on this figure"
    return f"the check now passes: {_fmt(check['value'])} within the limit {_fmt(check['limit'])}"


def classify(
    rows: list[dict[str, Any]], limits: Limits, orphans: list[Expected], af: str,
    all_failing: bool,
) -> Draft:  # fmt: skip
    """Sort a comparison's results into drafts, covered checks, stale entries and gaps."""
    out = Draft()
    by_slot = {r["slot"]: r for r in rows}
    entries = {(e.slot, e.check): e for e in limits.expected}
    for row in rows:
        checks = _checks(row)
        if not checks:
            out.gaps.append({"slot": row["slot"], "why": f"not compared: {row['status']}"})
            continue
        for check in checks:
            if check["ok"] not in (False, "expected"):
                continue
            item = {"slot": row["slot"], "check": check["check"], "value": check["value"],
                    "limit": check["limit"], "measured": _measured(row, check, af)}  # fmt: skip
            if check["ok"] is False:
                out.new.append(item)
            else:
                entry = entries[(row["slot"], check["check"])]
                out.covered.append(item | {"entry": entry})
        for note in row.get("excused", []):
            if note.get("stale"):
                out.excused.append({"slot": row["slot"], "note": note["note"]})
        explained = any(c["ok"] in (False, "stale") for c in checks) or any(
            n.get("stale") for n in row.get("excused", [])
        )
        if row["status"] == "FAIL" and not explained:  # nothing in the checks says why
            out.gaps.append({"slot": row["slot"], "why": "fails with no failing check"})
    for entry in [*limits.expected, *orphans]:
        why = _why_stale(by_slot.get(entry.slot), entry)
        if why:
            out.stale.append({"entry": entry, "why": why})
    if all_failing:
        out.new += out.covered
    return out


def _toml_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)  # a JSON string is a valid TOML basic string


def draft_toml(draft: Draft, source: str, sides: Sides | None = None) -> str:
    sides = sides or Sides()
    lines = [
        f"# DRAFT [[expected]] entries from {source}.",
        f"# ref = {sides.ref}; new = {sides.new}.",
        f"# They will NOT load: approved_by, decided and the reason's {DRAFT_REASON!r} are",
        "# placeholders the limits loader refuses. For each, write the justification, then the",
        "# approver and the date of the decision.",
        "",
    ]
    for item in draft.new:
        reason = f"{DRAFT_REASON}: the justification. {item['measured']}"
        lines += [
            "[[expected]]",
            f"slot = {_toml_string(item['slot'])}",
            f"check = {_toml_string(item['check'])}",
            f"reason = {_toml_string(reason)}",
            f"approved_by = {_toml_string(DRAFT_APPROVER)}",
            f"decided = {DRAFT_DATE.isoformat()}",
            "",
        ]
    return "\n".join(lines)


def _entry(entry: Expected) -> str:
    reason = entry.reason if len(entry.reason) <= 160 else entry.reason[:157] + "..."
    return f"{reason} ({entry.approved_by}, {entry.decided.isoformat()})"


def draft_markdown(draft: Draft, source: str, all_failing: bool, sides: Sides | None = None) -> str:
    sides = sides or Sides()
    lines = [f"# Known differences to re-derive: {source}", "",
             f"- **ref** = {sides.ref}", f"- **new** = {sides.new}", ""]  # fmt: skip
    title = "Failing checks to approve" if all_failing else "Failing checks no entry covers"
    lines += [f"## {title} ({len(draft.new)}), drafted in DRAFT-expected.toml", ""]
    lines += [f"- {i['slot']} / {i['check']}: {i['measured']}" for i in draft.new] or ["- none"]
    if not all_failing:
        lines += ["", f"## Failing checks an existing entry covers ({len(draft.covered)})", ""]
        lines += [
            f"- {i['slot']} / {i['check']}: {i['measured']}. Entry: {_entry(i['entry'])}"
            for i in draft.covered
        ] or ["- none"]
    lines += ["", f"## Entries that would go stale ({len(draft.stale)})", ""]
    lines += [
        f"- {s['entry'].slot} / {s['entry'].check}: {s['why']}. Entry: {_entry(s['entry'])}"
        for s in draft.stale
    ] or ["- none"]
    lines += ["", f"## Excused point lists that no longer hold ({len(draft.excused)})", ""]
    lines += [f"- {e['slot']}: {e['note']}" for e in draft.excused] or ["- none"]
    lines += ["", f"## Gaps: failing or uncompared, nothing drafted ({len(draft.gaps)})", ""]
    lines += [f"- {g['slot']}: {g['why']}" for g in draft.gaps] or ["- none"]
    return "\n".join(lines) + "\n"


def run(
    record: dict[str, Any], reference: Path, limits: Limits, how: str, clades: Any,
    all_failing: bool = False,
) -> Draft:  # fmt: skip
    """Compare with the limits' entries for figures still in the report; classify everything."""
    used = {f["slot"] for f in record["figures"]}
    orphans = [e for e in limits.expected if e.slot not in used]
    kept_excused: list[Excused] = []
    for e in limits.excused:
        gone = [s for s in e.slots if s not in used]
        if gone and len(gone) == len(e.slots):
            continue  # reported below, as stale excused lists
        kept_excused.append(dataclasses.replace(e, slots=[s for s in e.slots if s in used]))
    live = dataclasses.replace(
        limits, expected=[e for e in limits.expected if e.slot in used], excused=kept_excused
    )
    rows, _ = compare_report(record, reference, live, how, clades)
    af = str(record.get("af", {}).get("commit") or "unknown")[:12]
    draft = classify(rows, live, orphans, af, all_failing)
    for e in limits.excused:
        gone = [s for s in e.slots if s not in used]
        if gone:
            draft.excused.append({"slot": ", ".join(gone), "note": f"{e.points}: figure(s) no "
                                  "longer in the report"})  # fmt: skip
    return draft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft [[expected]] entries; list stale ones.")
    parser.add_argument("record", type=Path, help="the build record, <report id>.build.json")
    parser.add_argument("reference", type=Path)
    parser.add_argument("--limits", type=Path, required=True)
    parser.add_argument("--match", choices=maps.MATCH_MODES, default="identity")
    parser.add_argument("--clades", type=Path)
    parser.add_argument("--all-failing", action="store_true",
                        help="draft every failing check, also those an entry covers")  # fmt: skip
    parser.add_argument("--out", type=Path, required=True)
    add_side_arguments(parser)
    args = parser.parse_args(argv)
    limits = load_limits(args.limits)
    clades = load_clades(args.clades) if args.clades else None
    record = json.loads(args.record.read_text())
    draft = run(record, args.reference, limits, args.match, clades, args.all_failing)
    source = f"{record.get('report', args.record.name)} vs {args.reference}"
    args.out.mkdir(parents=True, exist_ok=True)
    sides = sides_from(args, record)
    (args.out / "DRAFT-expected.toml").write_text(draft_toml(draft, source, sides))
    (args.out / "DRAFT.md").write_text(draft_markdown(draft, source, args.all_failing, sides))
    print(f"{len(draft.new)} drafted, {len(draft.covered)} covered, {len(draft.stale)} stale, "
          f"{len(draft.excused)} excused lists not holding, {len(draft.gaps)} gaps -> {args.out}",
          file=sys.stderr)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
