"""af.report.compare.draft: drafted entries never load, and every stale entry says why."""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path
from typing import Any

import pytest

from af.report.compare import draft
from af.report.compare.run import (
    DRAFT_APPROVER,
    DRAFT_DATE,
    Excused,
    Expected,
    Limits,
    load_limits,
)
from af.util.config import ConfigError, parse_config

DAY = dt.date(2026, 9, 30)
ADOPTION = {"adoption": {"status": "final", "adopted_by": "a reviewer", "adopted": DAY}}


def _limits(tmp_path: Path, expected: list[Expected]) -> Limits:
    limits = parse_config(ADOPTION, Limits, base_dir=tmp_path)
    return Limits(adoption=limits.adoption, expected=expected)


def _entry(slot: str, check: str) -> Expected:
    return Expected(slot=slot, check=check, reason="a reason", approved_by="a reviewer",
                    decided=DAY)  # fmt: skip


def _check(name: str, value: float, limit: float | None, ok: Any, **extra: Any) -> dict[str, Any]:
    return {"check": name, "value": value, "limit": limit, "ok": ok, **extra}


def test_a_drafted_entry_cannot_be_made_or_loaded(tmp_path: Path) -> None:
    drafts: list[dict[str, Any]] = [
        {"approved_by": DRAFT_APPROVER}, {"approved_by": " "}, {"decided": DRAFT_DATE},
        {"reason": "TO BE WRITTEN: why"},
    ]  # fmt: skip
    for kwargs in drafts:
        values: dict[str, Any] = {"slot": "map/x/all", "check": "clade ARI", "reason": "r",
                                  "approved_by": "a reviewer", "decided": DAY} | kwargs  # fmt: skip
        with pytest.raises(ValueError):
            Expected(**values)
    with pytest.raises(ValueError, match="not approved"):
        Excused(slots=["map/x/all"], points=tmp_path / "k.txt", reason="r",
                approved_by=DRAFT_APPROVER, decided=DAY)  # fmt: skip
    path = tmp_path / "limits.toml"
    path.write_text(
        '[adoption]\nstatus = "final"\nadopted_by = "a reviewer"\nadopted = 2026-09-30\n\n'
        '[[expected]]\nslot = "map/x/all"\ncheck = "clade ARI"\nreason = "TO BE WRITTEN: why"\n'
        f'approved_by = "{DRAFT_APPROVER}"\ndecided = {DRAFT_DATE.isoformat()}\n'
    )
    with pytest.raises(ConfigError) as refused:
        load_limits(path)
    text = str(refused.value)  # every problem is named, not just the first
    assert "not approved" in text and "placeholder" in text and "still contains" in text
    approved = path.read_text().replace(DRAFT_APPROVER, "a reviewer")
    path.write_text(
        approved.replace("TO BE WRITTEN: why", "why").replace("0001-01-01", "2026-09-30")
    )
    assert load_limits(path).expected[0].approved_by == "a reviewer"


def test_classify_drafts_failing_checks_and_says_why_each_entry_is_stale(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {"slot": "map/a/all", "status": "FAIL", "detail": {}, "checks": [
            _check("antigens jaccard", 0.9, 0.99, False),         # failing, no entry: drafted
            _check("clade ARI", 0.8, 0.95, "expected"),           # failing, entry covers it
            _check("p95 displacement", 0.2, 0.5, "stale"),        # entry, now passes
            _check("rotation deg", 3.0, None, None),              # entry, no limit
            _check("frac moved > 1", float("nan"), 0.02, None, not_tested="identical layout"),
        ], "excused": [{"stale": 2, "note": "STALE: 2 of 5 excused points"}]},
        {"slot": "map/b/all", "status": "placeholder"},
        {"slot": "map/c/all", "status": "no reference"},
    ]  # fmt: skip
    entries = [_entry("map/a/all", c) for c in
               ("clade ARI", "p95 displacement", "rotation deg", "frac moved > 1",
                "leaves jaccard")]  # fmt: skip
    entries.append(_entry("map/b/all", "clade ARI"))
    orphan = _entry("map/gone/all", "clade ARI")
    got = draft.classify(rows, _limits(tmp_path, entries), [orphan], "abc123", all_failing=False)
    assert [(i["slot"], i["check"]) for i in got.new] == [("map/a/all", "antigens jaccard")]
    assert [(i["slot"], i["check"]) for i in got.covered] == [("map/a/all", "clade ARI")]
    why = {(s["entry"].slot, s["entry"].check): s["why"] for s in got.stale}
    assert why == {
        ("map/a/all", "p95 displacement"): "the check now passes: 0.2 within the limit 0.5",
        ("map/a/all", "rotation deg"): "the check has no limit (value 3)",
        ("map/a/all", "frac moved > 1"): "the check is not tested (identical layout)",
        ("map/a/all", "leaves jaccard"): "this figure has no 'leaves jaccard' check",
        ("map/b/all", "clade ARI"): "the figure is a placeholder",
        ("map/gone/all", "clade ARI"): "the figure is no longer in the report",
    }
    assert got.excused == [{"slot": "map/a/all", "note": "STALE: 2 of 5 excused points"}]
    assert {g["slot"] for g in got.gaps} == {"map/b/all", "map/c/all"}
    every = draft.classify(rows, _limits(tmp_path, entries), [], "abc123", all_failing=True)
    assert {(i["slot"], i["check"]) for i in every.new} == {
        ("map/a/all", "antigens jaccard"), ("map/a/all", "clade ARI")}  # fmt: skip


def test_a_fail_nothing_explains_is_a_gap(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [{"slot": "map/a/all", "status": "FAIL", "detail": {},
             "checks": [_check("clade ARI", 0.99, 0.95, True)]}]  # fmt: skip
    got = draft.classify(rows, _limits(tmp_path, []), [], "abc123", all_failing=False)
    assert got.gaps == [{"slot": "map/a/all", "why": "fails with no failing check"}]


def test_the_drafted_toml_carries_the_measurement_and_is_refused(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [{"slot": "map/a/12m", "status": "FAIL",
             "detail": {"antigens": {"only_new_keys": ["n1", "n2"], "only_ref_keys": ["r1"]}},
             "checks": [_check("antigens jaccard", 0.875, 0.99, False)]}]  # fmt: skip
    got = draft.classify(rows, _limits(tmp_path, []), [], "abc123", all_failing=False)
    text = draft.draft_toml(got, "a report vs a reference")
    entry = tomllib.loads(text)["expected"][0]
    assert entry["slot"] == "map/a/12m" and entry["check"] == "antigens jaccard"
    assert "antigens jaccard = 0.875 (limit 0.99)" in entry["reason"]
    assert "2 af-only, 1 reference-only" in entry["reason"] and "af abc123" in entry["reason"]
    path = tmp_path / "limits.toml"
    path.write_text(
        '[adoption]\nstatus = "final"\nadopted_by = "a reviewer"\nadopted = 2026-09-30\n\n' + text
    )
    with pytest.raises(ConfigError, match="not approved"):
        load_limits(path)
    md = draft.draft_markdown(got, "a report vs a reference", all_failing=False)
    assert "map/a/12m / antigens jaccard" in md and "## Gaps" in md


def test_both_sides_are_named_in_every_output(tmp_path: Path) -> None:
    import argparse

    from af.report.compare.run import Sides, markdown, sides_from

    sides = Sides(ref="the round as shipped by toolchain X", new="af's rebuild of that round")
    limits = _limits(tmp_path, [])
    text = markdown({"report": "r", "built": "b"}, [], "l.toml", "identity", limits.adoption, sides)
    assert "- **ref** = the round as shipped by toolchain X" in text
    assert "- **new** = af's rebuild of that round" in text
    got = draft.classify([], limits, [], "abc123", all_failing=False)
    assert "**ref** = the round as shipped" in draft.draft_markdown(got, "s", False, sides)
    assert "# ref = the round as shipped" in draft.draft_toml(got, "s", sides)
    # unlabelled: still says what each side is, never a bare "ref"/"new"
    args = argparse.Namespace(ref_label=None, new_label=None, reference=tmp_path / "ref-i7")
    default = sides_from(args, {"report": "rep-1", "af": {"commit": "0123456789abcdef"}})
    assert default.ref == f"the reference I7s in {tmp_path / 'ref-i7'}"
    assert default.new == "af's report rep-1 (af 0123456789ab)"


def test_a_sera_row_says_ids_were_matched_after_dropping_the_lab_token() -> None:
    dropped = {"lab": "LABX", "ref": 0, "new": 5}
    sera = {"only_new_keys": ["a"], "only_ref_keys": [], "serum_id_lab_dropped": dropped}
    row = {"detail": {"sera": sera}}
    text = draft._context(row, "sera jaccard")
    assert text == (
        "1 af-only, 0 reference-only; matched after dropping a leading 'LABX' from serum ids "
        "(5 af, 0 reference)"
    )
