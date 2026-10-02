"""af.report.howmade: a note per map, every fact read from a recorded artefact, gaps counted."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from af.report import howmade
from af.store import Provenance, Store, StoreRef

T0 = dt.datetime(2026, 10, 2, tzinfo=dt.UTC)
OPTIONS = {"minimum_column_basis": "none", "column_bases": "adjust-to-next", "dimensions": 2,
           "scratch_starts": 100}  # fmt: skip


def _reference() -> dict[str, Any]:
    count = {"count": 1}
    return {
        "label": "the reference map",
        "composition": {"antigens": {"map": 10, "reference": 9, "matched": 9},
                        "sera": {"map": 2, "reference": 2, "matched": 2},
                        "map_only": {"antigens": count},
                        "reference_only": {"antigens": {"count": 0}},
                        "jaccard": 0.9},
        "basin": {"seeded_minus_map": 1.0, "relative": 0.01, "rmsd": 0.2, "apart_over_0_5": 1,
                  "apart_over_1": 0, "seeded_points": 9, "points": 12, "starts": 5,
                  "seeded_stress": 101.0, "map_stress": 100.0, "movers": []},
    }  # fmt: skip


def _chain(store: Store, *, full: bool) -> StoreRef:
    """A one-step chain version; ``full`` = with every field the note reads."""
    step: dict[str, Any] = {
        "table_id": "labx-t2", "chosen": "scratch", "stress": {"scratch": 100.0},
        "platform": {"release": "abcdef0123456789"},
        "diagnostics": {"antigens": 10, "sera": 2, "disconnected": [], "trapped": 0,
                        "dropped_cells": [{"cell": "a"}]},
    }  # fmt: skip
    chain: dict[str, Any] = {
        "config": {"tables_source": {"kind": "tables", "dataset": "labx/hi", "version": "1" * 16}},
        "steps": [{"index": 0, "directory": "steps/0000", "table_id": "labx-t2"}],
    }
    if full:
        chain["mode"] = "merge_all"
        step["merge_all"] = [{"table_id": "labx-t1"}, {"table_id": "labx-t2"}]
        chain["config"]["column_basis_adjustments"] = [
            {"rule": "all-sera", "value": -1.0, "reason": "a ruling", "decided": "2026-10-02"}
        ]
        step["column_basis_adjustments"] = [{"rule": "all-sera", "sera_changed": 2,
                                             "max_change": 1.0}]  # fmt: skip
        chain["config"]["select_remove"] = [{"what": "antigen", "name": "a name", "reason": "r"}]
        step["removed"] = {"antigen": ["one"]}
        chain["non_ferret_sera"] = {"verification": "All 2 sera on the map are ferret sera."}
        chain["references"] = [_reference()]
    with store.build("chains", "labx/hi/merged") as build:
        (build.path / "steps/0000").mkdir(parents=True)
        (build.path / "steps/0000/step.json").write_text(json.dumps(step))
        (build.path / "chain.json").write_text(json.dumps(chain))
        params = {"seed": 1, "optimiser": "an optimiser", "options": OPTIONS}
        return build.publish(Provenance("af.chain", (), params, T0, T0))


def _figure(ref: StoreRef | None, *, full: bool) -> dict[str, Any]:
    inputs: dict[str, Any] = {"colour_scheme": {"name": "clades", "source": "store"}}
    prov: dict[str, Any] = {"inputs": inputs, "decisions": {}}
    if ref:
        inputs["chain"] = {"kind": ref.kind, "dataset": ref.dataset, "version": ref.version,
                           "manifest_sha256": ref.manifest_sha256}  # fmt: skip
    else:
        prov["stand_in"] = {"layout": "a stand-in chart"}
    dec = prov["decisions"]
    dec["moves"] = [
        {"override": "m1", "applied": True, "movers": 3, "worst_from_target": 0.5,
         "stress_before": 100.0, "stress_after": 101.0, "reason": "a reason",
         "decided": "2026-10-01" if full else None},
        {"override": "m2", "applied": False, "why": "relaxed back", "guard": "from_target",
         "measured": 4.5, "limit": 4.0, "bound": "max"},
    ]  # fmt: skip
    dec["hides"] = [{"name": "h1", "count": 2, "reason": "r", "decided": "2026-10-01"}]
    dec["column_bases"] = [{"override": "cb", "sera": 2, "changed": 1, "reason": "r",
                            "decided": "2026-10-01"}]  # fmt: skip
    if full:
        dec["sera"] = {"verification": "No non-ferret sera on this map."}
        dec["vaccines"] = [{"rule": "choose", "scope": "map", "name": "a vaccine",
                            "passage": "E1", "reason": "r", "decided": "2026-10-01",
                            "used": True}]  # fmt: skip
    orientation = {"reference": "the last map", "common_points": 9, "fit_degrees": 1.0, "rmsd": 0.1,
                   "overrides": [{"name": "turn", "degrees": 90, "reason": "r",
                                  **({"decided": "2026-10-01"} if full else {})}]}  # fmt: skip
    return {"provenance": prov, "map": {"orientation": orientation, "flags": ["a flag"]}}


def _comparison_row(window: str) -> dict[str, Any]:
    checks = [{"check": "antigens jaccard", "value": 0.9, "limit": 0.99, "ok": "expected",
               "expected": "expected difference: an approved reason (approved by a reviewer, "
               "2026-10-01); found 0.900 against limit 0.99"}]  # fmt: skip
    return {"slot": f"map/labx-hi/{window}", "status": "ok (expected differences)",
            "checks": checks, "detail": {"antigens": {"jaccard": 0.9}, "sera": {"jaccard": 1.0},
                                         "procrustes": {"p95": 0.2}}}  # fmt: skip


def test_a_full_record_leaves_nothing_missing(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=True)
    cfg = {"folder": "labx-hi", "moves": [{"name": "m1"}, {"name": "m2"}]}
    rows = [_comparison_row("all"), _comparison_row("12m")]
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, cfg, rows)
    text = "\n".join(note.lines)
    assert note.missing == [], note.missing
    assert note.mode == "merge_all" and note.source == f"labx/hi/merged@{ref.version}"
    for expected in (
        "(the CURRENT version)", "2 tables merged in one step", "labx-t1 to labx-t2",
        "af release `abcdef012345`", "Minimum column basis: none",
        "Adjustment all-sera -1.0: 2 sera changed", "Removed by rule antigen a name: 1 point(s)",
        "All 2 sera on the map are ferret sera.", "Move m1: applied, 3 movers",
        "Move m2: REFUSED by the from_target guard: 4.500 against a max limit of 4.000",
        "Hide h1: 2 point(s)", "1 of 2 sera changed", "Rotation turn: 90 deg",
        "Vaccine rule (map) choose a vaccine E1: used", "Flags: a flag",
        "Against the reference map: antigens 10 here", "Basin: this map is at or below",
        "Approved difference, antigens jaccard (all, 12m): an approved reason",
    ):  # fmt: skip
        assert expected in text, expected


def test_an_older_record_says_what_it_lacks(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=False)
    cfg = {"folder": "labx-hi", "moves": [{"name": "m1"}, {"name": "m3"}],
           "vaccine_choose": [{"name": "a vaccine"}]}  # fmt: skip
    note = howmade.map_note("labx-hi", [_figure(ref, full=False)], store, cfg, [])
    assert sorted(note.missing) == sorted([
        "chain or merge_all", "named column-basis adjustments", "named removals",
        "non-ferret sera removed", "the map stage's non-ferret sera check",
        "what configured move m3 did", "the configured vaccine rules as applied",
        "decision dates of 1 move(s)", "decision dates of 1 rotation(s)",
        "the chain's checks against the ae round's maps", "the report's comparison of this map",
    ])  # fmt: skip
    assert note.lines[-1].startswith(f"_{len(note.missing)} fact(s) MISSING")


def test_a_stand_in_map_and_an_unreadable_chain(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    note = howmade.map_note("labx-hi", [_figure(None, full=True)], store, None, [])
    assert note.mode == "stand-in" and "the column bases" in note.missing
    assert "which points were removed" in note.missing
    ref = _chain(store, full=True)
    bad = StoreRef(ref.kind, ref.dataset, ref.version, ref.version + "0" * 48)  # wrong manifest
    note = howmade.map_note("labx-hi", [_figure(bad, full=True)], store, None, [])
    assert "the chain's records" in note.missing  # said, not a crash
    assert any("does not match" in line for line in note.lines)


def test_write_notes_writes_one_note_per_map_and_an_index(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=True)
    figures = []
    for window in ("all", "12m"):
        path = tmp_path / f"{window}.i7.json"
        path.write_text(json.dumps(_figure(ref, full=True)))
        figures.append({"slot": f"map/labx-hi/{window}", "i7": str(path)})
    tree = tmp_path / "tree.i7.json"
    tree.write_text("{}")
    figures.append({"slot": "tree/x/report", "i7": str(tree)})  # not a map: no note
    record = {"report": "rep-1", "figures": figures}
    rows = [_comparison_row("all"), _comparison_row("12m")]
    notes = howmade.write_notes(record, store, {"maps": [{"folder": "labx-hi"}]}, rows,
                                tmp_path / "out")  # fmt: skip
    assert [n.folder for n in notes] == ["labx-hi"]
    assert (tmp_path / "out/labx-hi.md").read_text().startswith("# labx-hi: how this map was made")
    index = (tmp_path / "out/README.md").read_text()
    assert "| [labx-hi](labx-hi.md) | labx/hi/merged@" in index and "| merge_all | 0 |" in index
