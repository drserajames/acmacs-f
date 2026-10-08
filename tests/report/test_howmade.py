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
                        "dropped_cells": [{"cell": "a"}], "sd_too_big_cells": 1},
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
        "chain or merge_all",
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


def test_notes_are_not_written_while_the_store_is_being_published(tmp_path: Path) -> None:
    import pytest

    from af.store.busy import StoreBusy

    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=True)
    path = tmp_path / "all.i7.json"
    path.write_text(json.dumps(_figure(ref, full=True)))
    record = {"report": "rep-1", "figures": [{"slot": "map/labx-hi/all", "i7": str(path)}]}
    with store.batch("a-sweep", ["chains/labx/hi/merged"]):
        with pytest.raises(StoreBusy, match="a-sweep"):
            howmade.write_notes(record, store, None, [], tmp_path / "out")
        assert not (tmp_path / "out").exists()
        howmade.write_notes(record, store, None, [], tmp_path / "out", ignore_busy=True)
    index = (tmp_path / "out/README.md").read_text()
    assert "READ DESPITE batch(es) publishing: a-sweep" in index
    howmade.write_notes(record, store, None, [], tmp_path / "out2")
    assert "CURRENT(s) read." in (tmp_path / "out2/README.md").read_text()


def test_a_map_kept_as_built_reads_its_reference_record_beside_the_store(tmp_path: Path) -> None:
    import hashlib

    store = Store.create(tmp_path / "store")
    step = {"table_id": "labx-t1", "chosen": "scratch", "stress": {"scratch": 1.0},
            "platform": {"release": "abcdef0123456789"}, "diagnostics": {}}  # fmt: skip
    chain = {"mode": "merge_all", "config": {},
             "steps": [{"directory": "steps/0000", "table_id": "labx-t1",
                        "chosen_file": "chosen.ace"}]}  # fmt: skip
    with store.build("chains", "labx/hi/merged") as build:
        (build.path / "steps/0000").mkdir(parents=True)
        (build.path / "steps/0000/step.json").write_text(json.dumps(step))
        (build.path / "steps/0000/chosen.ace").write_text("a map")
        (build.path / "chain.json").write_text(json.dumps(chain))
        ref = build.publish(Provenance("af.chain", (), {"options": OPTIONS}, T0, T0))
    records = tmp_path / "reference-records"
    path = records / "labx/hi/merged" / f"{ref.version}.json"
    path.parent.mkdir(parents=True)
    record = {"dataset": "labx/hi/merged", "version": ref.version,
              "map_sha256": hashlib.sha256(b"a map").hexdigest(), "measured": "2026-10-02",
              "measured_by": {"release": "fedcba9876543210", "script": "a script"},
              "method_note": "published map kept as built.",
              "references": [_reference()]}  # fmt: skip
    path.write_text(json.dumps(record))
    fig = _figure(ref, full=True)
    note = howmade.map_note("labx-hi", [fig], store, None, [], records)
    text = "\n".join(note.lines)
    assert "published map kept as built. Measured 2026-10-02 by a script" in text
    assert "Against the reference map: antigens 10 here" in text
    assert "the chain's checks against the ae round's maps" not in note.missing
    assert "scratch precision fine" in text
    record["map_sha256"] = "0" * 64  # a record of another map is refused, not quoted
    path.write_text(json.dumps(record))
    note = howmade.map_note("labx-hi", [fig], store, None, [], records)
    assert "the reference record for this version" in note.missing
    assert not any("Against the reference map" in line for line in note.lines)


def test_a_fact_the_records_lack_is_counted_never_a_question_mark(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    step = {"table_id": "labx-t1", "chosen": "scratch", "stress": {"scratch": 1.0},
            "platform": {"release": "abcdef0123456789"},
            "diagnostics": {"antigens": 3}}  # fmt: skip
    chain = {"mode": "chain", "config": {}, "steps": [{"directory": "steps/0000",
                                                       "table_id": "labx-t1"}]}  # fmt: skip
    with store.build("chains", "labx/hi/main") as build:
        (build.path / "steps/0000").mkdir(parents=True)
        (build.path / "steps/0000/step.json").write_text(json.dumps(step))
        (build.path / "chain.json").write_text(json.dumps(chain))
        ref = build.publish(Provenance("af.chain", (), {"options": {}}, T0, T0))
    fig = _figure(ref, full=True)
    fig["provenance"]["inputs"]["colour_scheme"] = {}
    note = howmade.map_note("labx-hi", [fig], store, None, [])
    text = "\n".join(note.lines)
    assert " ?" not in text and "(?" not in text
    for what in ("seed", "optimiser", "dimensions", "the start counts",
                 "the final map's sera count", "the final map's dropped cells count",
                 "colour scheme", "colour source"):  # fmt: skip
        assert what in note.missing, what


def test_within_table_repeat_drops_are_counted_from_the_tables_record(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    step = {"table_id": "t3", "chosen": "scratch", "stress": {"scratch": 1.0},
            "platform": {"release": "abcdef0123456789"}, "diagnostics": {}}  # fmt: skip
    tables = [{"table_id": "t1", "repeat_drops": ["cell a", "cell b"]}, {"table_id": "t2"},
              {"table_id": "t3", "repeat_drops": ["cell c"]}]  # fmt: skip
    chain = {"mode": "merge_all", "config": {"tables": tables},
             "steps": [{"directory": "steps/0000", "table_id": "t3"}]}  # fmt: skip
    with store.build("chains", "labx/hi/merged") as build:
        (build.path / "steps/0000").mkdir(parents=True)
        (build.path / "steps/0000/step.json").write_text(json.dumps(step))
        (build.path / "chain.json").write_text(json.dumps(chain))
        (build.path / "steps/0000/chosen.ace").write_text("the map")
        options = {**OPTIONS, "sd_limit": 1.0}
        ref = build.publish(Provenance("af.chain", (), {"options": options}, T0, T0))
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [])
    assert any("(sd_limit 1.0): 3 cell(s) in 2 of 3 tables" in line for line in note.lines)
    records = tmp_path / "records"
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [], records)
    assert "the reference comparison of these repeat drops" in note.missing  # no record yet
    import hashlib

    path = records / "labx/hi/merged" / f"{ref.version}.repeat-drops.json"
    path.parent.mkdir(parents=True)
    chosen = store.version_dir(ref) / "steps/0000/chosen.ace"
    sha = hashlib.sha256(chosen.read_bytes()).hexdigest()
    path.write_text(json.dumps({
        "map_sha256": sha, "reference": {"label": "the old map"}, "map_dropped": 2,
        "tables_with_map_drops": 1, "both": 1, "only_map": [{"cell": "x"}], "only_reference": [],
        "measured": "2026-10-02", "measured_by": "a script",
        "matching": "name and passage"}))  # fmt: skip
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [], records)
    assert any("Of the 2 cells in this map set to * this way (1 tables), the reference the old "
               "map has no value for 1; it kept a value for 1; and it dropped 0" in line
               for line in note.lines)  # fmt: skip
    assert "the reference comparison of these repeat drops" not in note.missing
    assert any("paired with the reference by name and passage" in line for line in note.lines)


def test_sd_limit_drops_come_from_the_diagnostics_or_a_record_checked_against_the_map(
    tmp_path: Path,
) -> None:
    import hashlib

    store = Store.create(tmp_path / "store")
    step = {"table_id": "t1", "chosen": "scratch", "stress": {"scratch": 1.0},
            "platform": {"release": "abcdef0123456789"},
            "diagnostics": {"antigens": 3, "sera": 1, "disconnected": 0, "trapped": 0}}  # fmt: skip
    chain = {"mode": "merge_all", "steps": [{"directory": "steps/0000", "table_id": "t1"}]}
    with store.build("chains", "labx/hi/merged") as build:
        (build.path / "steps/0000").mkdir(parents=True)
        (build.path / "steps/0000/step.json").write_text(json.dumps(step))
        (build.path / "chain.json").write_text(json.dumps(chain))
        (build.path / "steps/0000/chosen.ace").write_text("the map")
        ref = build.publish(Provenance("af.chain", (), {"options": OPTIONS}, T0, T0))
    what = "the final map's dropped cells count"
    records = tmp_path / "records"
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [], records)
    assert what in note.missing
    path = records / "labx/hi/merged" / f"{ref.version}.merge-drops.json"
    path.parent.mkdir(parents=True)
    sha = hashlib.sha256((store.version_dir(ref) / "steps/0000/chosen.ace").read_bytes())
    record = {"map_sha256": sha.hexdigest(), "sd_too_big_cells": 4,
              "less_and_more_than_cells": 1, "measured": "2026-10-05", "measured_by": "a tool",
              "outcomes": {"sd-too-big": 4}}  # fmt: skip
    path.write_text(json.dumps(record))
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [], records)
    assert what not in note.missing
    expected = ("cells dropped by the SD limit 4, and 1 emptied for reading both < and > "
                f"(from {path.name}, measured 2026-10-05 by a tool")  # fmt: skip
    assert any(expected in line for line in note.lines)
    path.write_text(json.dumps({**record, "map_sha256": "0" * 64}))  # another map's record
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [], records)
    assert what in note.missing


def test_an_undatable_vaccine_rule_says_not_known_and_is_not_counted(tmp_path: Path) -> None:
    """Sarah, 8 Oct (Q126): a date no source gives is "not known", printed, not a blank."""
    from af.report.vaccine_dates import VaccineDates

    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=True)
    fig = _figure(ref, full=True)
    fig["provenance"]["decisions"]["vaccines"][0].pop("decided")
    without = howmade.map_note("labx-hi", [fig], store, None, [])
    assert any("vaccine rule" in m for m in without.missing)  # no resolver: counted, as before
    note = howmade.map_note("labx-hi", [fig], store, None, [], None, VaccineDates([], {}))
    assert not any("vaccine rule" in m for m in note.missing)
    assert any("decided not known: neither WHO" in line for line in note.lines)


def test_a_reference_record_lacking_a_field_counts_it_missing(tmp_path: Path) -> None:
    import hashlib

    store = Store.create(tmp_path / "store")
    step = {"table_id": "labx-t1", "chosen": "scratch", "stress": {"scratch": 1.0},
            "platform": {"release": "abcdef0123456789"}, "diagnostics": {}}  # fmt: skip
    chain = {"mode": "merge_all", "config": {},
             "steps": [{"directory": "steps/0000", "table_id": "labx-t1",
                        "chosen_file": "chosen.ace"}]}  # fmt: skip
    with store.build("chains", "labx/hi/merged") as build:
        (build.path / "steps/0000").mkdir(parents=True)
        (build.path / "steps/0000/step.json").write_text(json.dumps(step))
        (build.path / "steps/0000/chosen.ace").write_text("a map")
        (build.path / "chain.json").write_text(json.dumps(chain))
        ref = build.publish(Provenance("af.chain", (), {"options": OPTIONS}, T0, T0))
    records = tmp_path / "reference-records"
    path = records / "labx/hi/merged" / f"{ref.version}.json"
    path.parent.mkdir(parents=True)
    record = {"map_sha256": hashlib.sha256(b"a map").hexdigest(),
              "measured_by": {"script": "a script"}, "method_note": "kept as built.",
              "references": [_reference()]}  # fmt: skip
    path.write_text(json.dumps(record))
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [], records)
    text = "\n".join(note.lines)
    assert "?" not in text.split("kept as built")[1].split("\n")[0]
    assert "the reference record's measured" in note.missing
    assert "the reference record's measured_by.release" in note.missing
    assert "the reference record's measured_by.script" not in note.missing


WARNING = ("WARNING: 3 point(s) (1 reference preparation(s)) are not painted: their sequences "
           "support clade X, and colour scheme clades has no row for X.")  # fmt: skip


def test_the_map_steps_warnings_are_listed_verbatim_per_window_and_counted(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=True)
    figures = []
    for window in ("all", "12m"):
        fig = _figure(ref, full=True)
        fig["map"]["window"] = {"name": window}
        record = {"clade": "X", "points": 3, "reference_preparations": 1, "warning": WARNING}
        fig["provenance"]["decisions"]["unpainted_clades"] = [record]
        figures.append(fig)
    note = howmade.map_note("labx-hi", figures, store, None, [])
    assert note.warnings == [
        f"**WARNING** ({w}): " + WARNING.removeprefix("WARNING: ") for w in ("all", "12m")
    ]
    assert "## Warnings" in note.lines and not any("clade X" in m for m in note.missing)
    assert howmade.figure_warnings(figures[0]) == [WARNING]
    assert howmade.figure_warnings(_figure(ref, full=True)) == []  # none recorded: none shown
