"""af.report.howmade: a note per map, every fact read from a recorded artefact, gaps counted."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

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
        step["platform"] |= {"system": "SysA", "machine": "archB", "cxx_version": "a compiler 1.0",
                             "site": "site-c"}  # fmt: skip
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
    drawn_by = {"commit": "fedcba9876543210", "dirty": False,
                "release": {"commit": "fedcba9876543210"}}  # fmt: skip
    note = howmade.map_note(
        "labx-hi", [_figure(ref, full=True)], store, cfg, rows, drawn_by=drawn_by
    )
    text = "\n".join(note.lines)
    assert note.missing == [], note.missing
    assert note.mode == "merge_all" and note.source == f"labx/hi/merged@{ref.version}"
    for expected in (
        "(the CURRENT version)", "2 tables merged in one step", "labx-t1 to labx-t2",
        "af release `abcdef012345`", "Minimum column basis: none",
        "Layout optimised (last chain step) on SysA archB, compiler a compiler 1.0; build site "
        "`site-c`", "Figures drawn by af `fedcba987654` (release `fedcba987654`)",
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
        "the build platform (system, architecture, compiler)", "the build site",
        "the af that drew the figures",
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
    af = {"commit": "fedcba9876543210", "dirty": False, "release": {"commit": "fedcba9876543210"}}
    record = {"report": "rep-1", "figures": figures, "af": af}
    rows = [_comparison_row("all"), _comparison_row("12m")]
    notes = howmade.write_notes(record, store, {"maps": [{"folder": "labx-hi"}]}, rows,
                                tmp_path / "out")  # fmt: skip
    assert [n.folder for n in notes] == ["labx-hi"]
    text = (tmp_path / "out/labx-hi.md").read_text()
    assert text.startswith("# labx-hi: how this map was made")
    assert "Figures drawn by af `fedcba987654` (release `fedcba987654`)" in text
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
    flagged = _figure(ref, full=True)  # painted by placement (Q183 b): its flag shows as well
    placed = {"tree_clades": {"X": 2}, "points": 2, "warning": "WARNING: placed."}
    flagged["provenance"]["decisions"]["painted_by_placement"] = [placed]
    flagged["provenance"]["decisions"]["unpainted_clades"] = [{"warning": WARNING}]
    assert howmade.figure_warnings(flagged) == [WARNING, "WARNING: placed."]
    vaccines = [{"name": "kept", "warning": "WARNING: kept."}, {"name": "recoloured"}]  # Q184
    flagged["provenance"]["decisions"]["vaccine_colours"] = vaccines
    assert howmade.figure_warnings(flagged) == [WARNING, "WARNING: placed.", "WARNING: kept."]
    del figures[1]["map"]["window"]  # a window the figure doesn't name is counted, never "?"
    note = howmade.map_note("labx-hi", figures, store, None, [])
    assert "the warned figure's window" in note.missing
    assert not any("(?)" in w for w in note.warnings)


def _chain_version(store: Store, chosen: bytes) -> StoreRef:
    step = {"table_id": "labx-t1", "chosen": "scratch", "stress": {"scratch": 1.0},
            "platform": {"release": "abcdef0123456789"}, "diagnostics": {}}  # fmt: skip
    chain = {"mode": "merge_all", "config": {},
             "steps": [{"directory": "steps/0000", "table_id": "labx-t1",
                        "chosen_file": "chosen.ace"}]}  # fmt: skip
    with store.build("chains", "labx/hi/merged") as build:
        (build.path / "steps/0000").mkdir(parents=True)
        (build.path / "steps/0000/step.json").write_text(json.dumps(step))
        (build.path / "steps/0000/chosen.ace").write_bytes(chosen)
        (build.path / "chain.json").write_text(json.dumps(chain))
        return build.publish(Provenance("af.chain", (), {"options": OPTIONS}, T0, T0))


def _continuity_record(ref: StoreRef, parent: StoreRef, chosen: bytes) -> dict[str, Any]:
    import hashlib

    return {
        "dataset": ref.dataset, "version": ref.version,
        "map_sha256": hashlib.sha256(chosen).hexdigest(),
        "previous": {"version": parent.version, "map_sha256": "0" * 64},
        "same_problem": True, "inputs_differ": False, "parameters_differ": [], "retires": None,
        "thresholds": {"path": "t.toml", "sha256": "0" * 64,
                       "apart": [{"distance": 0.5, "reason": "r"}],
                       "movers": {"distance": 1.0, "reason": "r"}},
        "stress": {"new": 11.0, "previous": 10.0, "difference": 1.0, "relative": 0.1},
        "points": {"paired": 7, "added": {"antigens": 1, "sera": 0},
                   "removed": {"antigens": 0, "sera": 0}, "ambiguous": {"count": 0},
                   "filled": {"count": 0, "listed": []}},
        "procrustes": {"compared": 7, "disconnected": 0, "rmsd": 0.3,
                       "apart": [{"distance": 0.5, "count": 2}]},
        "movers": [{"kind": "antigen", "point": "POINT-A", "distance": 1.5}],
        "method_note": "rebuilt map against the version it replaces.",
        "measured": "2026-10-08", "measured_by": {"script": "a script", "release": "a release"},
    }  # fmt: skip


def test_a_rebuilt_map_reads_its_continuity_record_and_a_first_version_says_so(
    tmp_path: Path,
) -> None:
    store = Store.create(tmp_path / "store")
    records = tmp_path / "reference-records"
    first = _chain_version(store, b"first map")
    note = howmade.map_note("labx-hi", [_figure(first, full=True)], store, None, [], records)
    assert any("First version of labx/hi/merged" in line for line in note.lines)
    assert not any("continuity record" in m for m in note.missing)

    second = _chain_version(store, b"second map")
    fig = _figure(second, full=True)
    note = howmade.map_note("labx-hi", [fig], store, None, [], records)
    assert f"the continuity record against {first.version}" in note.missing

    path = records / second.dataset / f"{second.version}.continuity.json"
    path.parent.mkdir(parents=True)
    record = _continuity_record(second, first, b"second map")
    path.write_text(json.dumps(record))
    note = howmade.map_note("labx-hi", [fig], store, None, [], records)
    text = "\n".join(note.lines)
    assert "rebuilt map against the version it replaces. Measured 2026-10-08 by a script" in text
    assert f"Against the previous version {first.version}: 7 points shared" in text
    assert "furthest: POINT-A 1.50" in text
    assert not any("continuity" in m for m in note.missing)

    record["map_sha256"] = "0" * 64  # a record of another map is refused, not quoted
    path.write_text(json.dumps(record))
    note = howmade.map_note("labx-hi", [fig], store, None, [], records)
    assert f"the continuity record against {first.version}" in note.missing
    assert not any("points shared" in line for line in note.lines)

    record["map_sha256"] = _continuity_record(second, first, b"second map")["map_sha256"]
    del record["points"]["filled"]  # a record older than its renderer is a gap, not a crash
    path.write_text(json.dumps(record))
    note = howmade.map_note("labx-hi", [fig], store, None, [], records)
    assert "the continuity record's fields as af.chain.continuity reads them" in note.missing


def test_a_tree_same_in_content_is_said_with_both_versions_and_how_the_record_came(
    tmp_path: Path,
) -> None:
    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=True)
    use = {"used_by": {"trees/bx/weekly": ["map/labx-hi/all"]},
           "trees_same_content": {"bx/weekly": ["v-old", "v-read", "rec.export.json",
                                                "back-filled"]}}  # fmt: skip
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [], None, None, use)
    line = next(x for x in note.lines if "Same content, not the same version" in x)
    assert "exported from sequences v-old; this report read v-read" in line
    assert line.endswith("rec.export.json (back-filled)")
    other = howmade.map_note("labx-other", [_figure(ref, full=True)], store, None, [], None,
                             None, use)  # fmt: skip
    assert not any("Same content" in x for x in other.lines)  # not under this map's slots


def test_figures_drawn_by_a_working_tree_say_it_is_not_a_release(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=True)
    dirty = {"commit": "0123456789abcdef", "dirty": True, "release": None}
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [], drawn_by=dirty)
    assert (
        "Figures drawn by af `0123456789ab` (NOT a release, with uncommitted changes)"
        in "\n".join(note.lines)
    )


def _declarations_file(tmp_path: Path, version: str) -> Path:
    path = tmp_path / "declarations.toml"
    path.write_text(
        "[[declaration]]\n"
        'ruling = "Q1 a"\n'
        'dataset = "labx/hi/merged"\n'
        f'version = "{version}"\n'
        'text = "3 of 5 points sit where a lower position exists; furthest an antigen, 2.50 u"\n'
        'file = "declarations/a-declaration.md"\n'
        "\n[[declaration]]\n"
        'ruling = "Q1 a"\n'
        'dataset = "laby/hi/merged"\n'
        f'version = "{"2" * 16}"\n'
        'text = "another map"\n'
        'file = "declarations/a-declaration.md"\n'
    )
    return path


def test_a_declaration_about_the_maps_chain_version_is_quoted_with_its_file(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=True)
    declarations = howmade.load_declarations(_declarations_file(tmp_path, ref.version))
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [],
                            declarations=declarations)  # fmt: skip
    text = "\n".join(note.lines)
    assert "## Declared" in text
    assert ("- Q1 a: 3 of 5 points sit where a lower position exists; furthest an antigen, 2.50 "
            "u (in full: declarations/a-declaration.md)") in text  # fmt: skip
    assert "another map" not in text
    assert not [m for m in note.missing if "declaration" in m]


def test_a_declaration_about_another_version_is_not_applied_and_counted_missing(
    tmp_path: Path,
) -> None:
    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=True)
    declarations = howmade.load_declarations(_declarations_file(tmp_path, "3" * 16))
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [],
                            declarations=declarations)  # fmt: skip
    text = "\n".join(note.lines)
    other = "3" * 16
    assert f"declared for version {other}, not for this map's {ref.version}, so not applied" in text
    assert "points sit where" not in text
    assert "the Q1 a declaration for this version" in note.missing


def test_a_map_with_no_declaration_has_no_declared_section(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    ref = _chain(store, full=True)
    note = howmade.map_note("labx-hi", [_figure(ref, full=True)], store, None, [])
    assert "## Declared" not in "\n".join(note.lines)


def test_a_declaration_lacking_a_field_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "declarations.toml"
    path.write_text('[[declaration]]\nruling = "Q1 a"\ndataset = "labx/hi/merged"\ntext = "t"\n')
    with pytest.raises(ValueError, match="declaration 1 lacks version, file"):
        howmade.load_declarations(path)


# ---------------------------------------------------------------- geographic slots


def _published(store: Store, kind: str, dataset: str, content: str = "a") -> StoreRef:
    with store.build(kind, dataset) as build:
        (build.path / "data.txt").write_text(content)
        return build.publish(Provenance("a step", (), {}, T0, T0))


def _geo_figure(
    refs: list[StoreRef], month: str, *, full: bool, cutoff: str | None = None
) -> dict[str, Any]:
    """One month of a geographic slot as af.geo.build records it; ``full`` = with the slot's
    own facts (table subtype, colour source, undated and unlocated counts, the cut-off)."""
    prov: dict[str, Any] = {
        "store_refs": [r.to_json() for r in refs],
        "store_read": {"read": "geo-build", "started": "2026-10-02T00:00:00+00:00",
                       "currents_read": {"serology/all": ["x"]}, "overrode_batches": []},
        "created": "2026-10-02T01:00:00+00:00", "dot_rule": "virus",
        "merges": {"A(HX)": {"one preparation": 4, "preparations agree": 1}},
        "merged_preparations": {"A(HX)": 6},
        "look": {"packing": "honeycomb", "dot_diameter": 8.0, "central_longitude": 11.5},
        "figure_config": {"path": "/r/config/figures.toml", "sha256": "c" * 64},
        "coastline": {"path": "/s/coast.shp", "sha256": "d" * 64},
        "fonts": {"font_sans": {"family": "Sans A"}, "font_mono": {"family": "Mono B"}},
        "drawn": 3 if month == "2026-01" else 2,
        "no_coordinates": {"NOWHERE": 1} if month == "2026-02" else {},
        "moved_inside_edge": {"EDGETOWN": 2} if month == "2026-01" else {},
        "clades_behind": {"x1": ["behind"], "x2": ["another subtype's"]},
    }  # fmt: skip
    if full:
        prov |= {"table_subtype": "A(HX)", "undated": 7, "no_location": 1, "cutoff": cutoff,
                 "colour_source": {"x1": {"source": "store", "scheme": "x1 clades",
                                          "user_tables": {"/d/tables.py": "e" * 64}}}}  # fmt: skip
    return {"provenance": prov, "geo": {"subtype": "x1", "month": month, "locations": []}}


def _geo_row(month: str) -> dict[str, Any]:
    checks = [{"check": "dots by location, frac diff", "value": 0.06, "limit": 0.05,
               "ok": "expected", "expected": "expected difference: one dot per virus (approved "
               "by a reviewer, 2026-10-01); found 0.060 against limit 0.05"},
              {"check": "same month", "value": 1.0, "limit": 1.0, "ok": True}]  # fmt: skip
    return {"slot": f"geo/x1/{month}", "status": "ok (expected differences)",
            "geo_checks": checks}  # fmt: skip


CLADE_KEY = {"sections": {"HX geographic data": "x1"},
             "uncoloured_classes": {"HX geographic data": {"not_sequenced": 4,
                                                           "clade_not_defined": 1}}}  # fmt: skip


def _geo_store(tmp_path: Path) -> tuple[Store, list[StoreRef]]:
    store = Store.create(tmp_path / "store")
    refs = [_published(store, "serology", "all"), _published(store, "sequences", "x1"),
            _published(store, "clades", "x1")]  # fmt: skip
    return store, refs


def test_a_geo_slot_note_reads_every_fact_from_its_month_figures(tmp_path: Path) -> None:
    store, refs = _geo_store(tmp_path)
    figures = [_geo_figure(refs, m, full=True) for m in ("2026-02", "2026-01")]
    drawn_by = {"commit": "fedcba9876543210", "dirty": False,
                "release": {"commit": "fedcba9876543210"}}  # fmt: skip
    rows = [_geo_row("2026-01"), _geo_row("2026-02")]
    note = howmade.geo_note("x1", figures, store, rows, drawn_by, CLADE_KEY)
    text = "\n".join(note.lines)
    assert note.missing == [], note.missing
    assert note.mode == "geo" and note.source == f"serology/all@{refs[0].version}"
    assert note.folder == "geo-x1" and text.startswith("# geo/x1: how these geographic maps")
    for expected in (
        f"Serology `all` version `{refs[0].version}` (the CURRENT version)",
        f"Clades `x1` version `{refs[2].version}` (the CURRENT version)",
        "Store read 2026-10-02T00:00:00+00:00 (geo-build): 1 CURRENT(s) read",
        "Figures drawn by af `fedcba987654` (release `fedcba987654`)",
        "Look `[geo.x1]` of figures.toml `cccccccccccc`: honeycomb packing, dots 8.0 pt",
        "Coastline: coast.shp `dddddddddddd`", "Fonts: Sans A (font_sans); Mono B (font_mono)",
        "One dot per virus; dots drawn in 2 month(s): 2026-01 3, 2026-02 2",  # in month order
        "5 A(HX) viruses in the window, from 6 preparations; each virus's colour: one "
        "preparation 4, preparations agree 1",
        "Not drawn: 7 viruses with no collection date (so in no month); 1 names in the window",
        "Not drawn, the location has no coordinates: NOWHERE 1 (2026-02)",
        "Clusters moved inward so the map's edge cuts no dot: EDGETOWN 2 (2026-01)",
        "Colours, subtype row x1: scheme `x1 clades` (source: store); the user's tables: "
        "tables.py `eeeeeeeeeeee`",
        'Clades behind, x1: ["behind"]',
        "Uncoloured dots in the window, by class (the key on the HX geographic data page): not "
        "sequenced 4, clade not defined 1",
        "Report comparison, 2026-01: ok (expected differences); dots by location, frac diff "
        "0.060 (limit 0.050, approved); same month 1.000 (limit 1.000)",
        "Approved difference, dots by location, frac diff (2026-01, 2026-02): one dot per virus",
    ):  # fmt: skip
        assert expected in text, expected
    assert "another subtype's" not in text  # the build's record for the other slots


def test_a_geo_figure_drawn_before_the_slot_facts_says_what_it_lacks(tmp_path: Path) -> None:
    store, refs = _geo_store(tmp_path)
    figures = [_geo_figure(refs, "2026-01", full=False)]
    del figures[0]["provenance"]["moved_inside_edge"]  # drawn before clusters were moved in
    note = howmade.geo_note("x1", figures, store, [])
    assert sorted(note.missing) == sorted([
        "the af that drew the figures", "how each virus's preparations were merged",
        "the viruses not drawn for want of a date or a location", "moved inside edge per month",
        "the colour scheme and the user's tables", "the key's uncoloured dots by class",
        "the report's comparison of these maps", "the day the data were fetched",
    ])  # fmt: skip
    assert note.lines[-1].startswith(f"_{len(note.missing)} fact(s) MISSING")


def test_a_geo_note_says_when_the_data_were_cut_off(tmp_path: Path) -> None:
    """Q243: a monthly report's last month is drawn to the day the data were fetched."""
    store, refs = _geo_store(tmp_path)

    def text(cutoff: str | None) -> str:
        figures = [_geo_figure(refs, m, full=True, cutoff=cutoff) for m in ("2026-01", "2026-02")]
        return "\n".join(howmade.geo_note("x1", figures, store, [], None, CLADE_KEY).lines)

    assert "No cut-off given: every month's page is titled as the whole month" in text(None)
    assert (
        "Data fetched on 21 February 2026: February 2026 is drawn to that day, and its title "
        "says so" in text("2026-02-21")
    )
    assert "Data fetched on 2026-02-28, after the window's months: every month is whole" in text(
        "2026-02-28"
    )


def test_a_geo_slot_citing_a_version_the_store_lacks_says_so(tmp_path: Path) -> None:
    store, refs = _geo_store(tmp_path)
    bad = StoreRef(refs[1].kind, refs[1].dataset, refs[1].version, refs[1].version + "0" * 48)
    newer = _published(store, "clades", "x1", "b")
    note = howmade.geo_note("x1", [_geo_figure([refs[0], bad, refs[2]], "2026-01", full=True)],
                            store, [], None, CLADE_KEY)  # fmt: skip
    text = "\n".join(note.lines)
    assert f"sequences/x1 {refs[1].version}" in note.missing
    assert f"(not CURRENT: CURRENT is {newer.version})" in text
    note = howmade.geo_note("x1", [_geo_figure(refs, "2026-01", full=True)], None, [])
    assert f"serology/all {refs[0].version}" in note.missing  # no store: nothing checked


def test_a_geo_declaration_is_quoted_on_its_serology_version_only(tmp_path: Path) -> None:
    store, refs = _geo_store(tmp_path)
    path = tmp_path / "declarations.toml"
    path.write_text(
        '[[declaration]]\nruling = "Q2 a"\nfigure = "geo/x1"\n'
        f'version = "{refs[0].version}"\ntext = "the key keeps the legend text"\n'
        'file = "declarations/geo-key.md"\n'
        '\n[[declaration]]\nruling = "Q2 a"\nfigure = "geo/x2"\n'
        f'version = "{refs[0].version}"\ntext = "another slot"\nfile = "declarations/geo-key.md"\n'
    )
    declarations = howmade.load_declarations(path)
    figures = [_geo_figure(refs, "2026-01", full=True)]
    note = howmade.geo_note("x1", figures, store, [], None, CLADE_KEY, declarations)
    text = "\n".join(note.lines)
    assert "## Declared" in text and "another slot" not in text
    assert "- Q2 a: the key keeps the legend text (in full: declarations/geo-key.md)" in text
    newer = _published(store, "serology", "all", "b")
    figures = [_geo_figure([newer, *refs[1:]], "2026-01", full=True)]
    note = howmade.geo_note("x1", figures, store, [], None, CLADE_KEY, declarations)
    assert (f"declared for version {refs[0].version}, not for the serology these figures read, "
            f"{newer.version}, so not applied") in "\n".join(note.lines)  # fmt: skip
    assert "the Q2 a declaration for this version" in note.missing


def test_a_declaration_names_a_chain_or_a_figure_never_both(tmp_path: Path) -> None:
    path = tmp_path / "declarations.toml"
    path.write_text('[[declaration]]\nruling = "Q1 a"\nversion = "1"\ntext = "t"\nfile = "f"\n')
    with pytest.raises(ValueError, match="declaration 1 lacks dataset or figure"):
        howmade.load_declarations(path)
    path.write_text('[[declaration]]\nruling = "Q1 a"\ndataset = "d"\nfigure = "geo/x1"\n'
                    'version = "1"\ntext = "t"\nfile = "f"\n')  # fmt: skip
    with pytest.raises(ValueError, match="declaration 1 names both dataset and figure"):
        howmade.load_declarations(path)


def test_a_geo_figures_warnings_are_listed_by_month(tmp_path: Path) -> None:
    store, refs = _geo_store(tmp_path)
    figure = _geo_figure(refs, "2026-01", full=True)
    figure["provenance"]["decisions"] = {"unpainted_clades": [{"warning": WARNING}]}
    note = howmade.geo_note("x1", [figure], store, [], None, CLADE_KEY)
    assert note.warnings == ["**WARNING** (2026-01): " + WARNING.removeprefix("WARNING: ")]


def test_write_notes_writes_a_note_per_geo_slot_beside_the_maps(tmp_path: Path) -> None:
    store, refs = _geo_store(tmp_path)
    figures = []
    for month in ("2026-01", "2026-02"):
        path = tmp_path / f"geo-{month}.i7.json"
        path.write_text(json.dumps(_geo_figure(refs, month, full=True)))
        figures.append({"slot": f"geo/x1/{month}", "i7": str(path)})
    af = {"commit": "fedcba9876543210", "dirty": False, "release": {"commit": "fedcba9876543210"}}
    record = {"report": "rep-1", "figures": figures, "af": af, "clade_key": CLADE_KEY}
    rows = [_geo_row("2026-01"), _geo_row("2026-02")]
    notes = howmade.write_notes(record, store, None, rows, tmp_path / "out")
    assert [n.folder for n in notes] == ["geo-x1"] and notes[0].missing == []
    assert "Report comparison, 2026-02" in (tmp_path / "out/geo-x1.md").read_text()
    index = (tmp_path / "out/README.md").read_text()
    assert "0 maps, 1 geographic slot(s); 0 fact(s) MISSING in all" in index
    assert f"| [geo-x1](geo-x1.md) | serology/all@{refs[0].version} | geo | 0 | 0 |" in index
