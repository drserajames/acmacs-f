"""A human-readable note per map of how it was made, written from recorded artefacts only.

Sarah, 2 Oct 2026: "an automatically generated human readable note in the round that explains
how the map was created - direct from chain, column bases, additional modifications". Every
statement is read from something a step recorded, never recomputed or inferred:

- the map figure's I7: ``provenance.inputs`` (the chain version, the colour source),
  ``provenance.decisions`` (moves, blocks, hides, map-stage column bases, sera, chain_until,
  vaccine rules), ``provenance.stand_in``, ``map.orientation`` and ``map.flags``;
- the chain version in the store (opened through its manifest hash): ``chain.json`` (mode,
  tables, column-basis adjustments, selection, non-ferret sera, reference checks),
  ``PROVENANCE.json`` (seed, optimiser, options) and the last step's ``step.json`` (release,
  stress, diagnostics), and the dataset's ``CURRENT``;
- the round's map config, only to catch a configured change that left no record;
- the report comparison's rows for the map.

A fact no artefact records is printed as **MISSING**, with the artefact that should carry it,
and counted (design rule 1): never skipped, never filled in by hand. Configured reasons are
quoted as written.

Run: ``python -m af.report.howmade BUILD_RECORD --store S --maps-config config/maps.toml
--comparison report/comparison/COMPARISON.json --out report/how-made``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.report.vaccine_dates import VaccineDates, load_who
from af.store import Store, StoreError, StoreRef
from af.util.artefacts import sha256_path

MAP_PREFIX = "map/"


@dataclass
class Note:
    folder: str
    source: str = ""
    mode: str = ""
    lines: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    undated: dict[str, int] = field(default_factory=dict)  # kind of entry -> how many undated
    warnings: list[str] = field(default_factory=list)  # recorded by the map step, verbatim

    def gap(self, what: str, where: str) -> str:
        """A fact no artefact records: counted, and said in place."""
        self.missing.append(what)
        return f"**MISSING**: {what} (would come from {where})"

    def section(self, title: str) -> None:
        self.lines += ["", f"## {title}", ""]

    def item(self, text: str) -> None:
        self.lines.append(f"- {text}")


def _num(value: Any, places: int = 3) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.{places}f}"
    return str(value)


def _count(value: Any) -> Any:
    return len(value) if isinstance(value, (list, dict)) else value


def _recorded(
    note: Note, record: dict[str, Any], key: str, where: str, what: str | None = None
) -> str:
    """A recorded value, or a counted MISSING naming where it would be recorded."""
    if record.get(key) is not None:
        return str(record[key])
    return note.gap(what or key, where)


def _when(note: Note, item: dict[str, Any], category: str) -> str:
    """An entry's decision date; an undated one is counted, and said once per kind of entry."""
    if item.get("decided"):
        return str(item["decided"])
    note.undated[category] = note.undated.get(category, 0) + 1
    return "decision date not recorded"


# ---------------------------------------------------------------- the chain version


@dataclass
class ChainFiles:
    ref: StoreRef
    chain: dict[str, Any]
    provenance: dict[str, Any]
    last_step: dict[str, Any]
    current: str | None
    chosen_sha256: str | None = None  # the final map's file, to check records made beside the store


def open_chain(store: Store, ref: StoreRef) -> ChainFiles:
    """The chain version's records, through the store's manifest-hash check."""
    directory = store.resolve(ref)
    chain = json.loads((directory / "chain.json").read_text())
    provenance = json.loads((directory / "PROVENANCE.json").read_text())
    last = chain["steps"][-1]
    step = json.loads((directory / last["directory"] / "step.json").read_text())
    try:
        current: str | None = store.current(ref.kind, ref.dataset).version
    except StoreError:
        current = None
    chosen = directory / last["directory"] / last.get("chosen_file", "chosen.ace")
    chosen_sha = sha256_path(chosen) if chosen.is_file() else None
    return ChainFiles(ref, chain, provenance, step, current, chosen_sha)


def _source(note: Note, files: ChainFiles, records: Path | None = None) -> None:
    ch, params, step = files.chain, files.provenance.get("parameters", {}), files.last_step
    ref = files.ref
    where = "the CURRENT version" if files.current == ref.version else (
        f"not CURRENT: CURRENT is {files.current}" if files.current else "no CURRENT")  # fmt: skip
    note.item(f"Chain `{ref.dataset}` version `{ref.version}` ({where})")
    note.mode = ch.get("mode", "")
    note.item("Mode: " + (ch["mode"] if "mode" in ch else note.gap(
        "chain or merge_all", "chain.json mode (written since chain-merge-all)")))  # fmt: skip
    src = ch.get("config", {}).get("tables_source")
    note.item("Tables: " + (f"`{src.get('kind')}/{src.get('dataset')}` version "
                            f"`{src.get('version')}`" if src else note.gap(
        "tables version", "chain.json config.tables_source")))  # fmt: skip
    if ch.get("mode") == "merge_all" and "merge_all" in step:
        tables = [t["table_id"] for t in step["merge_all"]]
        note.item(f"{len(tables)} tables merged in one step and optimised from scratch: "
                  f"{tables[0]} to {tables[-1]}" if tables else "no tables merged")  # fmt: skip
    else:
        ids = [s.get("table_id", "?") for s in ch.get("steps", [])]
        note.item(f"{len(ids)} steps, {ids[0]} to {ids[-1]}" if ids else note.gap(
            "chain steps", "chain.json steps"))  # fmt: skip
    release = step.get("platform", {}).get("release")
    opts = params.get("options", {})
    starts = ", ".join(f"{k.split('_')[0]} {opts[k]}" for k in
                       ("scratch_starts", "incremental_starts") if k in opts)  # fmt: skip
    note.item(
        (
            "af release `" + release[:12] + "`"
            if release
            else note.gap("af release", "last step.json platform.release")
        )  # fmt: skip
        + f"; seed {_recorded(note, params, 'seed', 'PROVENANCE.json parameters')}; "
        f"{_recorded(note, params, 'optimiser', 'PROVENANCE.json parameters')}; "
        f"{_recorded(note, opts, 'dimensions', 'PROVENANCE.json parameters.options')} dimensions; "
        "starts: "
        + (starts or note.gap("the start counts", "PROVENANCE.json parameters.options"))
        + "; scratch precision "
        + str(opts.get("scratch_precision", "fine (af.chain records the key only when not fine)"))
    )
    stress, chosen = step.get("stress", {}), step.get("chosen")
    diag = step.get("diagnostics", {})
    if chosen in stress:
        others = ", ".join(f"{k} {_num(v)}" for k, v in stress.items() if k != chosen)

        def diag_value(key: str) -> str:
            if key in diag:
                return str(_count(diag[key]))
            return note.gap(f"the final map's {key.replace('_', ' ')} count", "last step.json "
                            "diagnostics")  # fmt: skip

        note.item(
            f"Final map: stress {_num(stress[chosen])} ({chosen} chosen"
            + (f"; {others}" if others else "")
            + f"); {diag_value('antigens')} antigens, {diag_value('sera')} sera; disconnected "
            f"{diag_value('disconnected')}, trapped {diag_value('trapped')}, cells dropped by the "
            f"SD limit {_sd_dropped(note, files, records)}"
        )
    else:
        note.item(note.gap("final stress", "last step.json stress / chosen"))


def _column_bases(note: Note, files: ChainFiles) -> None:
    opts = files.provenance.get("parameters", {}).get("options", {})
    mcb = opts.get("minimum_column_basis")
    if mcb is None:
        mcb = note.gap(
            "minimum column basis", "PROVENANCE.json parameters.options.minimum_column_basis"
        )
    mode = f"; column bases: {opts['column_bases']}" if "column_bases" in opts else ""
    note.item(f"Minimum column basis: {mcb}{mode}")
    config = files.chain.get("config", {})
    if "column_basis_adjustments" not in config:
        # af.chain writes the key only when the chain has adjustments: absent = none.
        note.item("Named adjustments: none (af.chain records them only when a chain has any)")
        return
    measured = {m.get("rule"): m for m in files.last_step.get("column_basis_adjustments", [])}
    rows = config["column_basis_adjustments"]
    if not rows:
        note.item("Named adjustments: none")
    for row in rows:
        rule = row.get("rule")
        m = measured.get(rule)
        if m:
            effect = f"{m.get('sera_changed')} sera changed, largest {_num(m.get('max_change'))}"
        else:
            effect = note.gap(f"what adjustment {rule} changed", "step.json")
        value = f" {row['value']}" if "value" in row else ""
        when = _when(note, row, "column-basis adjustment(s)")
        note.item(f"Adjustment {rule}{value}: {effect} ({row.get('reason', '')}; {when})")


def _sd_dropped(note: Note, files: ChainFiles, records: Path | None) -> str:
    """Cells the final merge dropped by the SD limit: the last step's diagnostics, else a record.

    ``sd_too_big_cells``, not ``len(dropped_cells)`` (which also holds cells emptied for reading
    both < and >) and not ``merge.outcomes`` (in a merge_all version that sums every
    intermediate merge, each over the whole chart so far). Versions published before the engine
    wrote the count have a record beside the store,
    ``<records>/<dataset>/<version>.merge-drops.json``, checked against the chosen map.
    """
    diag = files.last_step.get("diagnostics", {})
    if "sd_too_big_cells" in diag:
        return str(diag["sd_too_big_cells"]) + _less_and_more(diag)
    what = "the final map's dropped cells count"
    name = f"{files.ref.version}.merge-drops.json"
    path = records / files.ref.dataset / name if records is not None else None
    if path is None or not path.is_file():
        return note.gap(what, f"last step.json diagnostics, or {name} beside the store")
    rec = json.loads(path.read_text())
    if rec.get("map_sha256") != files.chosen_sha256:
        return note.gap(what, f"{path} (its map_sha256 is not this version's map)")
    if "sd_too_big_cells" not in rec:
        return note.gap(what, f"{path.name} sd_too_big_cells")
    return (
        f"{rec['sd_too_big_cells']}{_less_and_more(rec)} (from {path.name}, measured "
        f"{rec.get('measured')} by {rec.get('measured_by')}; cells listed there)"
    )


def _less_and_more(record: dict[str, Any]) -> str:
    n = record.get("less_and_more_than_cells")
    return f", and {n} emptied for reading both < and >" if n else ""


def _repeat_drops_record(note: Note, files: ChainFiles, records: Path | None) -> None:
    """The map's repeat drops against the reference's, from a record beside the store."""
    if records is None:
        return
    path = records / files.ref.dataset / f"{files.ref.version}.repeat-drops.json"
    if not path.is_file():
        what = "the reference comparison of these repeat drops"
        note.item(note.gap(what, f"{path.name} beside the store"))
        return
    rec = json.loads(path.read_text())
    if rec.get("map_sha256") != files.chosen_sha256:
        what = "the reference comparison of these repeat drops"
        note.item(note.gap(what, f"{path} (its map_sha256 is not this version's map)"))
        return
    label = rec.get("reference", {}).get("label", "the reference")
    only_map, only_ref = rec.get("only_map", []), rec.get("only_reference", [])
    note.item(
        f"Of the {rec.get('map_dropped')} cells in this map set to * this way "
        f"({rec.get('tables_with_map_drops')} tables), the reference {label} has no value for "
        f"{rec.get('both')}; it kept a value for {len(only_map)}; and it dropped "
        f"{len(only_ref)} cells this map kept. (Cells paired with the reference by "
        + (
            str(rec["matching"])
            if rec.get("matching")
            else note.gap("how the repeat-drops record paired cells", f"{path.name} matching")
        )  # fmt: skip
        + f"; measured {rec.get('measured')} by {rec.get('measured_by')}; cells listed in "
        f"{path.name}.)"
    )


def _selection(note: Note, files: ChainFiles, records: Path | None = None) -> None:
    config = files.chain.get("config", {})
    if "select_remove" not in config:
        # af.chain writes the key only when the chain has removal rules: absent = none.
        note.item("Named removals: none (af.chain records them only when a chain has any)")
    else:
        removed = files.last_step.get("removed", {})
        rules = config["select_remove"] or []
        if not rules:
            note.item("Named removals: none")
        for rule in rules:
            what = rule.get("what", "?")
            name = rule.get("name") or rule.get("designation") or rule.get("passage") or ""
            count = removed.get(what) if isinstance(removed, dict) else None
            n = f"{_count(count)} point(s)" if count is not None else "count not in step.json"
            note.item(f"Removed by rule {what} {name}: {n} ({rule.get('reason', '')})")
    tables = [t for t in config.get("tables", []) if isinstance(t, dict)]
    dropped = [t for t in tables if t.get("repeat_drops")]
    sd_limit = files.provenance.get("parameters", {}).get("options", {}).get("sd_limit")
    rule = f" (sd_limit {sd_limit})" if sd_limit is not None else ""
    if dropped:
        cells = sum(len(t["repeat_drops"]) for t in dropped)
        note.item(
            f"Within-table repeats: readings of one antigen against one serum in one table are "
            f"merged, and set to * when their spread is too large{rule}: {cells} cell(s) in "
            f"{len(dropped)} of {len(tables)} tables (listed in chain.json "
            "config.tables[].repeat_drops; counted on the tables, so including cells on points "
            "the selection rules then remove)"
        )
        _repeat_drops_record(note, files, records)
    elif tables:
        note.item(
            f"Within-table repeats: no table records a dropped cell{rule} (af.chain lists "
            "repeat_drops per table, when non-zero, since 26 Sep 2026)"
        )
    window = config.get("selection")
    if window:
        note.item(f"Tables selected: {json.dumps(window)}")
    nf = files.chain.get("non_ferret_sera")
    if nf:
        note.item(f"Ferret-only sera: {nf.get('verification')}")
    else:
        where = "chain.json non_ferret_sera"
        note.item("Ferret-only sera: " + note.gap("non-ferret sera removed", where))


# ---------------------------------------------------------------- the map stage


def _refused(entry: dict[str, Any]) -> str:
    """A refused move or block in the guard's own numbers when recorded, else its sentence."""
    if "guard" in entry:
        limit = f"a {entry.get('bound', '?')} limit of {_num(entry.get('limit'))}"
        numbers = f"{_num(entry.get('measured'))} against {limit}"
        return f"REFUSED by the {entry['guard']} guard: {numbers}. {entry.get('why', '')}".strip()
    return f"REFUSED: {entry.get('why', '')}"


def _moves(note: Note, dec: dict[str, Any], cfg: dict[str, Any] | None) -> None:
    recorded = set()
    for mv in dec.get("moves", []):
        recorded.add(mv.get("override"))
        name = mv.get("override")
        if not mv.get("applied"):
            note.item(f"Move {name}: {_refused(mv)}")
            continue
        if mv.get("no_op"):
            effect = "changed nothing"
        else:
            effect = f"worst mover {_num(mv.get('worst_from_target'))} u from target"
        stress = f"stress {_num(mv.get('stress_before'))} -> {_num(mv.get('stress_after'))}"
        when = _when(note, mv, "move(s)")
        note.item(
            f"Move {name}: applied, {mv.get('movers')} movers, {effect}; {stress} "
            f"({mv.get('reason', '')}; {when})"
        )
    for block in dec.get("blocks", []):
        name = block.get("override", "")
        if block.get("applied") is False:
            note.item(f"Block {name}: {_refused(block)}")
        else:
            shift = block.get("shift")
            where = f"({_num(shift[0], 2)}, {_num(shift[1], 2)})" if shift else "?"
            stress = (
                f"stress {_num(block.get('stress_before'))} -> {_num(block.get('stress_after'))}"
            )
            when = _when(note, block, "block(s)")
            note.item(
                f"Block {name}: {block.get('movers', '?')} movers shifted by {where}, "
                f"{block.get('settled', '?')} settled; {stress} ({block.get('reason', '')}; {when})"
            )
    for configured in [m.get("name") for m in (cfg or {}).get("moves", [])]:
        if configured not in recorded:
            what = f"what configured move {configured} did"
            note.item(f"Move {configured}: " + note.gap(what, "figure provenance.decisions.moves"))


def _vaccines(
    note: Note, dec: dict[str, Any], cfg: dict[str, Any] | None,
    dates: VaccineDates | None = None,
) -> None:  # fmt: skip
    rules = dec.get("vaccines")
    if rules is None:
        if (cfg or {}).get("vaccine_choose") or (cfg or {}).get("vaccine_disable"):
            what, where = "the configured vaccine rules as applied", "provenance.decisions.vaccines"
            note.item("Vaccine rules: " + note.gap(what, where))
        return
    for v in rules:
        used = "used" if v.get("used") else "matched nothing"
        if dates is None:
            when = _when(note, v, "vaccine rule(s)")
        else:  # Sarah, 8 Oct (Q126): derived, or "not known", never a counted blank
            date, source = dates.resolve(v)
            when = f"decided {date}: {source}"
        note.item(
            f"Vaccine rule ({v.get('scope', '')}) {v.get('rule', '')} {v.get('name', '')} "
            f"{v.get('passage', '')}: {used} ({v.get('reason', '')}; {when})"
        )


def figure_warnings(doc: dict[str, Any]) -> list[str]:
    """The warnings the map step recorded in a figure, verbatim (Sarah: unpainted points)."""
    records = doc.get("provenance", {}).get("decisions", {}).get("unpainted_clades") or []
    return [str(r["warning"]) for r in records if r.get("warning")]


def _warnings(note: Note, figures: list[dict[str, Any]]) -> None:
    """Every window's recorded warnings, in the map step's own words; counted apart from MISSING."""
    lines = [
        f"**WARNING** ({fig.get('map', {}).get('window', {}).get('name', '?')}): "
        + warning.removeprefix("WARNING: ")
        for fig in figures
        for warning in figure_warnings(fig)
    ]
    if lines:
        note.section("Warnings")
        for line in lines:
            note.item(line)
        note.warnings += lines


def _orientation(note: Note, ori: dict[str, Any] | None) -> None:
    if not ori:
        return
    for o in ori.get("overrides", []):
        turn = f"{o.get('degrees')} deg" + (", reflected" if o.get("reflect") else "")
        when = _when(note, o, "rotation(s)")
        note.item(f"Rotation {o.get('name')}: {turn} ({o.get('reason', '')}; {when})")
    fit = f"{_num(ori.get('fit_degrees'), 1)} deg, RMSD {_num(ori.get('rmsd'))}"
    note.item(f"Orientation fitted to {ori.get('reference')} over {ori.get('common_points')} "
              f"points: {fit}")  # fmt: skip


def _map_stage(
    note: Note, fig: dict[str, Any], cfg: dict[str, Any] | None,
    dates: VaccineDates | None = None,
) -> None:  # fmt: skip
    prov = fig.get("provenance", {})
    dec = prov.get("decisions", {})
    _moves(note, dec, cfg)
    for hide in dec.get("hides", []):
        when = _when(note, hide, "hide(s)")
        reason = hide.get("reason", "")
        note.item(f"Hide {hide.get('name')}: {hide.get('count')} point(s) ({reason}; {when})")
    column_bases = dec.get("column_bases", [])
    for cb in [column_bases] if isinstance(column_bases, dict) else column_bases:
        when = _when(note, cb, "map-stage column-basis change(s)")
        note.item(
            f"Column bases at the map stage, {cb.get('override')}: {cb.get('changed')} of "
            f"{cb.get('sera')} sera changed ({cb.get('reason', '')}; {when})"
        )
    if "chain_until" in dec:
        cu = dec["chain_until"]
        when = _when(note, cu, "chain cut(s)")
        note.item(f"Chain used up to table {cu.get('table')} ({cu.get('reason', '')}; {when})")
    if "sera" in dec:
        note.item(f"Sera check: {dec['sera'].get('verification', '')}")
    else:
        what, where = "the map stage's non-ferret sera check", "figure provenance.decisions.sera"
        note.item("Sera check: " + note.gap(what, where))
    _orientation(note, fig.get("map", {}).get("orientation"))
    _vaccines(note, dec, cfg, dates)
    for key in ("vaccine_defaults", "vaccine_rules_unused_optional"):
        if key in dec:
            note.item(f"{key.replace('_', ' ').capitalize()}: {dec[key]}")
    for name, item in (prov.get("stand_in") or {}).items():
        note.item(f"Stand-in {name}: {item}")
    scheme = prov.get("inputs", {}).get("colour_scheme", {})
    where = "figure provenance.inputs.colour_scheme"
    note.item(f"Colours: {_recorded(note, scheme, 'name', where, 'colour scheme')} "
              f"(source: {_recorded(note, scheme, 'source', where, 'colour source')})")  # fmt: skip
    flags = fig.get("map", {}).get("flags") or []
    note.item("Flags: " + ("; ".join(flags) if flags else "none"))
    for category, n in note.undated.items():
        note.item(note.gap(f"decision dates of {n} {category}", "the config entries' `decided`"))
    note.undated.clear()


# ---------------------------------------------------------------- against the reference


def _references(note: Note, files: ChainFiles | None, records: Path | None) -> None:
    """The chain's own checks against the reference maps: in chain.json for maps built from
    a release that records them; for a published map kept as built (Q110), in a record file
    written beside the store, ``<records>/<dataset>/<version>.json``, checked against the map."""
    from af.chain.reference import sentences

    if files is None:
        return
    refs = files.chain.get("references")
    if refs is None and "reference" in files.chain:  # the single-reference form of 1 Oct
        refs = [{"label": files.chain["reference"].get("chart", "reference"),
                 **files.chain["reference"]}]  # fmt: skip
    if refs is None and records is not None:
        path = records / files.ref.dataset / f"{files.ref.version}.json"
        if path.is_file():
            record = json.loads(path.read_text())
            if record.get("map_sha256") != files.chosen_sha256:
                what = "the reference record for this version"
                note.item(note.gap(what, f"{path} (its map_sha256 is not this version's map)"))
                return
            by = record.get("measured_by") or {}
            by = by if isinstance(by, dict) else {}

            def field(value: Any, name: str) -> str:
                if value in (None, ""):
                    return note.gap(f"the reference record's {name}", str(path))
                return str(value)

            method = str(record.get("method_note") or "").rstrip(".")
            release = field(by.get("release"), "measured_by.release")
            release = release if release.startswith("**MISSING**") else release[:12]
            note.item(
                f"{method or field(None, 'method_note')}. Measured "
                f"{field(record.get('measured'), 'measured')} by "
                f"{field(by.get('script'), 'measured_by.script')} on af "
                f"{release} "
                f"({path.name})"
            )
            refs = record.get("references")
    if refs is None:
        where = "chain.json references, or a reference record beside the store"
        note.item("Chain reference checks: " + note.gap(
            "the chain's checks against the ae round's maps", where))  # fmt: skip
        return
    for entry in refs:
        for sentence in sentences(entry, entry.get("label", "reference")):
            note.item(sentence)


def _comparison(note: Note, rows: list[dict[str, Any]]) -> None:
    if not rows:
        where = "report/comparison/COMPARISON.json"
        note.item("Report comparison: " + note.gap("the report's comparison of this map", where))
        return
    approvals: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        window = row["slot"].rsplit("/", 1)[-1]
        if "checks" not in row:
            note.item(f"Report comparison, {window}: {row['status']}")
            continue
        a, s = row["detail"]["antigens"], row["detail"]["sera"]
        p = row["detail"].get("procrustes", {})
        failing = [c["check"] for c in row["checks"] if c["ok"] is False]
        approved = [c for c in row["checks"] if c["ok"] == "expected"]
        p95 = f"{_num(p['p95'])} u" if "p95" in p else "n/a"
        note.item(
            f"Report comparison, {window}: {row['status']}; antigens Jaccard {_num(a['jaccard'])}, "
            f"sera {_num(s['jaccard'])}; p95 displacement {p95}"
            + (f"; FAILING: {', '.join(failing)}" if failing else "")
            + (f"; approved: {', '.join(c['check'] for c in approved)}" if approved else "")
        )
        for c in approved:
            text = str(c.get("expected", "")).removeprefix("expected difference: ")
            approvals.setdefault((c["check"], text.split("; found ")[0]), []).append(window)
    for (check, why), windows in approvals.items():
        note.item(f"Approved difference, {check} ({', '.join(windows)}): {why}")


# ---------------------------------------------------------------- one map, all maps


def map_note(
    folder: str, figures: list[dict[str, Any]], store: Store | None,
    cfg: dict[str, Any] | None, rows: list[dict[str, Any]],
    reference_records: Path | None = None, dates: VaccineDates | None = None,
) -> Note:  # fmt: skip
    """The note for one map: ``figures`` are its window figures' I7 documents."""
    note = Note(folder)
    note.lines += [
        f"# {folder}: how this map was made",
        "",
        "_Written from recorded artefacts; **MISSING** marks a fact no artefact records. "
        '"The ae round" is the round as shipped, the reference._',
    ]
    if not figures:
        note.section("Source")
        note.item(note.gap("the map's figures", "the build record's map slots"))
        return note
    chains = {json.dumps(f.get("provenance", {}).get("inputs", {}).get("chain"), sort_keys=True)
              for f in figures}  # fmt: skip
    fig = figures[0]
    chain_input = fig.get("provenance", {}).get("inputs", {}).get("chain")
    files: ChainFiles | None = None
    note.section("Source")
    if len(chains) > 1:
        note.item("The windows of this map were drawn from different sources; this note reads "
                  "the first window's")  # fmt: skip
    if chain_input:
        note.source = f"{chain_input.get('dataset')}@{chain_input.get('version')}"
        if store is None:
            note.item(note.gap("the chain's records", "the store (no --store given)"))
        else:
            try:
                ref = StoreRef(chain_input["kind"], chain_input["dataset"],
                               chain_input["version"], chain_input["manifest_sha256"])  # fmt: skip
                files = open_chain(store, ref)
            except (StoreError, OSError, KeyError, ValueError) as error:
                note.item(note.gap("the chain's records", f"the store ({error})"))
        if files:
            _source(note, files, reference_records)
    else:
        layout = (fig.get("provenance", {}).get("stand_in") or {}).get("layout")
        note.source = "stand-in layout" if layout else "unknown"
        if layout:
            note.mode = "stand-in"
            note.item(f"Not drawn from an af chain: the layout is a stand-in, {layout}")
        else:
            where = "figure provenance.inputs.chain or stand_in"
            note.item(note.gap("the map's source", where))
    note.section("Column bases")
    if files:
        _column_bases(note, files)
    else:
        note.item(note.gap("the column bases", "a chain version (this map has none)"))
    note.section("Selection")
    if files:
        _selection(note, files, reference_records)
    else:
        note.item(note.gap("which points were removed", "a chain version (this map has none)"))
    note.section("Changes at the map stage (af.map.build)")
    _map_stage(note, fig, cfg, dates)
    note.section("Against the ae round")
    _references(note, files, reference_records)
    _comparison(note, rows)
    _warnings(note, figures)
    note.lines += ["", f"_{len(note.missing)} fact(s) MISSING"
                   + (f": {'; '.join(note.missing)}._" if note.missing else "._")]  # fmt: skip
    return note


def write_notes(
    record: dict[str, Any], store: Store | None, maps_config: dict[str, Any] | None,
    comparison: list[dict[str, Any]], out: Path, *, ignore_busy: bool = False,
    reference_records: Path | None = None, dates: VaccineDates | None = None,
) -> list[Note]:  # fmt: skip
    """One note per map folder in the build record, and an index; returns the notes.

    Every note is composed under ``Store.reading`` (refused while a batch publishes, failed if a
    chain's CURRENT moves mid-read), and only then written, so a refused read writes nothing.
    ``ignore_busy`` reads anyway (diagnosis); the index says so.
    """
    by_folder: dict[str, list[dict[str, Any]]] = {}
    for fig in record["figures"]:
        if fig["slot"].startswith(MAP_PREFIX):
            folder = fig["slot"][len(MAP_PREFIX) :].split("/")[0]
            by_folder.setdefault(folder, []).append(json.loads(Path(fig["i7"]).read_text()))
    configs = {m["folder"]: m for m in (maps_config or {}).get("maps", [])}
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in comparison:
        if row["slot"].startswith(MAP_PREFIX):
            rows.setdefault(row["slot"][len(MAP_PREFIX) :].split("/")[0], []).append(row)
    guarded = (
        store.reading("howmade", override=ignore_busy) if store is not None else nullcontext(None)
    )
    with guarded as guard:
        notes = [
            map_note(
                folder,
                by_folder[folder],
                store,
                configs.get(folder),
                rows.get(folder, []),
                reference_records,
                dates,
            )  # fmt: skip
            for folder in sorted(by_folder)
        ]
    out.mkdir(parents=True, exist_ok=True)
    for note in notes:
        (out / f"{note.folder}.md").write_text("\n".join(note.lines) + "\n")
    index = [f"# How each map was made: {record.get('report', '')}", "",
             f"{len(notes)} maps; {sum(len(n.missing) for n in notes)} fact(s) MISSING in all. "
             "Each note says which artefact a MISSING fact would come from.", ""]  # fmt: skip
    if guard is not None:
        seen = guard.to_json()
        overrode = [m["name"] for m in seen["overrode_batches"]]
        index += [f"Store read {seen['started']}: {len(seen['currents_read'])} CURRENT(s) read"
                  + (f"; READ DESPITE batch(es) publishing: {', '.join(overrode)}"
                     if overrode else "") + ".", ""]  # fmt: skip
    index += ["| Map | Source | Mode | MISSING | WARNINGS |", "|---|---|---|---|---|"]
    index += [f"| [{n.folder}]({n.folder}.md) | {n.source} | {n.mode or '?'} | {len(n.missing)} "
              f"| {len(n.warnings)} |"
              for n in notes]  # fmt: skip
    (out / "README.md").write_text("\n".join(index) + "\n")
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a how-this-map-was-made note per map.")
    parser.add_argument("record", type=Path, help="the report's build record (<id>.build.json)")
    parser.add_argument("--store", type=Path, help="the af store the maps' chains are in")
    parser.add_argument("--maps-config", type=Path, help="the round's map config (maps.toml)")
    parser.add_argument("--comparison", type=Path, help="the report comparison's COMPARISON.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reference-records", type=Path,
        help="where reference checks of maps kept as built are recorded, beside the store "
        "(<dir>/<dataset>/<version>.json)",
    )  # fmt: skip
    parser.add_argument(
        "--who-recommendations", type=Path,
        help="WHO's recommendations (who-vaccine-recommendations data/*.json): with it, a vaccine "
        "rule's date is derived when config has none (WHO, then git), else 'not known'",
    )  # fmt: skip
    parser.add_argument(
        "--ignore-busy", action="store_true",
        help="read the store even while a batch is publishing (diagnosis only; the index says so)",
    )  # fmt: skip
    args = parser.parse_args(argv)
    record = json.loads(args.record.read_text())
    store = Store.open(args.store) if args.store else None
    maps_config = tomllib.loads(args.maps_config.read_text()) if args.maps_config else None
    comparison = json.loads(args.comparison.read_text()) if args.comparison else []
    dates = None
    if args.who_recommendations:
        files = {}
        if args.maps_config:
            files["map"] = args.maps_config
            if (defaults := (maps_config or {}).get("vaccine_defaults")) is not None:
                files["subtype default"] = (args.maps_config.parent / defaults).resolve()
        dates = VaccineDates(load_who(args.who_recommendations), files)
    notes = write_notes(
        record, store, maps_config, comparison, args.out, ignore_busy=args.ignore_busy,
        reference_records=args.reference_records, dates=dates,
    )  # fmt: skip
    print(f"{len(notes)} notes, {sum(len(n.missing) for n in notes)} fact(s) MISSING -> "
          f"{args.out}", file=sys.stderr)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
