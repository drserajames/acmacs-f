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

The geographic maps get one note per slot (all its months), the same way: from each month
figure's I7 ``provenance`` (the store versions read, the dot rule and how preparations were
merged into a virus, what was not drawn and why, the look, the colour scheme and the user's
tables, ``af.geo.build``), the build record's ``clade_key`` (the key's uncoloured classes) and
the report comparison's rows for the slot.

A fact no artefact records is printed as **MISSING**, with the artefact that should carry it,
and counted (design rule 1): never skipped, never filled in by hand. Configured reasons are
quoted as written.

Run: ``python -m af.report.howmade BUILD_RECORD --store S --maps-config config/maps.toml
--comparison report/comparison/COMPARISON.json --out report/how-made``.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
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
GEO_PREFIX = "geo/"


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


def _built(
    note: Note, files: ChainFiles | None, drawn_by: dict[str, Any] | None, *, stand_in: bool
) -> None:
    """Where the layout was optimised and what drew the figure: a map is repeatable on one
    machine and not across machines, so a reader comparing two builds needs both."""
    if files:
        platform = files.last_step.get("platform", {})
        parts = [platform.get(k) for k in ("system", "machine", "cxx_version")]
        note.item(
            "Layout optimised (last chain step) on "
            + (
                f"{parts[0]} {parts[1]}, compiler {parts[2]}"
                if all(parts)
                else note.gap(
                    "the build platform (system, architecture, compiler)",
                    "last step.json platform.system, .machine, .cxx_version",
                )
            )
            + "; build site "
            + (
                f"`{platform['site']}`"
                if platform.get("site")
                else note.gap("the build site", "last step.json platform.site")
            )
        )  # fmt: skip
    elif stand_in:
        note.item("Layout not optimised by af: a stand-in")
    if not drawn_by or not drawn_by.get("commit"):
        note.item(note.gap("the af that drew the figures", "the build record's af"))
        return
    release = (drawn_by.get("release") or {}).get("commit")
    how = (
        f"release `{release[:12]}`"
        if release and release == drawn_by["commit"] and not drawn_by.get("dirty")
        else "NOT a release" + (", with uncommitted changes" if drawn_by.get("dirty") else "")
    )
    note.item(f"Figures drawn by af `{drawn_by['commit'][:12]}` ({how})")


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
    """The warnings the map step recorded in a figure, verbatim (Sarah: unpainted points, and
    points painted by tree placement alone, Q183 b; vaccines that kept their own colour, Q184)."""
    decisions = doc.get("provenance", {}).get("decisions", {})
    records = [
        *(decisions.get("unpainted_clades") or []),
        *(decisions.get("painted_by_placement") or []),
        *(decisions.get("vaccine_colours") or []),  # only a vaccine that kept its colour (Q184)
    ]
    return [str(r["warning"]) for r in records if r.get("warning")]


def _warnings(note: Note, figures: list[dict[str, Any]]) -> None:
    """Every window's recorded warnings, in the map step's own words; counted apart from MISSING."""
    lines = []
    for fig in figures:
        window = ((fig.get("map") or {}).get("window") or {}).get("name") or (
            (fig.get("geo") or {}).get("month")  # a geographic figure's window is its month
        )
        for warning in figure_warnings(fig):
            where = window or note.gap("the warned figure's window", "figure map.window.name")
            lines.append(f"**WARNING** ({where}): " + warning.removeprefix("WARNING: "))
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


def _continuity(
    note: Note, store: Store | None, files: ChainFiles | None, records: Path | None
) -> None:
    """The rebuilt map against the version it replaces (HISTORY.jsonl's parent), from the
    continuity record written beside the store, ``<records>/<dataset>/<version>.continuity.json``,
    checked against both maps. A first version has nothing to compare with, and says so."""
    from af.chain.continuity import sentences

    if store is None or files is None:
        note.item(note.gap("the chain version's history", "the store's HISTORY.jsonl"))
        return
    ref = files.ref
    published = [e for e in store.history(ref.kind, ref.dataset)
                 if e.get("version") == ref.version and e.get("event") == "published"]  # fmt: skip
    if len(published) != 1:
        what = f"{ref.dataset} {ref.version} published once in HISTORY.jsonl"
        note.item(note.gap(what, f"the store ({len(published)} publish entries)"))
        return
    parent = published[0].get("parent")
    if parent is None:
        note.item(f"First version of {ref.dataset}: no earlier version to compare with "
                  "(HISTORY.jsonl parent is none)")  # fmt: skip
        return
    what = f"the continuity record against {parent}"
    if records is None:
        note.item(note.gap(what, "a continuity record beside the store (no --reference-records)"))
        return
    path = records / ref.dataset / f"{ref.version}.continuity.json"
    if not path.is_file():
        note.item(note.gap(what, str(path)))
        return
    record = json.loads(path.read_text())
    if record.get("map_sha256") != files.chosen_sha256:
        note.item(note.gap(what, f"{path} (its map_sha256 is not this version's map)"))
        return
    if (record.get("previous") or {}).get("version") != parent:
        note.item(note.gap(what, f"{path} (it compares another version, not {parent})"))
        return
    by = record.get("measured_by") or {}
    by = by if isinstance(by, dict) else {}

    def field(value: Any, name: str) -> str:
        if value in (None, ""):
            return note.gap(f"the continuity record's {name}", str(path))
        return str(value)

    method = str(record.get("method_note") or "").rstrip(".")
    note.item(
        f"{method or field(None, 'method_note')}. Measured "
        f"{field(record.get('measured'), 'measured')} by "
        f"{field(by.get('script'), 'measured_by.script')} on "
        f"{field(by.get('release'), 'measured_by.release')} ({path.name})"
    )
    try:
        said = sentences(record)
    except (KeyError, TypeError) as error:  # written by an older af.chain.continuity
        what = "the continuity record's fields as af.chain.continuity reads them"
        note.item(note.gap(what, f"{path} (lacks {error})"))
        return
    for sentence in said:
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


def _same_content(note: Note, folder: str, store_use: dict[str, Any] | None) -> None:
    """A tree under this map exported from another sequences version than the report read, with
    an export record showing the same export: said, with both versions and how the record came to
    be (Sarah's rule, 5 Oct: same content is information, not a refusal)."""
    if not store_use:
        return
    used_by = store_use.get("used_by", {})
    for dataset, (exported, read, path, how) in sorted(
        store_use.get("trees_same_content", {}).items()
    ):
        if any(s.startswith(f"{MAP_PREFIX}{folder}/") for s in used_by.get(f"trees/{dataset}", [])):
            note.item(
                f"Same content, not the same version: trees/{dataset} was exported from "
                f"sequences {exported}; this report read {read}; their exports are identical "
                f"(alignment.fasta and leaves.parquet sha256), per the export record {path} "
                f"({how})"
            )


DECLARATION_FIELDS = ("ruling", "version", "text", "file")
DECLARATION_ABOUT = ("dataset", "figure")


def load_declarations(path: Path) -> list[dict[str, Any]]:
    """The round's declarations: TOML ``[[declaration]]`` tables with ``ruling``, ``version``,
    ``text``, ``file`` (where it is set out in full) and what it is about, one of:

    - ``dataset``, a chain: ``version`` is the chain version the declaration was made on;
    - ``figure = "geo/<slot>"``, a geographic slot: ``version`` is the serology version its
      dots were read from, so the declaration is checked again when the dots can change.
    """
    declarations = tomllib.loads(path.read_text()).get("declaration", [])
    for n, entry in enumerate(declarations, 1):
        about = [k for k in DECLARATION_ABOUT if entry.get(k)]
        if len(about) > 1:
            raise ValueError(f"{path}: declaration {n} names both dataset and figure")
        lacking = [k for k in DECLARATION_FIELDS if not entry.get(k)]
        if lacking or not about:
            lacking = ([] if about else ["dataset or figure"]) + lacking
            raise ValueError(f"{path}: declaration {n} lacks {', '.join(lacking)}")
    return declarations


def _declared(note: Note, mine: list[dict[str, Any]], version: str | None, whose: str) -> None:
    """Quote each declaration made on ``version``; one made on another version is not applied,
    and the declaration this version lacks is counted MISSING."""
    if not mine:
        return
    note.section("Declared")
    for d in mine:
        if d["version"] == version:
            note.item(f"{d['ruling']}: {d['text']} (in full: {d['file']})")
        else:
            gap = note.gap(f"the {d['ruling']} declaration for this version", d["file"])
            note.item(f"{d['ruling']}: declared for version {d['version']}, not for {whose} "
                      f"{version}, so not applied; {gap}")  # fmt: skip


def _declarations(
    note: Note, chain_input: dict[str, Any] | None, declarations: list[dict[str, Any]]
) -> None:
    """What the round declared about this map's chain, quoted with the file that sets it out."""
    if not chain_input:
        return
    mine = [d for d in declarations if d.get("dataset") == chain_input.get("dataset")]
    _declared(note, mine, chain_input.get("version"), "this map's")


def map_note(
    folder: str, figures: list[dict[str, Any]], store: Store | None,
    cfg: dict[str, Any] | None, rows: list[dict[str, Any]],
    reference_records: Path | None = None, dates: VaccineDates | None = None,
    store_use: dict[str, Any] | None = None, drawn_by: dict[str, Any] | None = None,
    declarations: list[dict[str, Any]] | None = None,
) -> Note:  # fmt: skip
    """The note for one map: ``figures`` are its window figures' I7 documents; ``store_use`` is
    the build record's ``store`` (what the report's provenance check found); ``drawn_by`` its
    ``af`` (the af that drew the figures); ``declarations`` the round's
    (:func:`load_declarations`)."""
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
        _same_content(note, folder, store_use)
    else:
        layout = (fig.get("provenance", {}).get("stand_in") or {}).get("layout")
        note.source = "stand-in layout" if layout else "unknown"
        if layout:
            note.mode = "stand-in"
            note.item(f"Not drawn from an af chain: the layout is a stand-in, {layout}")
        else:
            where = "figure provenance.inputs.chain or stand_in"
            note.item(note.gap("the map's source", where))
    note.section("Built")
    _built(note, files, drawn_by, stand_in=note.mode == "stand-in")
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
    note.section("Against the previous version")
    _continuity(note, store, files, reference_records)
    _declarations(note, chain_input, declarations or [])
    _warnings(note, figures)
    note.lines += ["", f"_{len(note.missing)} fact(s) MISSING"
                   + (f": {'; '.join(note.missing)}._" if note.missing else "._")]  # fmt: skip
    return note


# ---------------------------------------------------------------- a geographic slot


def _store_version(note: Note, store: Store | None, ref: dict[str, Any]) -> str:
    """A store version a slot's figures cite, opened through its manifest hash, and whether it
    is CURRENT."""
    kind, dataset, version = ref.get("kind"), ref.get("dataset"), ref.get("version")
    said = f"{str(kind).capitalize()} `{dataset}` version `{version}`"
    if store is None:
        return f"{said}: " + note.gap(f"{kind}/{dataset} {version}", "the store (no --store given)")
    try:
        store.resolve(StoreRef(str(kind), str(dataset), str(version), ref["manifest_sha256"]))
    except (StoreError, OSError, KeyError, ValueError) as error:
        return f"{said}: " + note.gap(f"{kind}/{dataset} {version}", f"the store ({error})")
    try:
        current: str | None = store.current(str(kind), str(dataset)).version
    except StoreError:
        current = None
    where = (
        "the CURRENT version"
        if current == version
        else (f"not CURRENT: CURRENT is {current}" if current else "no CURRENT")
    )
    return f"{said} ({where})"


def _geo_source(note: Note, store: Store | None, figures: list[dict[str, Any]]) -> str | None:
    """The store versions the dots and colours were read from; returns the serology version."""
    prov = figures[0].get("provenance", {})
    cited = {json.dumps(f.get("provenance", {}).get("store_refs"), sort_keys=True) for f in figures}
    if len(cited) > 1:
        note.item(
            "The months of this slot were read from different store versions; this note "
            "reads the first month's"
        )
    refs = prov.get("store_refs")
    if not refs:
        note.item(
            note.gap("the store versions the dots were read from", "figure provenance.store_refs")
        )
        return None
    for ref in refs:
        note.item(_store_version(note, store, ref))
    read = prov.get("store_read")
    if read:
        overrode = [b.get("name") for b in read.get("overrode_batches", [])]
        note.item(
            f"Store read {read.get('started')} ({read.get('read')}): "
            f"{len(read.get('currents_read', {}))} CURRENT(s) read"
            + (
                f"; READ DESPITE batch(es) publishing: {', '.join(map(str, overrode))}"
                if overrode
                else ""
            )
        )
    else:
        note.item(note.gap("the geo build's store read", "figure provenance.store_read"))
    serology = [r for r in refs if r.get("kind") == "serology"]
    if len(serology) != 1:
        note.item(
            note.gap(
                "the one serology version the dots came from",
                f"figure provenance.store_refs ({len(serology)} serology versions)",
            )
        )
        return None
    note.source = f"serology/{serology[0].get('dataset')}@{serology[0].get('version')}"
    return str(serology[0].get("version"))


def _geo_built(
    note: Note, slot: str, prov: dict[str, Any], drawn_by: dict[str, Any] | None
) -> None:
    _built(note, None, drawn_by, stand_in=False)
    note.item(
        "Drawn "
        + _recorded(
            note, prov, "created", "figure provenance.created", "when the figures were drawn"
        )
    )
    look, config = prov.get("look"), prov.get("figure_config")
    if look and config:
        note.item(
            f"Look `[geo.{slot}]` of {Path(config['path']).name} `{config['sha256'][:12]}`: "
            f"{look.get('packing')} packing, dots {_num(look.get('dot_diameter'), 1)} pt, centred "
            f"on longitude {look.get('central_longitude')}"
        )
    else:
        note.item(
            note.gap(
                "the look the figures were drawn with", "figure provenance.look and .figure_config"
            )
        )
    coast = prov.get("coastline")
    note.item(
        "Coastline: "
        + (
            f"{Path(coast['path']).name} `{coast['sha256'][:12]}`"
            if coast
            else note.gap("the coastline", "figure provenance.coastline")
        )
    )
    fonts: dict[str, list[str]] = {}
    for role, font in (prov.get("fonts") or {}).items():
        fonts.setdefault(str(font.get("family")), []).append(role)
    note.item(
        "Fonts: "
        + (
            "; ".join(f"{family} ({', '.join(roles)})" for family, roles in fonts.items())
            if fonts
            else note.gap("the fonts", "figure provenance.fonts")
        )
    )


def _geo_cutoff(note: Note, figures: list[dict[str, Any]]) -> None:
    """The day the data were fetched, when the window's last month was not over (Q243): that
    month is drawn to that day, and its page title says so (af.geo.records.partial_title)."""
    if any("cutoff" not in fig.get("provenance", {}) for fig in figures):
        note.item(note.gap("the day the data were fetched", "figure provenance.cutoff"))
        return
    found = {fig["provenance"]["cutoff"] for fig in figures}
    if found == {None}:
        note.item("No cut-off given: every month's page is titled as the whole month")
        return
    if len(found) > 1:
        note.warnings.append(
            f"the slot's figures record different cut-offs: {sorted(map(str, found))}"
        )
        return
    (cutoff,) = found
    day = dt.date.fromisoformat(cutoff)
    month = f"{day.year:04d}-{day.month:02d}"
    cut = [fig for fig in figures if (fig.get("geo") or {}).get("month") == month]
    last_day = calendar.monthrange(day.year, day.month)[1]
    if cut and day.day < last_day:
        note.item(
            f"Data fetched on {day.day} {calendar.month_name[day.month]} {day.year}: "
            f"{calendar.month_name[day.month]} {day.year} is drawn to that day, and its title "
            "says so"
        )
    else:
        note.item(f"Data fetched on {cutoff}, after the window's months: every month is whole")


def _geo_dots(note: Note, figures: list[dict[str, Any]]) -> None:
    prov = figures[0].get("provenance", {})
    months = []
    for fig in figures:
        month = (fig.get("geo") or {}).get("month", "?")
        drawn = fig.get("provenance", {}).get("drawn")
        months.append(
            f"{month} {drawn}"
            if drawn is not None
            else f"{month} " + note.gap(f"the dots drawn in {month}", "figure provenance.drawn")
        )
    rule = _recorded(note, prov, "dot_rule", "figure provenance.dot_rule", "the dot rule")
    note.item(f"One dot per {rule}; dots drawn in {len(figures)} month(s): {', '.join(months)}")
    _geo_cutoff(note, figures)
    subtype = prov.get("table_subtype")
    merges = (prov.get("merges") or {}).get(subtype) if subtype else None
    if merges is None:
        note.item(
            note.gap(
                "how each virus's preparations were merged",
                "figure provenance.table_subtype and .merges",
            )
        )
    else:
        cases = ", ".join(f"{case} {n}" for case, n in merges.items())
        preps = (prov.get("merged_preparations") or {}).get(subtype)
        note.item(
            f"{sum(merges.values())} {subtype} viruses in the window, from {preps} "
            f"preparations; each virus's colour: {cases}"
        )
    if "undated" in prov and "no_location" in prov:
        note.item(
            f"Not drawn: {prov['undated']} viruses with no collection date (so in no month); "
            f"{prov['no_location']} names in the window with no location the lookup "
            "resolves"
        )
    else:
        note.item(
            note.gap(
                "the viruses not drawn for want of a date or a location",
                "figure provenance.undated and .no_location",
            )
        )
    for key, what in (
        ("no_coordinates", "Not drawn, the location has no coordinates"),
        ("moved_inside_edge", "Clusters moved inward so the map's edge cuts no dot"),
    ):
        said = [
            f"{name} {n} ({(fig.get('geo') or {}).get('month')})"
            for fig in figures
            for name, n in (fig.get("provenance", {}).get(key) or {}).items()
        ]
        if any(key not in fig.get("provenance", {}) for fig in figures):
            note.item(
                f"{what}: "
                + note.gap(f"{key.replace('_', ' ')} per month", f"figure provenance.{key}")
            )
        else:
            note.item(f"{what}: " + (", ".join(said) if said else "none"))


def _geo_colours(
    note: Note, slot: str, prov: dict[str, Any], clade_key: dict[str, Any] | None
) -> None:
    sources = prov.get("colour_source")
    if not sources:
        note.item(
            note.gap("the colour scheme and the user's tables", "figure provenance.colour_source")
        )
    for row, source in sorted((sources or {}).items()):
        tables = ", ".join(
            f"{Path(path).name} `{sha[:12]}`"
            for path, sha in (source.get("user_tables") or {}).items()
        )
        note.item(
            f"Colours, subtype row {row}: scheme `{source.get('scheme')}` "
            f"(source: {source.get('source')}); the user's tables: {tables or 'none'}"
        )
    read = {r.get("dataset") for r in prov.get("store_refs") or []}
    for key in ("clades_behind", "clades_same_content"):  # the build's, for every subtype
        for dataset, entry in (prov.get(key) or {}).items():
            if dataset in read:
                note.item(f"{key.replace('_', ' ').capitalize()}, {dataset}: {json.dumps(entry)}")
    sections = [t for t, s in ((clade_key or {}).get("sections") or {}).items() if s == slot]
    classes = (clade_key or {}).get("uncoloured_classes", {}).get(sections[0]) if sections else None
    if classes is None:
        note.item(
            "Uncoloured: "
            + note.gap(
                "the key's uncoloured dots by class",
                "the build record's clade_key.uncoloured_classes",
            )
        )
        return
    counts = ", ".join(f"{c.replace('_', ' ')} {n}" for c, n in classes.items())
    note.item(
        f"Uncoloured dots in the window, by class (the key on the {sections[0]} page): {counts}"
    )


def _geo_comparison(note: Note, rows: list[dict[str, Any]]) -> None:
    if not rows:
        where = "report/comparison/COMPARISON.json"
        note.item("Report comparison: " + note.gap("the report's comparison of these maps", where))
        return
    approvals: dict[tuple[str, str], list[str]] = {}
    for row in sorted(rows, key=lambda r: str(r["slot"])):
        month = row["slot"].rsplit("/", 1)[-1]
        checks = []
        for c in row.get("geo_checks", []):
            ok = {False: ", FAILING", "expected": ", approved", "stale": ", STALE"}.get(c["ok"], "")
            checks.append(f"{c['check']} {_num(c['value'])} (limit {_num(c['limit'])}{ok})")
            if c["ok"] == "expected":
                text = str(c.get("expected", "")).removeprefix("expected difference: ")
                approvals.setdefault((c["check"], text.split("; found ")[0]), []).append(month)
        note.item(
            f"Report comparison, {month}: {row['status']}"
            + (f"; {'; '.join(checks)}" if checks else "")
        )
    for (check, why), months in approvals.items():
        note.item(f"Approved difference, {check} ({', '.join(months)}): {why}")


def geo_note(
    slot: str,
    figures: list[dict[str, Any]],
    store: Store | None,
    rows: list[dict[str, Any]],
    drawn_by: dict[str, Any] | None = None,
    clade_key: dict[str, Any] | None = None,
    declarations: list[dict[str, Any]] | None = None,
) -> Note:
    """The note for one geographic slot (all its months): ``figures`` are its month figures' I7
    documents; ``clade_key`` the build record's (the key's uncoloured classes); ``declarations``
    the round's, those naming ``figure = "geo/<slot>"``."""
    note = Note(f"geo-{slot}", mode="geo")
    note.lines += [
        f"# geo/{slot}: how these geographic maps were made",
        "",
        "_Written from recorded artefacts; **MISSING** marks a fact no artefact records. "
        '"The ae round" is the round as shipped, the reference._',
    ]
    note.section("Source")
    if not figures:
        note.item(note.gap("the slot's figures", "the build record's geo slots"))
        return note
    figures = sorted(figures, key=lambda f: str((f.get("geo") or {}).get("month")))
    serology = _geo_source(note, store, figures)
    prov = figures[0].get("provenance", {})
    note.section("Built")
    _geo_built(note, slot, prov, drawn_by)
    note.section("Dots")
    _geo_dots(note, figures)
    note.section("Colours")
    _geo_colours(note, slot, prov, clade_key)
    note.section("Against the ae round")
    _geo_comparison(note, rows)
    mine = [d for d in declarations or [] if d.get("figure") == f"geo/{slot}"]
    _declared(note, mine, serology, "the serology these figures read,")
    _warnings(note, figures)
    note.lines += [
        "",
        f"_{len(note.missing)} fact(s) MISSING"
        + (f": {'; '.join(note.missing)}._" if note.missing else "._"),
    ]
    return note


def write_notes(
    record: dict[str, Any], store: Store | None, maps_config: dict[str, Any] | None,
    comparison: list[dict[str, Any]], out: Path, *, ignore_busy: bool = False,
    reference_records: Path | None = None, dates: VaccineDates | None = None,
    declarations: list[dict[str, Any]] | None = None,
) -> list[Note]:  # fmt: skip
    """One note per map folder and per geographic slot in the build record, and an index;
    returns the notes.

    Every note is composed under ``Store.reading`` (refused while a batch publishes, failed if a
    chain's CURRENT moves mid-read), and only then written, so a refused read writes nothing.
    ``ignore_busy`` reads anyway (diagnosis); the index says so.
    """
    by_folder: dict[str, list[dict[str, Any]]] = {}
    by_slot: dict[str, list[dict[str, Any]]] = {}
    for fig in record["figures"]:
        for prefix, into in ((MAP_PREFIX, by_folder), (GEO_PREFIX, by_slot)):
            if fig["slot"].startswith(prefix):
                name = fig["slot"][len(prefix) :].split("/")[0]
                into.setdefault(name, []).append(json.loads(Path(fig["i7"]).read_text()))
    configs = {m["folder"]: m for m in (maps_config or {}).get("maps", [])}
    rows: dict[str, list[dict[str, Any]]] = {}
    geo_rows: dict[str, list[dict[str, Any]]] = {}
    for row in comparison:
        for prefix, into in ((MAP_PREFIX, rows), (GEO_PREFIX, geo_rows)):
            if row["slot"].startswith(prefix):
                into.setdefault(row["slot"][len(prefix) :].split("/")[0], []).append(row)
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
                record.get("store"),
                record.get("af"),
                declarations,
            )  # fmt: skip
            for folder in sorted(by_folder)
        ]
        notes += [
            geo_note(
                slot,
                by_slot[slot],
                store,
                geo_rows.get(slot, []),
                record.get("af"),
                record.get("clade_key"),
                declarations,
            )  # fmt: skip
            for slot in sorted(by_slot)
        ]
    out.mkdir(parents=True, exist_ok=True)
    for note in notes:
        (out / f"{note.folder}.md").write_text("\n".join(note.lines) + "\n")
    geo = f", {len(by_slot)} geographic slot(s)" if by_slot else ""
    index = [f"# How each map was made: {record.get('report', '')}", "",
             f"{len(by_folder)} maps{geo}; {sum(len(n.missing) for n in notes)} fact(s) "
             "MISSING in all. Each note says which artefact a MISSING fact would come from.",
             ""]  # fmt: skip
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
    parser.add_argument(
        "--declarations", type=Path,
        help="the round's declarations (TOML [[declaration]] tables: ruling, dataset or figure, "
        "version, text, file); each map's note quotes those about its chain, each geographic "
        "slot's those naming figure = 'geo/<slot>'",
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
        declarations=load_declarations(args.declarations) if args.declarations else None,
    )  # fmt: skip
    print(f"{len(notes)} notes, {sum(len(n.missing) for n in notes)} fact(s) MISSING -> "
          f"{args.out}", file=sys.stderr)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
