"""Re-read every raw table input into the tables store and report what changed.

    python -m af.tables.update tables.toml [--dry-run]

with a config such as (paths relative to the config file)::

    [paths]
    store = "~/AC/eu/store"
    work = "~/AC/eu/work"

    [tables]
    rules = "../acmacs-f-data/rules/tables"
    run = "cdc/all"                          # work-area key: <work>/tables/cdc/all/state/

    [tables.cdc]
    tsv = "../fludata/CDC-Atlanta-WHO-CC/raw-data/CDC_titers_sept_2019_onwards.tsv"
    xlsx = ["../whocc-tables/h3-hi-guinea-pig-cdc/xlsx/<one workbook>.xlsx"]  # tests not in the TSV
    sequence_ids = ["../fludata/CDC-Atlanta-WHO-CC/raw-data/<season>_GISAID.tsv"]  # optional
    sequence_id_ties = "closest-passage"  # required with sequence_ids (Q221; af.tables.cdc_seqids)

    [[tables.vidrl]]                         # a lab's own workbooks, read under source_names names
    lab = "VIDRL"
    dir = "../fludata/VIDRL-Melbourne-WHO-CC/raw-data"   # the originals' root
    names = "vidrl-fludata-h1"              # a source_names set: file -> read_as rows
    start = "2019-04-02"
    subtype = "A(H1N1)"

Every run reads every input (a full re-read), so a rule that matched nothing is an error.
Nothing is published if anything went wrong: format errors, broken tables, unmatched
rules, a workbook duplicating a TSV test. Exit status 1 then.

The run writes two files into its work-area state directory: ``tables-report.txt`` (counts,
rule usage, the diff with its restart dates, errors) and ``tables-published.json`` (every
dataset ref published or reconfirmed). The published list is
the declared output of the pipeline step (:func:`make_step`), so any change to any group
re-runs whatever depends on the tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from af.pipeline.driver import Step, StepContext
from af.store import ExternalInput, PathsConfig, Provenance, Store, Work
from af.util.artefacts import Artefact
from af.util.config import load_config, parse_config

from . import cdc, identity, ids, warnings
from .model import Table
from .passage import PassageParser
from .rules import Rule, Rules
from .store import KIND, current_tables, previous_index, publish

STEP = "tables-update"


@dataclass(frozen=True)
class CDCInputs:
    tsv: Path
    xlsx: list[Path] = field(default_factory=list)
    season: list[Path] = field(default_factory=list)  # older per-season files, scoped by rules
    # CDC's EPI_ISL lookups (the season *_GISAID.tsv files) for antigens without one
    # (af.tables.cdc_seqids); with them, ``sequence_id_ties`` must say how several isolates
    # are decided: "blank", "closest-passage" or "original"
    sequence_ids: list[Path] = field(default_factory=list)
    sequence_id_ties: str = ""


@dataclass(frozen=True)
class LocationSources:
    """Where Chinese location names are romanised from (see af.tables.locations)."""

    locdb: Path  # acmacs-data locationdb.json.xz
    chinese_aliases: Path  # e.g. whocc-tables cnic-location-aliases.tsv


@dataclass(frozen=True)
class AceInputs:
    """A folder of ae's committed .ace charts; only those an ace_imports rule names are read."""

    lab: str
    dir: Path


@dataclass(frozen=True)
class AC21Inputs:
    """One folder of a lab's dated workbooks (AC Excel 2.1, or NIID's layout). ``start``
    bounds what is read: a workbook is read when the YYYYMMDD in its file name is on or after
    it, and its test date must then equal that file-name date (checked), so the bound is on
    the test date. ``exclude`` names workbooks in the folder not to read at all (e.g. one
    whose file name has no usable date); each must exist, so a stale entry is an error.

    ``names`` reads a lab's original workbooks instead of copies: ``dir`` is then their root,
    and the files are the source_names rows of this lab and set, each read under its
    ``read_as`` name (see :func:`_stage_sources`). ``start`` and ``exclude`` apply to those
    names."""

    lab: str
    dir: Path
    start: str  # ISO date
    exclude: list[str] = field(default_factory=list)  # file names
    names: str = ""  # a source_names set, e.g. "vidrl-fludata-h1"


@dataclass(frozen=True)
class VIDRLInputs:
    """A folder of VIDRL workbooks: as AC21Inputs, plus the flu type, which VIDRL's sheets do
    not state (``subtype`` "A(H1N1)", "A(H3N2)" or "B"; ``lineage`` for B). Crick's folders
    take the same fields: there the flu type picks the sheets to read, since one Crick
    workbook can hold several subtypes' tables."""

    lab: str
    dir: Path
    start: str  # ISO date
    subtype: str
    lineage: str = ""
    exclude: list[str] = field(default_factory=list)  # file names
    names: str = ""  # a source_names set: read originals (as AC21Inputs)


@dataclass(frozen=True)
class CrickSheetsInputs:
    """Named sheets of one Crick workbook that is not a whocc-tables copy: a compilation of
    several tests (Sarah, Q72: Crick's own table holds rows its per-test files lacked). Each
    dated workbook it ``replaces`` must be excluded from its folder, so that no test is read
    twice; a test read here and from a folder as well is an error. A test that replaces a
    workbook keeps that workbook's table id (identity.assign), so chains see it change, not
    vanish and reappear under a new id."""

    lab: str
    file: Path
    subtype: str
    sheets: list[str]
    replaces: list[str]  # the whocc-tables workbooks these sheets stand in for
    lineage: str = ""


@dataclass(frozen=True)
class TablesSettings:
    rules: Path
    run: str  # work-area key for this run's state, e.g. "cdc/all" (one run publishes many groups)
    cdc: CDCInputs | None = None
    ac21: list[AC21Inputs] = field(default_factory=list)
    niid: list[AC21Inputs] = field(default_factory=list)  # NIID's own layout (af.tables.niid)
    vidrl: list[VIDRLInputs] = field(default_factory=list)  # VIDRL's layout (af.tables.vidrl)
    crick: list[VIDRLInputs] = field(default_factory=list)  # Crick's layout (af.tables.crick)
    crick_sheets: list[CrickSheetsInputs] = field(default_factory=list)  # named sheets, one file
    ace: list[AceInputs] = field(default_factory=list)  # ae .ace where no workbook exists
    locations: LocationSources | None = None


@dataclass(frozen=True)
class TablesConfig:
    paths: PathsConfig
    tables: TablesSettings


REPORT = "tables-report.txt"
PUBLISHED = "tables-published.json"
SOURCES = "sources"  # in the run's work area: originals linked under their read names
WORKBOOKS = (".xlsx", ".xls", ".xlsm")


def state_dir(paths: PathsConfig, settings: TablesSettings) -> Path:
    """Where the run's report and published list live: the work area, never the store."""
    return Work.open(paths.work).dataset(KIND, settings.run).state


def update(
    settings: TablesSettings, paths: PathsConfig, *, dry_run: bool = False
) -> tuple[int, list[str]]:
    started = dt.datetime.now(dt.UTC)
    rules = Rules(settings.rules)
    store = Store.open(paths.store)
    state = state_dir(paths, settings)
    previous = previous_index(store)
    sources = Work.open(paths.work).dataset(KIND, settings.run).root / SOURCES
    staged, staging = _stage_sources(settings, rules, sources)
    tables, report, errors = _read_all(staged, rules)
    report[:0] = staging
    tables, repeats = exact_repeats(tables)
    report.append(f"tests entered twice (dropped, the first kept): {len(repeats)}")
    report.extend(f"  {line}" for line in repeats)
    errors.extend(identity.assign(tables, previous))
    shape_counts, shape_flags = ids.check(tables, rules)
    report.extend(rules.usage_report())
    report.append(
        "lab id shapes: " + ", ".join(f"{k} {v}" for k, v in sorted(shape_counts.items()))
    )
    report.extend(f"  {line}" for line in shape_flags)
    merged = [w for t in tables for w in t.warnings if w.startswith(cdc.MERGED_ISOLATES)]
    report.append(f"flagged ({cdc.MERGED_ISOLATES}; kept merged, Q13): {len(merged)}")
    report.extend(f"  {w}" for w in merged)
    imported = [r for r in rules.ace_imports.rules if r.hits]
    report.append(f"ae .ace imports, no workbook (ace_imports.tsv): {len(imported)}")
    report.extend(f"  {r.where}: {r['pattern']}: {r['evidence']}" for r in imported)
    held = [r for r in rules.excluded_tables.rules if r.hits]
    report.append(f"held exclusions (excluded_tables.tsv): {len(held)}")
    report.extend(f"  {r.where}: {r['pattern']}: {r['evidence']}" for r in held)
    report.extend(warnings.report(tables))
    errors.extend(warnings.curated_errors(tables))
    errors.extend(f"rule never matched: {r.where}" for t in rules.tables() for r in t.unmatched())
    suffixed = [t.table_id for t in tables if t.date_suffix > 1]
    report.append(f"tables sharing a date (suffix .2+): {len(suffixed)} {suffixed}")
    manifest = identity.Manifest.from_tables(tables, [], previous)
    if previous is not None:
        report.extend(identity.diff(previous, manifest).report())
    else:
        report.append(f"first run: {len(manifest.tables)} tables")
    report.extend(f"ERROR {e}" for e in errors)
    if errors or dry_run:
        return (1 if errors else 0), report
    provenance = Provenance(
        step=STEP,
        inputs=tuple(_external(p) for p in (settings.rules, *_input_files(settings, rules))),
        parameters={},
        started=started,
        finished=dt.datetime.now(dt.UTC),
    )
    results = publish(store, tables, manifest, provenance)
    report.extend(f"{event:11s} {ref}" for ref, event in results)
    published = [{"event": event, "ref": ref.to_json()} for ref, event in results]
    (state / PUBLISHED).write_text(json.dumps(published, indent=1) + "\n", encoding="utf-8")
    _check_round_trip(store, tables)
    return 0, report


def _read_all(settings: TablesSettings, rules: Rules) -> tuple[list[Table], list[str], list[str]]:
    tables, report, errors = _read_cdc(settings.cdc, rules) if settings.cdc else ([], [], [])
    if settings.ac21:
        from . import ac21
        from .locations import ChineseLocations

        if settings.locations is None:
            raise ValueError("[tables.locations] is required to read AC Excel 2.1 workbooks")
        locations = ChineseLocations.load(
            settings.locations.locdb, settings.locations.chinese_aliases
        )
        for inputs in settings.ac21:
            files = _folder_files(inputs, rules, report, errors)
            result = ac21.read(files, rules, locations, lab=inputs.lab)
            _add_workbooks(inputs, files, result, tables, report, errors)
    for inputs in settings.niid:
        from . import niid

        files = _folder_files(inputs, rules, report, errors)
        _add_workbooks(
            inputs, files, niid.read(files, rules, lab=inputs.lab), tables, report, errors
        )
    for vinputs in settings.vidrl:
        from . import vidrl

        files = _folder_files(vinputs, rules, report, errors)
        result = vidrl.read(
            files, rules, lab=vinputs.lab, subtype=vinputs.subtype, lineage=vinputs.lineage
        )
        _add_workbooks(vinputs, files, result, tables, report, errors)
    for cinputs in settings.crick:
        from . import crick

        files = _folder_files(cinputs, rules, report, errors)
        result = crick.read(
            files, rules, lab=cinputs.lab, subtype=cinputs.subtype, lineage=cinputs.lineage
        )
        _add_workbooks(cinputs, files, result, tables, report, errors)
    for sinputs in settings.crick_sheets:
        _add_sheets(settings, sinputs, rules, tables, report, errors)
    for ainputs in settings.ace:
        _add_ace(ainputs, rules, tables, report, errors)
    return tables, report, errors


def _add_ace(
    inputs: AceInputs, rules: Rules, tables: list[Table], report: list[str], errors: list[str]
) -> None:
    """The .ace files an ace_imports rule names, each reported as such in every run. The
    workbook arriving means the import must be retired, so that is an error. A workbook arrives
    under the .ace's own file name with its own extension, so the two pair by file stem: a lab's
    second plate of the day ("…_002") is a different test that shares the first's group and date.
    A table read from a many-test export (a TSV) pairs by group and date."""
    from . import ace_import

    if not inputs.dir.is_dir():
        raise FileNotFoundError(f"ace folder missing: {inputs.dir}")
    arrived: dict[tuple[str, str], str] = {}
    for t in tables:
        name = Path(str(t.meta.get("file", "")))
        one_test = name.suffix.lower() in (".xlsx", ".ace")  # not a many-test export (TSV)
        arrived[(t.group, name.stem if one_test else t.date)] = t.source_key
    for path in sorted(inputs.dir.glob("*.ace")):
        rule = rules.ace_imports.find(path.name, lab=inputs.lab)
        if rule is None:
            continue
        result = ace_import.read(path, rule, rules, lab=inputs.lab)
        errors.extend(result.errors)
        for table in result.tables:
            report.append(
                f"  ACE IMPORT {path.name} ({ace_import.SOURCE}) by {rule.where}: "
                f"{rule['evidence']}"
            )
            other = arrived.get((table.group, path.stem)) or arrived.get((table.group, table.date))
            if other is not None:
                errors.append(
                    f"{path.name}: {other} now reads this test from a workbook; retire {rule.where}"
                )
        tables.extend(result.tables)


def _add_sheets(
    settings: TablesSettings,
    inputs: CrickSheetsInputs,
    rules: Rules,
    tables: list[Table],
    report: list[str],
    errors: list[str],
) -> None:
    from . import crick

    if not inputs.file.is_file():
        raise FileNotFoundError(f"workbook missing: {inputs.file}")
    excluded = {name for folder in settings.crick for name in folder.exclude}
    if missing := [name for name in inputs.replaces if name not in excluded]:
        errors.append(f"{inputs.file.name}: replaces {missing}, not excluded from their folder")
    result = crick.read(
        [inputs.file],
        rules,
        lab=inputs.lab,
        subtype=inputs.subtype,
        lineage=inputs.lineage,
        sheets=inputs.sheets,
    )
    report.append(
        f"read {inputs.lab} {inputs.file.name} sheets {', '.join(inputs.sheets)}"
        f" -> {len(result.tables)} tables (replacing {', '.join(inputs.replaces) or 'nothing'})"
    )
    report.extend(f"skipped: {s}" for s in result.skipped_tests)
    errors.extend(result.errors)
    already = {(t.group, t.date) for t in tables}
    replaced = {_file_date(Path(name)): name for name in inputs.replaces}
    for table in result.tables:
        if (name := replaced.get(table.date)) is not None:
            table.meta["replaces"] = name  # identity.assign: the test keeps that workbook's id
        if (table.group, table.date) in already:
            errors.append(
                f"{inputs.file.name} [{table.meta['sheet']}]: {table.group} {table.date} is"
                " also read from a folder: exclude the workbook it replaces"
            )
    tables.extend(result.tables)


def _add_workbooks(
    inputs: AC21Inputs | VIDRLInputs,
    files: list[Path],
    result: cdc.ReadResult,
    tables: list[Table],
    report: list[str],
    errors: list[str],
) -> None:
    report.append(
        f"read {len(files)} {inputs.lab} workbooks in {inputs.dir} -> {len(result.tables)} tables"
    )
    if inputs.exclude:
        report.append(f"  excluded by config: {', '.join(inputs.exclude)}")
    report.extend(f"skipped: {s}" for s in result.skipped_tests)
    if result.dropped:
        report.append(
            "  dropped: " + ", ".join(f"{k} {v}" for k, v in sorted(result.dropped.items()))
        )
    errors.extend(result.errors)
    for table in result.tables:
        stem_date = _file_date(Path(table.meta["file"]))
        if stem_date != table.date:
            errors.append(
                f"{table.meta['file']}: test date {table.date} != file-name date {stem_date}"
            )
    tables.extend(result.tables)


def _held_out(
    files: list[Path], inputs: AC21Inputs | VIDRLInputs, rules: Rules, report: list[str]
) -> list[Path]:
    """Workbooks an excluded_tables rule holds out are not read. Each is reported with the
    rule and its evidence in every run (a decision such as "exclude now but revisit later"
    must stay visible), and counted in the summary."""
    kept = []
    for path in files:
        rule = rules.excluded_tables.find(path.name, lab=inputs.lab)
        if rule is None:
            kept.append(path)
            continue
        report.append(f"  HELD EXCLUSION {path.name} by {rule.where}: {rule['evidence']}")
    return kept


def _folder_files(
    inputs: AC21Inputs | VIDRLInputs, rules: Rules, report: list[str], errors: list[str]
) -> list[Path]:
    """The folder's workbooks to read: dated on or after ``start`` and not held out. An
    undated file name is counted as an error here and the rest of the folder is still read."""
    files, undated = _dated_files(inputs)
    errors.extend(undated)
    return _held_out(files, inputs, rules, report)


def _file_date(path: Path) -> str | None:
    m = re.search(r"(\d{4})(\d{2})(\d{2})", path.stem)
    return dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat() if m else None


def _dated_files(inputs: AC21Inputs | VIDRLInputs) -> tuple[list[Path], list[str]]:
    """Workbooks in the folder dated on or after ``start``; Excel lock files (~$) are not
    workbooks. A workbook whose name carries no date is an error, not silently skipped: it is
    returned as one, per file, so that one bad name does not stop the rest of the folder."""
    if not inputs.dir.is_dir():
        raise FileNotFoundError(f"workbook folder missing: {inputs.dir}")
    start = dt.date.fromisoformat(inputs.start).isoformat()
    if missing := [n for n in inputs.exclude if not (inputs.dir / n).is_file()]:
        raise FileNotFoundError(f"{inputs.dir}: excluded workbooks not found: {missing}")
    out, errors = [], []
    for path in sorted(inputs.dir.glob("*.xlsx")):
        if path.name.startswith("~$") or path.name in inputs.exclude:
            continue
        day = _file_date(path)
        if day is None:
            errors.append(f"{path}: no YYYYMMDD date in the file name")
            continue
        if dt.date.fromisoformat(day) >= dt.date.fromisoformat(start):
            out.append(path)
    return out, errors


def _read_cdc(inputs: CDCInputs, rules: Rules) -> tuple[list[Table], list[str], list[str]]:
    result = cdc.read(inputs.tsv, rules)
    report = [f"read {inputs.tsv}: {result.rows} rows -> {len(result.tables)} tables"]
    report.append("dropped: " + ", ".join(f"{k} {v}" for k, v in sorted(result.dropped.items())))
    report.extend(f"skipped: {s}" for s in result.skipped_tests)
    errors = list(result.errors)
    tables = list(result.tables)
    if inputs.xlsx:
        from . import cdc_xlsx  # needs openpyxl; imported only when there are workbooks

        xl = cdc_xlsx.read(list(inputs.xlsx), rules)
        report.append(f"read {len(inputs.xlsx)} CDC workbooks -> {len(xl.tables)} tables")
        report.extend(f"skipped: {s}" for s in xl.skipped_tests)
        errors.extend(xl.errors)
        errors.extend(duplicates(result.tables, xl.tables))
        tables.extend(xl.tables)
    main = list(tables)  # the TSV and the workbooks: season tables join their lots and dates
    for season_file in inputs.season:
        from . import cdc_season

        season = cdc_season.read(season_file, rules)
        report.append(
            f"read {season_file}: {season.rows} rows in scope -> {len(season.tables)} tables"
        )
        report.append(
            "season file: " + ", ".join(f"{k} {v}" for k, v in sorted(season.dropped.items()))
        )
        errors.extend(season.errors)
        errors.extend(duplicates(result.tables, season.tables))
        errors.extend(duplicates([t for t in main if " xlsx " in t.source_key], season.tables))
        if lots := cdc_season.join_lots(season.tables, main):
            report.append("  lots: " + ", ".join(f"{k} {v}" for k, v in sorted(lots.items())))
        joined = cdc_season.join_harvest_dates(season.tables, main)
        if joined:
            report.append(
                "  harvest dates: " + ", ".join(f"{k} {v}" for k, v in sorted(joined.items()))
            )
        tables.extend(season.tables)
    if inputs.sequence_ids:
        report.append(_join_sequence_ids(inputs, rules, main, tables))
    return tables, report, errors


def _join_sequence_ids(
    inputs: CDCInputs, rules: Rules, main: list[Table], tables: list[Table]
) -> str:
    from . import cdc_seqids

    if not inputs.sequence_id_ties:
        raise ValueError(
            "[tables.cdc] sequence_ids needs sequence_id_ties: " + ", ".join(cdc_seqids.TIES)
        )
    passages = PassageParser(rules.passage_tokens, cdc.LAB)

    def classify(raw: str) -> tuple[str, str]:
        text = passages.parse(raw).text
        return text, passages.passage_class(text)

    lookups = [cdc_seqids.read_lookup(p, classify) for p in inputs.sequence_ids]
    lookups.append(cdc_seqids.from_tables(main))
    counts = cdc_seqids.join(tables, lookups, inputs.sequence_id_ties)
    return f"sequence ids ({inputs.sequence_id_ties}): " + ", ".join(
        f"{k} {v}" for k, v in sorted(counts.items())
    )


def _input_files(settings: TablesSettings, rules: Rules | None = None) -> list[Path]:
    """Every raw input of a run; for a ``names`` entry the originals themselves, not the links
    the run reads them through."""
    files: list[Path] = []
    if settings.cdc:
        files += [settings.cdc.tsv, *settings.cdc.xlsx, *settings.cdc.season]
        files += settings.cdc.sequence_ids
    if settings.locations:
        files += [settings.locations.locdb, settings.locations.chinese_aliases]
    folders: list[AC21Inputs | VIDRLInputs] = [
        *settings.ac21,
        *settings.niid,
        *settings.vidrl,
        *settings.crick,
    ]
    for inputs in folders:
        if inputs.names:
            rules = rules or Rules(settings.rules)
            start = dt.date.fromisoformat(inputs.start).isoformat()
            files += [
                original
                for rule, original in _source_rows(inputs, rules)
                if (day := _file_date(Path(rule["read_as"]))) is not None
                and day >= start
                and rule["read_as"] not in inputs.exclude
            ]
            continue
        files += _dated_files(inputs)[0]  # an undated name is no input: the read reports it
    files += [s.file for s in settings.crick_sheets]
    return files


def _source_rows(inputs: AC21Inputs | VIDRLInputs, rules: Rules) -> list[tuple[Rule, Path]]:
    """The source_names rows of a ``names`` entry, each with its original under the entry's
    root. A set with no rows, or a row whose original is missing, is fatal (design rule 4)."""
    table = rules.source_names
    rows = [
        (rule, inputs.dir / rule["file"])
        for rule in table.rules
        if table.in_scope(rule, lab=inputs.lab) and rule["set"] == inputs.names
    ]
    if not rows:
        raise ValueError(f"{inputs.dir}: no source_names rows for {inputs.lab} {inputs.names}")
    if missing := [f"{rule.where}: {path}" for rule, path in rows if not path.is_file()]:
        raise FileNotFoundError("originals not found: " + "; ".join(missing))
    return rows


def _stage_sources(
    settings: TablesSettings, rules: Rules, sources: Path
) -> tuple[TablesSettings, list[str]]:
    """Link each ``names`` entry's originals under their read names in a folder of the run's
    work area, and point the entry at that folder: the readers then see the names they key
    rules and dates on (af.tables.vidrl reads the test date from the file name), and the
    originals stay where they are. The folder is rebuilt every run, so it never holds a link
    the rules no longer name. Also reports originals under a root that no row names, so that
    a new workbook dropped into the root is seen, not silently left out."""
    report: list[str] = []

    def stage(inputs: Any) -> Any:
        if not inputs.names:
            return inputs
        folder = sources / inputs.names
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        rows = _source_rows(inputs, rules)
        for rule, original in rows:
            (folder / rule["read_as"]).symlink_to(original.resolve())
            rule.hits += 1
        report.append(
            f"{inputs.lab} {inputs.names}: {len(rows)} originals in {inputs.dir},"
            " read under their source_names names"
        )
        return replace(inputs, dir=folder)

    seen: dict[tuple[str, str], str] = {}

    def once(inputs: Any) -> None:
        """Two sets of a lab staging one read name would read as two tests of one day."""
        if not inputs.names:
            return
        for rule, _ in _source_rows(inputs, rules):
            key = (inputs.lab.casefold(), rule["read_as"])
            if key in seen and seen[key] != inputs.names:
                raise ValueError(
                    f"{rule.where}: {rule['read_as']} is read from sets {seen[key]} and"
                    f" {inputs.names}"
                )
            seen[key] = inputs.names

    for inputs in [*settings.ac21, *settings.niid, *settings.vidrl, *settings.crick]:
        once(inputs)
    staged = replace(
        settings,
        ac21=[stage(i) for i in settings.ac21],
        niid=[stage(i) for i in settings.niid],
        vidrl=[stage(i) for i in settings.vidrl],
        crick=[stage(i) for i in settings.crick],
    )
    report += _unnamed_originals(settings, rules)
    return staged, report


def _unnamed_originals(settings: TablesSettings, rules: Rules) -> list[str]:
    """Workbooks under a ``names`` root that no source_names row of that lab names: counted,
    and listed when the file name dates them on or after the earliest start on that root.
    A workbook named as one of the lab's read names is accounted for: a curated copy whose
    test is now read from its original (a root of copies holds many)."""
    roots: dict[tuple[str, Path], list[str]] = {}
    folders: list[AC21Inputs | VIDRLInputs] = [
        *settings.ac21,
        *settings.niid,
        *settings.vidrl,
        *settings.crick,
    ]
    for inputs in folders:
        if inputs.names:
            roots.setdefault((inputs.lab, inputs.dir), []).append(inputs.start)
    lines = []
    for (lab, root), starts in sorted(roots.items()):
        table = rules.source_names
        rows = [r for r in table.rules if table.in_scope(r, lab=lab)]
        named = {(root / r["file"]).resolve() for r in rows}
        read_as = {r["read_as"] for r in rows}
        start = min(dt.date.fromisoformat(s) for s in starts).isoformat()
        unnamed = [
            p
            for p in sorted(root.rglob("*"))
            if p.suffix.lower() in WORKBOOKS
            and not p.name.startswith("~$")
            and p.resolve() not in named
            and p.name not in read_as
        ]
        recent = [p for p in unnamed if (_safe_file_date(p) or "") >= start]
        lines.append(
            f"{lab} workbooks in {root} not named by source_names: {len(unnamed)},"
            f" {len(recent)} dated on or after {start}"
        )
        lines += [f"  {p.relative_to(root)}" for p in recent]
    return lines


def _safe_file_date(path: Path) -> str | None:
    """A file-name date, or None when the eight digits are not a date (an original's name is
    the lab's, not ours)."""
    try:
        return _file_date(path)
    except ValueError:
        return None


def _check_round_trip(store: Store, tables: list[Table]) -> None:
    """Design rule 3: every table must read back from the store with its own hash."""
    stored = current_tables(store)
    for table in tables:
        if table.table_id not in stored:
            raise RuntimeError(f"{table.table_id} is not in the store after publishing")
        _, path = stored[table.table_id]
        back = Table.from_json(json.loads(path.read_text(encoding="utf-8")))
        if back.content_hash() != table.content_hash():
            raise RuntimeError(f"{path}: stored table does not reproduce its hash")


def _external(path: Path) -> ExternalInput:
    """An input outside the store, with the git commit of the repo it lives in, if any."""
    directory = path if path.is_dir() else path.parent
    git = subprocess.run(
        ["git", "-C", str(directory), "log", "-1", "--format=%H", "--", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = git.stdout.strip() if git.returncode == 0 and git.stdout.strip() else None
    return ExternalInput.of(path, version=commit)


def _content(table: Table) -> tuple[Any, ...]:
    """Everything a test measured: antigens and sera as identified, and every reading."""
    return (
        table.group,
        table.date,
        [
            (a.name, a.reassortant, a.annotations, a.passage, a.passage_date, a.lab_ids)
            for a in table.antigens
        ],
        [(s.name, s.reassortant, s.annotations, s.serum_id) for s in table.sera],
        table.titres,
    )


def exact_repeats(tables: list[Table]) -> tuple[list[Table], list[str]]:
    """A lab's export can hold one test twice under two test ids (CDC's TSV, 2022-11-21: two
    test ids, the same antigens, sera and every reading). Read twice it would be two tables
    and count every reading twice. The later one, in input order, is dropped and reported."""
    first: dict[str, Table] = {}
    kept, report = [], []
    for table in tables:
        key = repr(_content(table))
        if (earlier := first.get(key)) is not None:
            report.append(f"{table.source_key} repeats {earlier.source_key}")
            continue
        first[key] = table
        kept.append(table)
    return kept, report


def duplicates(tsv: list[Table], xlsx: list[Table]) -> list[str]:
    """A workbook test that CDC has since added to the TSV must not be read twice: same group
    and date, and most of its antigens (by CDC id) and sera (by lot) already in a TSV test."""
    out = []
    for x in xlsx:
        ag = {i for a in x.antigens for i in a.lab_ids}
        sr = {s.serum_id for s in x.sera}
        for t in tsv:
            if (t.group, t.date) != (x.group, x.date):
                continue
            ag_t = {i for a in t.antigens for i in a.lab_ids}
            sr_t = {s.serum_id for s in t.sera}
            if len(ag & ag_t) > len(ag) / 2 and len(sr & sr_t) > len(sr) / 2:
                out.append(
                    f"{x.source_key} looks like {t.source_key} "
                    f"({len(ag & ag_t)}/{len(ag)} antigens, "
                    f"{len(sr & sr_t)}/{len(sr)} sera shared): drop the workbook from the inputs"
                )
    return out


def make_step(parameters: Mapping[str, Any], *, base_dir: Path, paths: PathsConfig) -> Step:
    """The pipeline step. ``parameters`` is a ``[parameters.tables-update]`` table with the
    same keys as ``[tables]``; relative paths are resolved against ``base_dir`` (the
    directory of the pipeline config) and the roots come from the config's ``[paths]``.
    Inputs are named by role, so the step's record does not depend on where files live."""
    settings = parse_config(dict(parameters), TablesSettings, base_dir=base_dir)
    inputs = {"rules": settings.rules}
    for path in _input_files(settings):  # loads the rules only for a names entry
        inputs[f"input:{path.parent.name}/{path.name}"] = path
    state = state_dir(paths, settings)

    def action(_: StepContext) -> None:
        status, report = update(settings, paths)
        (state / REPORT).write_text("\n".join(report) + "\n", encoding="utf-8")
        if status:
            raise RuntimeError(f"{STEP} failed; see {state / REPORT}")

    return Step(
        name=STEP,
        action=action,
        outputs=[Artefact(state / PUBLISHED, parse=lambda p: json.loads(p.read_text()))],
        inputs=inputs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="read and report; write nothing"
    )
    args = parser.parse_args(argv)
    config = load_config(args.config, TablesConfig)
    status, report = update(config.tables, config.paths, dry_run=args.dry_run)
    text = "\n".join(report) + "\n"
    print(text, end="")
    if not args.dry_run:
        (state_dir(config.paths, config.tables) / REPORT).write_text(text, encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())
