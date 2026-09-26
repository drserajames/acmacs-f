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
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.pipeline.driver import Step, StepContext
from af.store import ExternalInput, PathsConfig, Provenance, Store, Work
from af.util.artefacts import Artefact
from af.util.config import load_config, parse_config

from . import cdc, identity, ids
from .model import Table
from .rules import Rules
from .store import KIND, current_tables, previous_index, publish

STEP = "tables-update"


@dataclass(frozen=True)
class CDCInputs:
    tsv: Path
    xlsx: list[Path] = field(default_factory=list)
    season: list[Path] = field(default_factory=list)  # older per-season files, scoped by rules


@dataclass(frozen=True)
class LocationSources:
    """Where Chinese location names are romanised from (see af.tables.locations)."""

    locdb: Path  # acmacs-data locationdb.json.xz
    chinese_aliases: Path  # e.g. whocc-tables cnic-location-aliases.tsv


@dataclass(frozen=True)
class AC21Inputs:
    """One folder of a lab's dated workbooks (AC Excel 2.1, or NIID's layout). ``start``
    bounds what is read: a workbook is read when the YYYYMMDD in its file name is on or after
    it, and its test date must then equal that file-name date (checked), so the bound is on
    the test date. ``exclude`` names workbooks in the folder not to read at all (e.g. one
    whose file name has no usable date); each must exist, so a stale entry is an error."""

    lab: str
    dir: Path
    start: str  # ISO date
    exclude: list[str] = field(default_factory=list)  # file names


@dataclass(frozen=True)
class VIDRLInputs:
    """A folder of VIDRL workbooks: as AC21Inputs, plus the flu type, which VIDRL's sheets do
    not state (``subtype`` "A(H1N1)", "A(H3N2)" or "B"; ``lineage`` for B)."""

    lab: str
    dir: Path
    start: str  # ISO date
    subtype: str
    lineage: str = ""
    exclude: list[str] = field(default_factory=list)  # file names


@dataclass(frozen=True)
class TablesSettings:
    rules: Path
    run: str  # work-area key for this run's state, e.g. "cdc/all" (one run publishes many groups)
    cdc: CDCInputs | None = None
    ac21: list[AC21Inputs] = field(default_factory=list)
    niid: list[AC21Inputs] = field(default_factory=list)  # NIID's own layout (af.tables.niid)
    vidrl: list[VIDRLInputs] = field(default_factory=list)  # VIDRL's layout (af.tables.vidrl)
    locations: LocationSources | None = None


@dataclass(frozen=True)
class TablesConfig:
    paths: PathsConfig
    tables: TablesSettings


REPORT = "tables-report.txt"
PUBLISHED = "tables-published.json"


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
    tables, report, errors = _read_all(settings, rules)
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
        inputs=tuple(_external(p) for p in (settings.rules, *_input_files(settings))),
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
            files = _dated_files(inputs)
            result = ac21.read(files, rules, locations, lab=inputs.lab)
            _add_workbooks(inputs, files, result, tables, report, errors)
    for inputs in settings.niid:
        from . import niid

        files = _dated_files(inputs)
        _add_workbooks(
            inputs, files, niid.read(files, rules, lab=inputs.lab), tables, report, errors
        )
    for vinputs in settings.vidrl:
        from . import vidrl

        files = _dated_files(vinputs)
        result = vidrl.read(
            files, rules, lab=vinputs.lab, subtype=vinputs.subtype, lineage=vinputs.lineage
        )
        _add_workbooks(vinputs, files, result, tables, report, errors)
    return tables, report, errors


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


def _file_date(path: Path) -> str | None:
    m = re.search(r"(\d{4})(\d{2})(\d{2})", path.stem)
    return dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat() if m else None


def _dated_files(inputs: AC21Inputs | VIDRLInputs) -> list[Path]:
    """Workbooks in the folder dated on or after ``start``; Excel lock files (~$) are not
    workbooks. A workbook whose name carries no date is an error, not silently skipped."""
    if not inputs.dir.is_dir():
        raise FileNotFoundError(f"workbook folder missing: {inputs.dir}")
    start = dt.date.fromisoformat(inputs.start).isoformat()
    if missing := [n for n in inputs.exclude if not (inputs.dir / n).is_file()]:
        raise FileNotFoundError(f"{inputs.dir}: excluded workbooks not found: {missing}")
    out = []
    for path in sorted(inputs.dir.glob("*.xlsx")):
        if path.name.startswith("~$") or path.name in inputs.exclude:
            continue
        day = _file_date(path)
        if day is None:
            raise ValueError(f"{path}: no YYYYMMDD date in the file name")
        if dt.date.fromisoformat(day) >= dt.date.fromisoformat(start):
            out.append(path)
    return out


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
        tables.extend(season.tables)
    return tables, report, errors


def _input_files(settings: TablesSettings) -> list[Path]:
    files: list[Path] = []
    if settings.cdc:
        files += [settings.cdc.tsv, *settings.cdc.xlsx, *settings.cdc.season]
    if settings.locations:
        files += [settings.locations.locdb, settings.locations.chinese_aliases]
    folders: list[AC21Inputs | VIDRLInputs] = [*settings.ac21, *settings.niid, *settings.vidrl]
    for inputs in folders:
        files += _dated_files(inputs)
    return files


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
    for path in _input_files(settings):
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
