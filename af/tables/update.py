"""Re-read every raw table input into the tables store and report what changed.

    python -m af.tables.update tables.toml [--dry-run]

with a config such as (paths relative to the config file)::

    [tables]
    rules = "../acmacs-f-data/rules/tables"
    store = "../store"                       # the af.store root (tables/ is inside it)
    report = "../store-state/tables-report.txt"
    published = "../store-state/tables-published.json"

    [tables.cdc]
    tsv = "../fludata/CDC-Atlanta-WHO-CC/raw-data/CDC_titers_sept_2019_onwards.tsv"
    xlsx = ["../whocc-tables/h3-hi-guinea-pig-cdc/xlsx/<one workbook>.xlsx"]  # tests not in the TSV

Every run reads every input (a full re-read), so a rule that matched nothing is an error.
Nothing is published if anything went wrong: format errors, broken tables, unmatched
rules, a workbook duplicating a TSV test. Exit status 1 then.

The run writes two files: ``report`` (counts, rule usage, the diff with its restart dates,
errors) and ``published`` (every dataset ref published or reconfirmed). ``published`` is
the declared output of the pipeline step (:func:`make_step`), so any change to any group
re-runs whatever depends on the tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from af.pipeline.driver import Step, StepContext
from af.store import ExternalInput, Provenance, Store
from af.util.artefacts import Artefact
from af.util.config import load_config, parse_config

from . import cdc, identity
from .model import Table
from .rules import Rules
from .store import current_tables, previous_index, publish

STEP = "tables-update"


@dataclass(frozen=True)
class CDCInputs:
    tsv: Path
    xlsx: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class TablesSettings:
    rules: Path
    store: Path
    report: Path
    published: Path
    cdc: CDCInputs


@dataclass(frozen=True)
class TablesConfig:
    tables: TablesSettings


def update(settings: TablesSettings, *, dry_run: bool = False) -> tuple[int, list[str]]:
    started = dt.datetime.now(dt.UTC)
    rules = Rules(settings.rules)
    store = Store.open(settings.store)
    previous = previous_index(store)
    tables, report, errors = _read_all(settings, rules)
    errors.extend(identity.assign(tables, previous))
    report.extend(rules.usage_report())
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
        inputs=tuple(_external(p) for p in (settings.rules, settings.cdc.tsv, *settings.cdc.xlsx)),
        parameters={},
        started=started,
        finished=dt.datetime.now(dt.UTC),
    )
    results = publish(store, tables, manifest, provenance)
    report.extend(f"{event:11s} {ref}" for ref, event in results)
    settings.published.parent.mkdir(parents=True, exist_ok=True)
    published = [{"event": event, "ref": ref.to_json()} for ref, event in results]
    settings.published.write_text(json.dumps(published, indent=1) + "\n", encoding="utf-8")
    _check_round_trip(store, tables)
    return 0, report


def _read_all(settings: TablesSettings, rules: Rules) -> tuple[list[Table], list[str], list[str]]:
    result = cdc.read(settings.cdc.tsv, rules)
    report = [f"read {settings.cdc.tsv}: {result.rows} rows -> {len(result.tables)} tables"]
    report.append("dropped: " + ", ".join(f"{k} {v}" for k, v in sorted(result.dropped.items())))
    report.extend(f"skipped: {s}" for s in result.skipped_tests)
    errors = list(result.errors)
    tables = list(result.tables)
    if settings.cdc.xlsx:
        from . import cdc_xlsx  # needs openpyxl; imported only when there are workbooks

        xl = cdc_xlsx.read(list(settings.cdc.xlsx), rules)
        report.append(f"read {len(settings.cdc.xlsx)} CDC workbooks -> {len(xl.tables)} tables")
        report.extend(f"skipped: {s}" for s in xl.skipped_tests)
        errors.extend(xl.errors)
        errors.extend(duplicates(result.tables, xl.tables))
        tables.extend(xl.tables)
    return tables, report, errors


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


def make_step(parameters: Mapping[str, Any], *, base_dir: Path) -> Step:
    """The pipeline step. ``parameters`` is a ``[parameters.tables-update]`` table with the
    same keys as ``[tables]``; relative paths are resolved against ``base_dir`` (the
    directory of the pipeline config). Inputs are named by role, so the step's record
    does not depend on where the files live."""
    settings = parse_config(dict(parameters), TablesSettings, base_dir=base_dir)
    inputs = {"rules": settings.rules, "cdc_tsv": settings.cdc.tsv}
    inputs.update({f"cdc_xlsx:{p.name}": p for p in settings.cdc.xlsx})

    def action(_: StepContext) -> None:
        status, report = update(settings)
        settings.report.parent.mkdir(parents=True, exist_ok=True)
        settings.report.write_text("\n".join(report) + "\n", encoding="utf-8")
        if status:
            raise RuntimeError(f"{STEP} failed; see {settings.report}")

    return Step(
        name=STEP,
        action=action,
        outputs=[Artefact(settings.published, parse=lambda p: json.loads(p.read_text()))],
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
    settings = load_config(args.config, TablesConfig).tables
    status, report = update(settings, dry_run=args.dry_run)
    text = "\n".join(report) + "\n"
    print(text, end="")
    if not args.dry_run:
        settings.report.parent.mkdir(parents=True, exist_ok=True)
        settings.report.write_text(text, encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())
