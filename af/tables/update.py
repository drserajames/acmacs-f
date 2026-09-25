"""Re-read the raw inputs into the tables store and report what changed.

    python -m af.tables.update tables.toml [--dry-run]

with a config such as (paths relative to the config file)::

    [tables]
    rules = "../acmacs-f-data/rules/tables"
    store = "../store/tables"

    [tables.cdc]
    tsv = "../fludata/CDC-Atlanta-WHO-CC/raw-data/CDC_titers_sept_2019_onwards.tsv"
    xlsx = ["../whocc-tables/h3-hi-guinea-pig-cdc/xlsx/<one workbook>.xlsx"]  # tests not in the TSV

Every run reads every input (a full re-read), so a rule that matched nothing is an error.
Exit status 1 if anything went wrong (format errors, broken tables, unmatched rules, a
workbook duplicating a TSV test); the store is then left as it was.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from af.util.config import load_config

from . import cdc, identity
from .model import Table
from .rules import Rules
from .store import Store, file_input


@dataclass(frozen=True)
class CDCInputs:
    tsv: Path
    xlsx: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class TablesSettings:
    rules: Path
    store: Path
    cdc: CDCInputs


@dataclass(frozen=True)
class TablesConfig:
    tables: TablesSettings


def update(
    cdc_tsv: Path,
    rules_dir: Path,
    store_root: Path,
    *,
    cdc_xlsx: list[Path] | None = None,
    dry_run: bool = False,
) -> tuple[int, list[str]]:
    rules = Rules(rules_dir)
    store = Store(store_root)
    previous = store.previous_manifest()
    result = cdc.read(cdc_tsv, rules)
    report = [f"read {cdc_tsv}: {result.rows} rows -> {len(result.tables)} tables"]
    report.append("dropped: " + ", ".join(f"{k} {v}" for k, v in sorted(result.dropped.items())))
    report.extend(f"skipped: {s}" for s in result.skipped_tests)
    errors = list(result.errors)
    inputs = [file_input(cdc_tsv)]
    if cdc_xlsx:
        from . import cdc_xlsx as reader  # needs openpyxl; only imported when there are workbooks

        xl = reader.read(list(cdc_xlsx), rules)
        report.append(f"read {len(cdc_xlsx)} CDC workbooks -> {len(xl.tables)} tables")
        report.extend(f"skipped: {s}" for s in xl.skipped_tests)
        errors.extend(xl.errors)
        errors.extend(duplicates(result.tables, xl.tables))
        result.tables.extend(xl.tables)
        inputs.extend(file_input(p) for p in cdc_xlsx)
    problems = identity.assign(result.tables, previous)
    errors.extend(problems)
    unmatched = [r for t in rules.tables() for r in t.unmatched()]
    report.extend(rules.usage_report())
    errors.extend(f"rule never matched: {r.where}" for r in unmatched)
    suffixed = [t.table_id for t in result.tables if t.date_suffix > 1]
    report.append(f"tables sharing a date (suffix .2+): {len(suffixed)} {suffixed}")
    manifest = identity.Manifest.from_tables(result.tables, inputs, previous)
    if previous is not None:
        report.extend(identity.diff(previous, manifest).report())
    else:
        report.append(f"first run: {len(manifest.tables)} tables")
    report.extend(f"ERROR {e}" for e in errors)
    if errors or dry_run:
        return (1 if errors else 0), report
    for table in result.tables:
        store.write_table(table)
    run = store.commit(manifest, report)
    report.append(f"store updated: run {run}")
    return 0, report


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
    status, report = update(
        settings.cdc.tsv,
        settings.rules,
        settings.store,
        cdc_xlsx=settings.cdc.xlsx,
        dry_run=args.dry_run,
    )
    print("\n".join(report))
    return status


if __name__ == "__main__":
    sys.exit(main())
