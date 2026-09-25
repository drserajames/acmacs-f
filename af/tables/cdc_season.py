"""Read CDC's older per-season titre exports (fludata ``HITest_<season>_titers.tsv``).

Used only where the main TSV does not reach: Sarah (25 Sep 2026) chose this file for CDC's
Aug 2019 H1 tests, just before the main TSV starts. What is read is set by
``season_files.tsv`` rules (file, subtype, date range); rows outside every rule are skipped
and counted, and a rule that selects nothing is an error.

These files differ from the main TSV, and the differences are stated, not papered over:

- **No test_id.** A table is one (assay_date, subtype, assay-type); the rule row's
  ``table_key`` column names this. Two tests on one day with the same subtype and assay
  cannot be told apart and become one table.
- **No not-for-use flags.** Every row is read as reportable; each table's ``meta`` says
  ``"not_for_use_flags": "absent in source"`` and the run reports the count.
- **No harvest dates**, so ``passage_date`` is empty. Antigens therefore do not match the
  same virus in main-TSV tables (whose passages carry a harvest date) under ae's identity.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
from collections import Counter
from pathlib import Path

from . import aliases
from .cdc import LAB, SUBTYPES, CDCFormatError, ReadResult, _titre_order
from .model import Antigen, Serum, Table
from .passage import PassageParser
from .rules import Rule, Rules

COLUMNS = [
    "virus_cdc_id",
    "virus_strain",
    "virus_collection_date",
    "virus_strain_passage",
    "serum_strain",
    "ferret_id",
    "lot #",
    "serum_antigen_passage",
    "assay_date",
    "subtype",
    "titer",
    "assay-type",
    "reported_by_fra",
    "tested_by_fra",
]
ASSAYS = {"HI": "HI", "FRA": "FRA"}
KEY = "assay_date+subtype+assay-type"  # the only table_key this reader implements


def read(path: Path, rules: Rules) -> ReadResult:
    data = path.read_bytes()
    lines, rejoined = _records(data.decode("utf-8").splitlines(), path)
    reader = csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)
    header = reader.fieldnames or []
    if sorted(header) != sorted(COLUMNS):
        raise CDCFormatError(f"{path}: columns {header} are not the season-file columns")
    selected = [r for r in rules.season_files.rules if r["file"] == path.name]
    if not selected:
        raise CDCFormatError(f"{path.name}: no season_files rule names this file")
    for rule in selected:
        if rule["table_key"] != KEY:
            raise CDCFormatError(f"{rule.where}: table_key {rule['table_key']!r}; only {KEY!r}")
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    result = ReadResult(tables=[], rows=0)
    if rejoined:
        result.dropped["records rejoined (a line break inside a field)"] = rejoined
    for no, row in enumerate(reader, start=2):
        row = {k: (v or "").strip() for k, v in row.items()}
        row["_line"] = str(no)
        matched = _rule_for(row, selected)
        if matched is None:
            result.dropped["rows: outside season_files scope"] += 1
            continue
        matched.hits += 1
        result.rows += 1
        groups.setdefault((row["assay_date"], row["subtype"], row["assay-type"]), []).append(row)
    result.dropped["rows: read with no not-for-use flags (absent in source)"] = result.rows
    provenance = {
        "file": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "reader": "af.tables.cdc_season",
    }
    for key in sorted(groups):
        try:
            table = _make_table(path.name, key, groups[key], rules)
        except CDCFormatError as err:
            result.errors.append(f"{path.name} {key}: {err}")
            continue
        table.provenance = dict(provenance)
        if problems := table.check() or aliases.check_titres(table, rules):
            result.errors.extend(f"{path.name} {key}: {p}" for p in problems)
        result.tables.append(table)
    return result


def _records(lines: list[str], path: Path) -> tuple[list[str], int]:
    """Physical lines -> records. Some records in these exports have a line break inside a
    strain name; such a record arrives as two lines whose tab-separated fields add up to the
    header's count. They are rejoined (without the break) and counted; anything else that
    does not have the header's field count is an error."""
    width = len(lines[0].split("\t"))
    out, rejoined, pending = [lines[0]], 0, ""
    for no, line in enumerate(lines[1:], start=2):
        record = pending + line
        fields = len(record.split("\t"))
        if fields < width:
            pending = record
            continue
        if fields > width:
            raise CDCFormatError(f"{path}:{no}: {fields} fields, expected {width}")
        rejoined += bool(pending)
        pending = ""
        out.append(record)
    if pending:
        raise CDCFormatError(f"{path}: the file ends inside a record")
    return out, rejoined


def _rule_for(row: dict[str, str], rules: list[Rule]) -> Rule | None:
    try:
        day = dt.date.fromisoformat(row["assay_date"])
    except ValueError:
        raise CDCFormatError(f"line {row['_line']}: assay_date {row['assay_date']!r}") from None
    for rule in rules:
        low, high = dt.date.fromisoformat(rule["date_from"]), dt.date.fromisoformat(rule["date_to"])
        if row["subtype"] == rule["subtype"] and low <= day <= high:
            return rule
    return None


def _make_table(
    file: str, key: tuple[str, str, str], rows: list[dict[str, str]], rules: Rules
) -> Table:
    date, test_subtype, assay_raw = key
    if test_subtype not in SUBTYPES or assay_raw not in ASSAYS:
        raise CDCFormatError(f"unknown subtype/assay {test_subtype!r}/{assay_raw!r}")
    subtype, lineage, prefix = SUBTYPES[test_subtype]
    assay = ASSAYS[assay_raw]
    default = rules.table_defaults.lookup(lab=LAB, subtype=subtype, assay=assay)
    if default is None:
        raise CDCFormatError(f"no table_defaults rule for {LAB} {subtype} {assay}")
    rbc = "" if default["rbc"] == "-" else default["rbc"]
    group = "-".join(p for p in (prefix, assay.lower(), rbc, LAB.lower()) if p)
    passages = PassageParser(rules.passage_tokens, LAB)
    warnings: list[str] = []
    dropped: Counter[str] = Counter()
    antigens: dict[tuple[str, ...], Antigen] = {}
    sera: dict[tuple[str, ...], Serum] = {}
    cells: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    for row in rows:
        lot = row["lot #"].replace(";", ",").replace(", ", ",")
        rule = rules.control_sera.find(row["lot #"], lab=LAB)
        if rule is not None and rule["action"] == "drop":
            dropped["rows: control serum"] += 1
            continue
        ag_key = (row["virus_strain"], row["virus_strain_passage"], row["virus_cdc_id"])
        sr_key = (row["serum_strain"], lot)
        if ag_key not in antigens:
            antigens[ag_key] = _antigen(row, subtype, lineage, rules, passages, warnings)
        if sr_key not in sera:
            sera[sr_key] = _serum(row, subtype, lineage, lot, rules, passages, warnings)
            if rule is not None and rule["action"] == "species":
                sera[sr_key].species = rule["value"]
        titre = _titre(row, assay, rules)
        if titre is not None:
            cells.setdefault((ag_key, sr_key), []).append(titre)
    ag_keys = [k for k in antigens if any((k, s) in cells for s in sera)]
    sr_keys = [s for s in sera if any((a, s) in cells for a in antigens)]
    return Table(
        table_id="",
        group=group,
        lab=LAB,
        subtype=subtype,
        lineage=lineage,
        assay=assay,
        rbc=rbc,
        date=dt.date.fromisoformat(date).isoformat(),
        date_suffix=0,
        source_key=f"CDC season file {file} {date} {test_subtype} {assay_raw}",
        antigens=[antigens[k] for k in ag_keys],
        sera=[sera[k] for k in sr_keys],
        titres=[
            [sorted(cells.get((a, s), []), key=_titre_order) for s in sr_keys] for a in ag_keys
        ],
        meta={"file": file, "not_for_use_flags": "absent in source", "test_id": "absent in source"},
        dropped=dict(sorted((k, v) for k, v in dropped.items() if v)),
        warnings=sorted(set(warnings)),
    )


def _antigen(
    row: dict[str, str],
    subtype: str,
    lineage: str,
    rules: Rules,
    passages: PassageParser,
    warnings: list[str],
) -> Antigen:
    name, renamed = aliases.parse_name(
        rules,
        row["virus_strain"],
        lab=LAB,
        subtype=subtype,
        applies_to="antigen",
        warnings=warnings,
    )
    passage = passages.parse(row["virus_strain_passage"])
    warnings.extend(passage.problems)
    antigen = Antigen(
        name=name.name,
        raw_name=row["virus_strain"],
        passage=passage.text,
        date=dt.date.fromisoformat(row["virus_collection_date"]).isoformat()
        if row["virus_collection_date"]
        else None,
        lab_ids=[f"CDC#{row['virus_cdc_id']}"],
        reassortant=name.reassortant,
        annotations=name.annotations,
        lineage=lineage,
        source={"virus_strain_passage": row["virus_strain_passage"]},
    )
    if renamed:
        antigen.source[aliases.SOURCE_KEY] = renamed
    return antigen


def _serum(
    row: dict[str, str],
    subtype: str,
    lineage: str,
    lot: str,
    rules: Rules,
    passages: PassageParser,
    warnings: list[str],
) -> Serum:
    name, renamed = aliases.parse_name(
        rules, row["serum_strain"], lab=LAB, subtype=subtype, applies_to="serum", warnings=warnings
    )
    passage = passages.parse(row["serum_antigen_passage"])
    warnings.extend(passage.problems)
    serum = Serum(
        name=name.name,
        raw_name=row["serum_strain"],
        serum_id=f"CDC {lot}",
        passage=passage.text,
        reassortant=name.reassortant,
        annotations=name.annotations,
        lineage=lineage,
        source={"ferret_id": row["ferret_id"], "lot": row["lot #"]},
    )
    if renamed:
        serum.source[aliases.SOURCE_KEY] = renamed
    return serum


def _titre(row: dict[str, str], assay: str, rules: Rules) -> str | None:
    raw = row["titer"]
    if (rule := rules.titre_tokens.find(raw, lab=LAB, assay=assay)) is not None:
        return None if rule["titre"] == "*" else rule["titre"]
    if raw.isdigit() and int(raw) >= 10:
        return raw
    raise CDCFormatError(f"line {row['_line']}: titre {raw!r} matches no titre_tokens rule")
