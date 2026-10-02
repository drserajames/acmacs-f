"""Tables read from ae's committed .ace when no workbook exists.

Some tests survive only as ae's chart: the lab's workbook was never sent or was lost, and ae
built the .ace from a hand-made torg. A decision to use such a chart anyway is data, not
config: a row of ``ace_imports.tsv`` names the file and gives the reason, and every run
reports it ("ae .ace, no workbook"), so the table is never mistaken for one af read itself.

ae's reading is taken as it is, through af's own name and passage parsers so the points
match af's tables, with two kinds of correction:

- known misreadings of ae's are fixed by ``cell_fixes`` rows, addressed to two pseudo-sheets
  built from the chart: ``antigens`` (one row per antigen after a header row; columns A name,
  B reassortant, C annotations, D passage, E date, F lab id) and ``sera`` (A name,
  B reassortant, C annotations, D serum id, E passage, F species);
- a lab id's spaces around its hyphen are removed ("X#18/19 - 15" -> "X#18/19-15"), the
  form af's readers write.

When the lab's workbook turns up, the import must be retired (af.tables.update checks that
no workbook table has the same group and date).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from af.chart.ace import decompress
from af.util.subtypes import subtypes

from . import aliases
from .cdc import ReadResult
from .model import Antigen, Serum, Table
from .passage import PassageParser
from .rules import Rule, Rules
from .sheet import Sheet, apply_cell_fixes

SOURCE = "ae .ace, no workbook"
ANTIGEN_COLUMNS = ("N", "R", "a", "P", "D", "l")
SERUM_COLUMNS = ("N", "R", "a", "I", "P", "s")
DATED_PASSAGE = re.compile(r"(.*?)\s*\((\d{4}-\d{2}-\d{2})\)")


def read(path: Path, rule: Rule, rules: Rules, *, lab: str) -> ReadResult:
    """One .ace named by an ace_imports row: one table."""
    data = path.read_bytes()

    chart: dict[str, Any] = json.loads(decompress(data))["c"]
    info = chart["i"]
    subtype = {"B": "B"}.get(info["V"], info["V"])
    lineage = info.get("s", "")
    if info.get("l", lab) != lab:
        raise ValueError(f"{path}: the chart's lab is {info.get('l')!r}, the rule says {lab!r}")
    test_date = dt.date(int(info["D"][:4]), int(info["D"][4:6]), int(info["D"][6:8]))
    antigens_sheet = _sheet(path, "antigens", chart["a"], ANTIGEN_COLUMNS)
    sera_sheet = _sheet(path, "sera", chart["s"], SERUM_COLUMNS)
    warnings = [f"{path.name}: {SOURCE}, read by {rule.where}"]
    warnings += apply_cell_fixes(antigens_sheet, rules.cell_fixes, lab)
    warnings += apply_cell_fixes(sera_sheet, rules.cell_fixes, lab)
    passages = PassageParser(rules.passage_tokens, lab)
    antigens = [
        _antigen(row, chart["a"][i], subtype, lineage, lab, passages, rules, test_date, warnings)
        for i, row in enumerate(antigens_sheet.rows[1:])
    ]
    sera = [
        _serum(row, subtype, lineage, lab, passages, rules, test_date, warnings)
        for row in sera_sheet.rows[1:]
    ]
    titres = _titres(chart["t"], len(antigens), len(sera))
    dropped: Counter[str] = Counter()
    keep = [i for i, row in enumerate(titres) if any(row)]  # as the workbook readers do
    dropped["antigens: no readings"] = len(antigens) - len(keep)
    antigens, titres = [antigens[i] for i in keep], [titres[i] for i in keep]
    used = [j for j in range(len(sera)) if any(row[j] for row in titres)]
    dropped["sera: no readings"] = len(sera) - len(used)
    sera, titres = [sera[j] for j in used], [[row[j] for j in used] for row in titres]
    group = "-".join(
        (subtypes().group_prefix(subtype, lineage), info["A"].lower(), info["r"], lab.lower())
    )
    table = Table(
        table_id="",
        group=group,
        lab=lab,
        subtype=subtype,
        lineage=lineage,
        assay=info["A"],
        rbc=info["r"],
        date=test_date.isoformat(),
        date_suffix=0,
        source_key=f"{lab} ace {path.name}",
        antigens=antigens,
        sera=sera,
        titres=titres,
        meta={"file": path.name, "source": SOURCE, "rule": rule.where},
        dropped={k: v for k, v in sorted(dropped.items()) if v},
        warnings=warnings,
    )
    table.provenance = {
        "file": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "reader": "af.tables.ace_import",
    }
    result = ReadResult(tables=[table], rows=len(antigens), dropped=Counter())
    if problems := table.check() or aliases.check_titres(table, rules):
        result.errors.extend(f"{path.name}: {p}" for p in problems)
    return result


def _sheet(path: Path, name: str, items: list[dict[str, Any]], keys: tuple[str, ...]) -> Sheet:
    rows = [list(keys)]
    for item in items:
        row = []
        for key in keys:
            value = item.get(key, "")
            if key == "a":
                value = " ".join(v for v in value if v != "DISTINCT")
            elif key == "l":
                value = value[0] if value else ""
            row.append(str(value))
        rows.append(row)
    return Sheet(path=path, name=name, rows=rows)


def _passage(text: str, passages: PassageParser, warnings: list[str]) -> tuple[str, str | None]:
    m = DATED_PASSAGE.fullmatch(text)
    raw, date = (m[1], m[2]) if m else (text, None)
    parsed = passages.parse(re.sub(r"(?<=[A-Za-z])\?", "X", raw))  # ae writes an unknown count "?"
    warnings.extend(parsed.problems)
    return parsed.text, date


def _name(raw: str, subtype: str, lab: str, rules: Rules, applies_to: str, year: int, w: list[str]):
    return aliases.parse_name(
        rules, raw, lab=lab, subtype=subtype, applies_to=applies_to, warnings=w, not_after=year
    )


def _antigen(row, item, subtype, lineage, lab, passages, rules, test_date, warnings) -> Antigen:  # noqa: ANN001
    name_raw, reassortant, annotations, passage_raw, date, lab_id = row
    name, renamed = _name(name_raw, subtype, lab, rules, "antigen", test_date.year, warnings)
    passage, harvest = _passage(passage_raw, passages, warnings)
    lab_id = re.sub(r"\s+-\s+", "-", lab_id)
    antigen = Antigen(
        name=name.name,
        raw_name=name_raw,
        passage=passage,
        passage_class=passages.passage_class(passage),
        passage_date=harvest,
        date=date or None,
        lab_ids=[lab_id] if lab_id else [],
        reference="R" in item.get("S", ""),
        lineage=lineage,
        reassortant=reassortant or name.reassortant,
        annotations=[*name.annotations, *annotations.split()],
        source={"ace": True, "passage": passage_raw},
    )
    if renamed:
        antigen.source[aliases.SOURCE_KEY] = renamed
    return antigen


def _serum(row, subtype, lineage, lab, passages, rules, test_date, warnings) -> Serum:  # noqa: ANN001
    name_raw, reassortant, annotations, serum_id, passage_raw, species = row
    name, renamed = _name(name_raw, subtype, lab, rules, "serum", test_date.year, warnings)
    passage, harvest = _passage(passage_raw, passages, warnings)
    serum = Serum(
        name=name.name,
        raw_name=name_raw,
        serum_id=f"{lab} {serum_id}" if serum_id else "",
        passage=passage,
        passage_class=passages.passage_class(passage),
        passage_date=harvest,
        species="" if species.upper() in ("", "FERRET") else species.upper(),
        lineage=lineage,
        reassortant=reassortant or name.reassortant,
        annotations=[*name.annotations, *annotations.split()],
        source={"ace": True, "passage": passage_raw},
    )
    if renamed:
        serum.source[aliases.SOURCE_KEY] = renamed
    return serum


def _titres(t: dict[str, Any], n_ag: int, n_sr: int) -> list[list[list[str]]]:
    sparse = t.get("d", [])
    dense = t.get("l") or [[row.get(str(j), "*") for j in range(n_sr)] for row in sparse]
    if len(dense) != n_ag or any(len(row) != n_sr for row in dense):
        raise ValueError(f"titre table is {len(dense)} rows, not {n_ag} x {n_sr}")
    return [[[] if v == "*" else [v] for v in row] for row in dense]
