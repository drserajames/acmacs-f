"""Read "AC Excel format 2.1" workbooks (CNIC sends its HI tables in this format).

The format is labelled, so every field is found by its label:

- header rows: ``AC Excel format 2.1``, then label/value pairs (``Test date``,
  ``Tested by (Lab)``, ``Assay (HI, VN, etc)``, ``RBC species``, ``Default flu type``);
- the ``TITERS`` row: antigen columns (name, then ``ID``, ``Specimen Date``), one column per
  serum headed by its index 1..n, then ``Neg.``, ``HA``, ``Back``, ``Passage``; the row
  below repeats the serum names;
- antigen rows (index in the first column) under ``Reference antigens`` / ``Test antigens``;
- the ``ANTISERA`` block: one row per serum index with name and serum ID.

Checks: every serum column has exactly one ANTISERA row with the same index and the same
name (a mismatch means a column and its serum were edited apart); every row between TITERS
and ANTISERA is an antigen, a section label, or blank; titres are valid; dates parse; the
lab in the sheet is the lab the config says.

Lab conventions come from rules: the flu-type labels (``flu_types``), ae-compatible name
spellings (``name_rewrites``), Chinese locations (:mod:`af.tables.locations`), and CNIC's
passage notation ``E3+1`` = E3 then E1 (``lab_conventions.passage_plus``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections import Counter
from pathlib import Path

from . import aliases, dates
from .cdc import ReadResult
from .locations import ChineseLocations
from .model import Antigen, Serum, Table
from .passage import PassageParser
from .rules import Rules
from .sheet import Sheet, SheetError, apply_cell_fixes, load

FORMAT_LABEL = "AC Excel format 2.1"
ASSAYS = {"HI": "HI"}
RBC = {"GUINEA PIG": "guinea-pig", "TURKEY": "turkey", "CHICKEN": "chicken"}
SECTION = re.compile(r"(reference|test) antigens", re.IGNORECASE)
STEP = re.compile(r"([A-Za-z]+)[0-9X]*$")  # the last passage step's name, for "E3+1"


class AC21Error(SheetError):
    pass


def read(paths: list[Path], rules: Rules, locations: ChineseLocations, *, lab: str) -> ReadResult:
    """Read every workbook; ``lab`` is what the sheets must say they are from."""
    result = ReadResult(tables=[], rows=0)
    for path in paths:
        data = path.read_bytes()
        provenance = {
            "file": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "reader": "af.tables.ac21",
        }
        for sheet in load(path):
            if not sheet.find(re.escape(FORMAT_LABEL)):
                result.skipped_tests.append(f"{sheet.where(0)}: no '{FORMAT_LABEL}' label")
                continue
            try:
                table = SheetReader(sheet, rules, locations, lab).table()
            except (SheetError, dates.DateError, LookupError, ValueError) as err:
                result.errors.append(str(err))
                continue
            table.provenance = dict(provenance)
            result.dropped.update(table.dropped)
            result.rows += len(table.antigens)
            if problems := table.check() or aliases.check_titres(table, rules):
                result.errors.extend(f"{sheet.where(0)}: {p}" for p in problems)
            result.tables.append(table)
    return result


class SheetReader:
    def __init__(self, sheet: Sheet, rules: Rules, locations: ChineseLocations, lab: str):
        self.s = sheet
        self.rules = rules
        self.locations = locations
        self.lab = lab
        self.passages = PassageParser(rules.passage_tokens, lab)
        convention = rules.lab_conventions.lookup(lab=lab)
        if convention is None:
            raise AC21Error(f"no lab_conventions rule for {lab}")
        self.date_order = convention["date_order"]
        self.passage_plus = convention["passage_plus"]
        # hand repairs to single cells, named and checked (af.tables.sheet.apply_cell_fixes)
        self.warnings: list[str] = apply_cell_fixes(sheet, rules.cell_fixes, lab)
        self.dropped: Counter[str] = Counter()
        self.index_reordered = False
        self.test_year: int | None = None

    def fail(self, r: int, c: int | None, message: str) -> AC21Error:
        return AC21Error(f"{self.s.where(r, c)}: {message}")

    def table(self) -> Table:
        header = self._header()
        if header["Tested by (Lab)"].upper() != self.lab.upper():
            raise self.fail(
                0, None, f"sheet says lab {header['Tested by (Lab)']!r}, config says {self.lab!r}"
            )
        test_date = dt.date.fromisoformat(dates.parse(header["Test date"], self.date_order)[0])
        self.test_year = test_date.year
        assay = ASSAYS.get(header["Assay (HI, VN, etc)"].upper())
        if assay is None:
            raise self.fail(
                0, None, f"assay {header['Assay (HI, VN, etc)']!r} not read by this reader"
            )
        rbc = RBC.get(header["RBC species"].strip().upper())
        if rbc is None:
            raise self.fail(0, None, f"RBC species {header['RBC species']!r}")
        flu = self.rules.flu_types.find(header["Default flu type"], lab=self.lab)
        if flu is None:
            raise self.fail(
                0, None, f"flu type {header['Default flu type']!r} matches no flu_types rule"
            )
        subtype, lineage = flu["subtype"], "" if flu["lineage"] == "-" else flu["lineage"]
        prefix = {"A(H1N1)": "h1pdm", "A(H3N2)": "h3"}.get(subtype) or {
            "VICTORIA": "bvic",
            "YAMAGATA": "byam",
        }.get(lineage, "b")
        group = "-".join((prefix, assay.lower(), rbc, self.lab.lower()))
        titres_row, columns = self._serum_columns()
        antisera_row = self._antisera_row(titres_row)
        sera = self._sera(antisera_row, columns, subtype, lineage)
        antigens, titres = self._antigens(
            titres_row, antisera_row, columns, subtype, lineage, test_date
        )
        return Table(
            table_id="",
            group=group,
            lab=self.lab,
            subtype=subtype,
            lineage=lineage,
            assay=assay,
            rbc=rbc,
            date=test_date.isoformat(),
            date_suffix=0,
            source_key=f"{self.lab} xlsx {self.s.path.name} [{self.s.name}]",
            antigens=antigens,
            sera=sera,
            titres=titres,
            meta={
                "file": self.s.path.name,
                "sheet": self.s.name,
                "format": FORMAT_LABEL,
                "tested_by": header.get("Tested by (Person)", ""),
            },
            dropped=dict(sorted((k, v) for k, v in self.dropped.items() if v)),
            warnings=sorted(set(self.warnings)),
        )

    # -- layout ---------------------------------------------------------------------------

    def _header(self) -> dict[str, str]:
        r0, c0 = self.s.find_one(re.escape(FORMAT_LABEL))
        fields: dict[str, str] = {}
        for r in range(r0 + 1, min(r0 + 15, len(self.s.rows))):
            label, value = self.s.cell(r, c0), self.s.cell(r, c0 + 1)
            if label == "TITERS":
                break
            if label:
                fields[label] = value
        for need in (
            "Test date",
            "Tested by (Lab)",
            "Assay (HI, VN, etc)",
            "RBC species",
            "Default flu type",
        ):
            if not fields.get(need):
                raise self.fail(r0, c0, f"header field {need!r} missing or empty")
        return fields

    def _serum_columns(self) -> tuple[int, dict[int, int]]:
        """Serum index -> column. The serum columns are those with a name in the row under
        TITERS. The index row above them is the lab's own numbering and has been found
        shifted, incomplete and permuted: when it is a permutation of 1..n it decides the
        mapping (as ae did); otherwise columns map by position, with a warning."""
        r, _ = self.s.find_one(r"TITERS")
        header, names = self.s.rows[r], self.s.rows[r + 1]
        end = header.index("Neg.") if "Neg." in header else len(names)
        cols = [c for c in range(len(names)) if c < end and "/" in names[c]]
        if not cols:
            raise self.fail(r + 1, None, "no serum names under TITERS")
        indices = [header[c] if c < len(header) else "" for c in cols]
        expected = [str(i) for i in range(1, len(cols) + 1)]
        if indices == expected:
            return r, {i + 1: c for i, c in enumerate(cols)}
        if sorted(indices, key=lambda x: int(x) if x.isdigit() else 0) == expected:
            self.warnings.append(
                f"{self.s.where(r)}: serum index row {indices} reorders the columns; "
                "followed it (as ae did)"
            )
            self.index_reordered = True
            return r, {int(x): c for x, c in zip(indices, cols, strict=True)}
        self.warnings.append(
            f"{self.s.where(r)}: serum index row {indices} is not 1..n; columns taken by position"
        )
        return r, {i + 1: c for i, c in enumerate(cols)}

    def _antisera_row(self, titres_row: int) -> int:
        r, _ = self.s.find_one(r"ANTISERA", start=titres_row)
        return r

    # -- sera -----------------------------------------------------------------------------

    def _sera(self, r0: int, columns: dict[int, int], subtype: str, lineage: str) -> list[Serum]:
        labels = self.s.rows[r0 + 1]
        name_col = labels.index("Name") if "Name" in labels else 1
        id_col = labels.index("ID") if "ID" in labels else 2
        by_index: dict[int, Serum] = {}
        r = r0 + 2
        while r < len(self.s.rows) and re.fullmatch(r"\d+", self.s.cell(r, 0)):
            index = int(self.s.cell(r, 0))
            raw, serum_id = self.s.cell(r, name_col), self.s.cell(r, id_col)
            if index in by_index:
                raise self.fail(r, 0, f"serum index {index} twice")
            if not raw or not serum_id:
                raise self.fail(r, name_col, "serum without a name or an ID")
            problems: list[str] = []
            name, renamed = aliases.parse_name(
                self.rules,
                raw,
                lab=self.lab,
                subtype=subtype,
                applies_to="serum",
                warnings=problems,
                locations=self.locations,
                not_after=self.test_year,
            )
            self.warnings.extend(f"{self.s.where(r, name_col)}: {p}" for p in problems)
            serum = Serum(
                name=name.name,
                raw_name=raw,
                serum_id=f"{self.lab} {serum_id}",
                passage_class="unknown",  # AC Excel 2.1 gives no serum passage
                reassortant=name.reassortant,
                annotations=name.annotations,
                lineage=lineage,
                source={"row": r + 1, "index": index},
            )
            if renamed:
                serum.source[aliases.SOURCE_KEY] = renamed
            by_index[index] = serum
            r += 1
        if missing := sorted(set(columns) - set(by_index)):
            raise self.fail(r0, None, f"titre columns {missing} have no ANTISERA row")
        if extra := sorted(set(by_index) - set(columns)):
            self.dropped["sera: in ANTISERA without a titre column"] += len(extra)
        # The name over each titre column should be its ANTISERA row's name. Where it is not,
        # a column and its serum have drifted apart in the sheet: an error, unless the index
        # row decided the mapping (then the name row is the one out of step; warned).
        titres_row = self.s.find_one(r"TITERS")[0]
        for index, c in columns.items():
            over = self.s.cell(titres_row + 1, c)
            if self._same_name(over, by_index[index].raw_name):
                continue
            message = (
                f"column {index} says {over!r}, ANTISERA {index} says {by_index[index].raw_name!r}"
            )
            if not self.index_reordered:
                raise self.fail(titres_row + 1, c, message)
            self.warnings.append(
                f"{self.s.where(titres_row + 1, c)}: {message}; followed the index row"
            )
        return [by_index[i] for i in sorted(columns, key=lambda i: columns[i])]

    def _same_name(self, a: str, b: str) -> bool:
        """Equal once the lab's systematic spellings (name_rewrites) are applied."""

        def norm(text: str) -> str:
            rule = self.rules.name_rewrites.find(text, lab=self.lab)
            if rule is not None:
                text = re.compile(rule["pattern"], re.IGNORECASE).sub(rule["replacement"], text)
            return re.sub(r"\s+", " ", text).strip().casefold()

        return norm(a) == norm(b)

    # -- antigens -------------------------------------------------------------------------

    def _antigens(
        self,
        titres_row: int,
        antisera_row: int,
        columns: dict[int, int],
        subtype: str,
        lineage: str,
        test_date: dt.date,
    ) -> tuple[list[Antigen], list[list[list[str]]]]:
        header = self.s.rows[titres_row]
        below = self.s.rows[titres_row + 1]
        passage_col = header.index("Passage") if "Passage" in header else None
        if passage_col is None:
            raise self.fail(titres_row, None, "no Passage column")
        id_col = below.index("ID") if "ID" in below else None
        date_col = header.index("Specimen Date") if "Specimen Date" in header else None
        if id_col is None or date_col is None:
            raise self.fail(titres_row, None, "no ID or Specimen Date column")
        name_col = id_col - 1
        antigens, titres = [], []
        for r in range(titres_row + 2, antisera_row):
            row = self.s.rows[r]
            first = self.s.cell(r, 0)
            raw = self.s.cell(r, name_col)
            if not any(row):
                continue
            if not first and SECTION.fullmatch(raw or self.s.cell(r, 1)):
                continue
            if self.rules.control_antigens.find(raw, lab=self.lab) is not None:
                self.dropped["antigens: control"] += 1
                continue
            if not re.fullmatch(r"\d+", first):
                if raw.count("/") < 3 or not any(self.s.cell(r, c) for c in columns.values()):
                    raise self.fail(
                        r,
                        0,
                        "row between TITERS and ANTISERA is neither an antigen nor a section: "
                        f"{row[:4]}",
                    )
                self.warnings.append(f"{self.s.where(r, 0)}: antigen row without an index number")
            problems: list[str] = []
            name, renamed = aliases.parse_name(
                self.rules,
                raw,
                lab=self.lab,
                subtype=subtype,
                applies_to="antigen",
                warnings=problems,
                locations=self.locations,
                not_after=self.test_year,
            )
            self.warnings.extend(f"{self.s.where(r, name_col)}: {p}" for p in problems)
            passage = self.passages.parse(self._expand_plus(self.s.cell(r, passage_col)))
            self.warnings.extend(f"{self.s.where(r, passage_col)}: {p}" for p in passage.problems)
            collected = self.s.cell(r, date_col)
            antigen = Antigen(
                name=name.name,
                raw_name=raw,
                passage=passage.text,
                passage_class=self.passages.passage_class(passage.text),
                date=self._date(collected, r, date_col, test_date) if collected else None,
                lab_ids=[f"{self.lab}#{self.s.cell(r, id_col)}"] if self.s.cell(r, id_col) else [],
                reassortant=name.reassortant,
                annotations=name.annotations,
                lineage=lineage,
                source={"row": r + 1, "passage": self.s.cell(r, passage_col)},
            )
            if renamed:
                antigen.source[aliases.SOURCE_KEY] = renamed
            antigens.append(antigen)
            # same order as the sera: by column position in the sheet
            titres.append([self._titre(r, c) for c in sorted(columns.values())])
        keep = [i for i, row in enumerate(titres) if any(row)]
        self.dropped["antigens: no readings"] += len(titres) - len(keep)
        return [antigens[i] for i in keep], [titres[i] for i in keep]

    def _date(self, text: str, r: int, c: int, test_date: dt.date) -> str:
        """A date cell; an Excel serial number (a date cell typed as a number) is converted."""
        if re.fullmatch(r"[1-9]\d{4}", text):
            day = dt.date(1899, 12, 30) + dt.timedelta(days=int(text))
            self.warnings.append(
                f"{self.s.where(r, c)}: date {text} is an Excel serial number: {day}"
            )
            return day.isoformat()
        return dates.parse(text, self.date_order, not_after=test_date)[0]

    def _expand_plus(self, text: str) -> str:
        """CNIC's "E3+1" = E3 then E1: a bare count after '+' repeats the previous step's
        name; '+' separates steps like '/'. Only for labs whose convention says so."""
        if self.passage_plus != "repeat-previous" or "+" not in text:
            return text
        parts = [p.strip() for p in text.split("+")]
        out = [parts[0]]
        for part in parts[1:]:
            if re.fullmatch(r"\d+", part):
                m = STEP.search(out[-1])
                if m is None:
                    raise ValueError(f"passage {text!r}: nothing before '+{part}' to repeat")
                part = m[1] + part
            out.append(part)
        return "/".join(out)

    def _titre(self, r: int, c: int) -> list[str]:
        raw = self.s.cell(r, c)
        if not raw:
            self.dropped["cells: blank"] += 1
            return []
        if (rule := self.rules.titre_tokens.find(raw, lab=self.lab, assay="HI")) is not None:
            return [] if rule["titre"] == "*" else [rule["titre"]]
        if re.fullmatch(r"[<>]?[1-9]\d*", raw) and (raw[0] in "<>" or int(raw) >= 10):
            return [raw]
        raise self.fail(r, c, f"titre {raw!r} matches no titre_tokens rule")
