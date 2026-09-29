"""Read Crick's HI workbooks: the Worldwide Influenza Centre's report tables, one per sheet.

Layout (the report-table template, used since at least 2019):

- a title naming the virus type and the test date (``Table H1-02. Antigenic analyses of
  influenza A(H1N1)pdm09 viruses (2026-04-19)``); an H3 title also names the red cells
  (``(Guinea Pig RBC with 20nM Oseltamivir)``);
- the sera in columns, described by labelled rows: two rows of the serum's name, abbreviated
  (``B/Exa`` over ``60/08``; a reassortant ``IVR-999`` over ``A/Exampleville/1/19``), then
  ``Passage`` (or ``Passage history``: ``Egg``, ``MDCK``, ``Egg J141O``), ``Ferret number``
  (``F12/18*2``: the serum id and footnote marks) and ``Genetic group``. The ``Ferret
  number`` label is the anchor: ``Passage`` is the row above it and the name rows the two
  filled rows above that;
- antigen columns labelled ``Viruses``, ``Other information`` (variant tags, clone numbers,
  and on some sheets the virus's HA substitutions), ``Collection date``, then the passage
  in the column just left of the sera; test viruses may carry Crick's sample number in a
  column of their own left of the name;
- antigen rows under ``REFERENCE VIRUSES`` and ``TEST VIRUSES``; below them a footnote
  legend (``1 < = <40; 2 < = <10; 3 hyperimmune sheep serum; ND = Not Done``) and vaccine
  marks under the serum columns. ``ND`` and ``*`` (not tested) are ``titre_tokens`` rules; a
  serum column with nothing else (a ferret number "to be given") is dropped and counted, and
  so is a lone titre in a row with no name (the template's diagonal of homologous titres left
  under the last reference antigen); two or more titres without a name are an error.

**A bare ``<`` means the serum's lowest dilution**, which the legend gives per footnote mark:
a serum ``F12/18*2`` reads ``<`` as ``<10`` when the legend says ``2 < = <10``. A ``<`` whose
serum has no mark, or two, is an error: never a guess. A mark the legend calls hyperimmune
sheep serum sets the serum's species.

**Other information is mostly description.** Which variant tags and clone numbers are part of
an antigen's identity is data: the ``identity_tags`` rules (lab, subtype), e.g. one position's
tags for one subtype. A tag or clone number no rule names, lists of HA substitutions, ``?`` and
notes describe the virus: they are counted ("tags: not identity") and the cell is kept verbatim
in the antigen's ``source``. The same filter applies to tags written in the virus name
(``Isolate 2 J141O``) and in a serum's passage row (``Egg J141O``), so a tag counts the same
wherever Crick wrote it; an isolate number is always identity. ae annotated the whole cell
where it read the column, so the same virus got a different identity from sheet to sheet.

**Serum names are abbreviations**, resolved to the table's own antigens as for VIDRL
(``af.tables.abbrev``), here with the year as well as the isolate. The serum id keeps the
lab that raised the ferret when the sheet names one (``NIB F01/21``, ``St Jude's F18/20``):
ae dropped it, which makes another lab's ferret look like Crick's.

**Neutralisation (PRN) tables** use the same template, with "Plaque Reduction
Neutralisation" in the title: the assay is PRN and there are no red cells. Their titres are
measured, not dilutions (``229``, ``436.5``): any positive value is kept, a decimal rounded
half up to a whole titre and counted (tables hold whole titres). The sheets have no footnote
legend, so a bare ``<`` needs a ``titre_tokens`` rule (assay PRN). Crick keeps each season's
PRN tests in one growing workbook, one sheet per test: a dated file reads the sheet of its
date, as for HI.

A workbook may hold other subtypes' tables (one per sheet) and other sheets (clade lists,
template notes): a sheet with no ``Ferret number`` row is not a table, and a table of
another subtype is reported and not read.
"""

from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import abbrev, aliases, dates
from .cdc import ReadResult
from .model import Antigen, Serum, Table
from .passage import PassageParser
from .rules import Rules
from .sheet import Sheet, SheetError, apply_cell_fixes, load

FERRET = re.compile(r"\s*Ferret\s+number\s*", re.IGNORECASE)
PASSAGE_LABEL = re.compile(r"\s*Passage(\s+history)?\s*", re.IGNORECASE)
ISO_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
RBC_IN_TITLE = re.compile(r"\(\s*(Guinea\s+Pig|Turkey|Chicken)\s+RBC", re.IGNORECASE)
RBC = {"GUINEA PIG": "guinea-pig", "TURKEY": "turkey", "CHICKEN": "chicken"}
TITRE = re.compile(r"[<>]?[1-9]\d*")
MEASURED = re.compile(r"[<>]?(?:[1-9]\d*|0)(?:\.\d+)?")  # PRN reads a titre, not a dilution
NEUTRALISATION = re.compile(r"Neutrali[sz]ation", re.IGNORECASE)
# a variant tag: J141, J141O, J141O/J; not part of a longer word or number
TAG = re.compile(r"(?<![A-Za-z0-9])([A-Z]\d{1,3}(?:[A-Z](?:/[A-Z])?)?)(?![A-Za-z0-9/])")
CLONE = re.compile(r"clone\s*(\d+(?:\.\d+)*)", re.IGNORECASE)  # "clone 37", "clone 3.4.1"
ISOLATE = re.compile(r"(?:Isolate|Isol|Isl)\s*(\d+)#?", re.IGNORECASE)
# "1 < = <40", "2< =<10": footnote mark and the titre a bare "<" stands for
LESS_THAN = re.compile(r"(\d)\s*<\s*=\s*(<\s*\d+)")
LESS_THAN_ALONE = re.compile(r"\s*<\s*=\s*(<\s*\d+)\s*")  # the mark is in the cell to its left
SHEEP = re.compile(r"(\d)\s*hyperimmune\s+sheep\s+serum", re.IGNORECASE)
MARKS = re.compile(r"(.*?)\s*\*\s*(\d(?:\s*,\s*\d)*)\s*")  # "F12/18*2", "Sh 5, 6*1,3"
SERUM_WORD = {"EGG": "E?", "MDCK": "MDCK?", "CELL": "MDCK?", "SIAT": "SIAT?", "HCK": "HCK?"}
REASSORTANT_FIRST = re.compile(r"\s*([A-Za-z]+(?:\s+X)?-?\s*\d+[A-Za-z]?)\s*\(([^()]+/[^()]+)\)\s*")


class CrickError(SheetError):
    pass


class NoTestDate(CrickError):
    """A sheet laid out as a table whose title has no test date: a summary or comparison
    Crick compiled from several tests."""


def read(
    paths: list[Path], rules: Rules, *, lab: str, subtype: str, lineage: str = ""
) -> ReadResult:
    """``subtype`` and ``lineage`` select the sheets to read: a workbook may hold several
    subtypes' tables. A file named with a date (whocc-tables: ``...-YYYYMMDD.xlsx``) holds
    that day's test; its sheets of other days are reported, not read."""
    result = ReadResult(tables=[], rows=0)
    for path in paths:
        data = path.read_bytes()
        provenance = {
            "file": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "reader": "af.tables.crick",
        }
        try:
            sheets = load(path)
        except Exception as err:  # openpyxl raises zip and XML errors of many kinds
            result.errors.append(f"{path.name}: not a readable workbook ({type(err).__name__})")
            continue
        file_date = _file_date(path)
        read_here, errors_before = 0, len(result.errors)
        for sheet in sheets:
            if not sheet.find(FERRET.pattern, stop=40):
                if any(any(row) for row in sheet.rows):
                    result.skipped_tests.append(f"{sheet.where(0)}: no 'Ferret number' row")
                continue
            try:
                reader = SheetReader(sheet, rules, lab, subtype, lineage)
                if not reader.wanted:
                    result.skipped_tests.append(
                        f"{sheet.where(0)}: a {reader.title_type} table, not {subtype} {lineage}"
                    )
                    continue
                if file_date is not None and reader.test_date.isoformat() != file_date:
                    result.skipped_tests.append(
                        f"{sheet.where(0)}: test date {reader.test_date} is not the file's"
                        f" ({file_date}): another test's sheet, not read"
                    )
                    continue
                table = reader.table()
            except NoTestDate as err:
                if file_date is not None:  # a dated file is one test: its table has a date
                    result.errors.append(str(err))
                else:
                    result.skipped_tests.append(f"{err} (a summary sheet, not a test)")
                continue
            except (SheetError, dates.DateError, LookupError, ValueError) as err:
                result.errors.append(str(err))
                continue
            read_here += 1
            table.provenance = dict(provenance)
            result.dropped.update(table.dropped)
            result.rows += len(table.antigens)
            if problems := table.check() or aliases.check_titres(table, rules):
                result.errors.extend(f"{sheet.where(0)}: {p}" for p in problems)
            result.tables.append(table)
        if file_date is not None and read_here != 1 and len(result.errors) == errors_before:
            result.errors.append(f"{path.name}: {read_here} sheets with the file's test date")
    return result


@dataclass
class Header:
    ferret_row: int
    label_col: int
    serum_cols: list[int]
    first_antigen_row: int
    last_antigen_row: int
    name_rows: tuple[int, int]  # the serum name's two rows, upper first


class SheetReader:
    def __init__(self, sheet: Sheet, rules: Rules, lab: str, subtype: str, lineage: str):
        self.s = sheet
        self.rules = rules
        self.lab = lab
        self.subtype = subtype
        self.lineage = lineage
        self.passages = PassageParser(rules.passage_tokens, lab)
        convention = rules.lab_conventions.lookup(lab=lab)
        if convention is None:
            raise CrickError(f"no lab_conventions rule for {lab}")
        self.date_order = convention["date_order"]
        self.warnings: list[str] = apply_cell_fixes(sheet, rules.cell_fixes, lab)
        self.dropped: Counter[str] = Counter()
        self.header = self._header()
        self.title = " ".join(
            v for r in range(self.header.name_rows[0]) for v in self.s.rows[r] if v
        )
        self.title_type = self._title_type()
        self.wanted = self.title_type == (subtype, lineage)
        self.assay = "PRN" if NEUTRALISATION.search(self.title) else "HI"
        self.test_date = self._test_date() if self.wanted else dt.date.min

    def fail(self, r: int, c: int | None, message: str) -> CrickError:
        return CrickError(f"{self.s.where(r, c)}: {message}")

    # -- layout ---------------------------------------------------------------------------

    def _header(self) -> Header:
        hits = self.s.find(FERRET.pattern, stop=40)
        if len(hits) != 1:
            raise self.fail(0, None, f"{len(hits)} 'Ferret number' labels")
        fr, fc = hits[0]
        if not PASSAGE_LABEL.fullmatch(self.s.cell(fr - 1, fc)):
            raise self.fail(fr - 1, fc, "no 'Passage' label above 'Ferret number'")
        width = max(len(row) for row in self.s.rows)
        cols = [
            c
            for c in range(fc + 1, width)
            if self.s.cell(fr, c) or self.s.cell(fr - 3, c) or self.s.cell(fr - 2, c)
        ]
        # a serum column has titres; trailing header cells without them (notes) are not sera
        antigen_rows = [
            r
            for r in range(fr + 1, len(self.s.rows))
            if any("/" in self.s.cell(r, c) for c in range(fc + 1))
            and sum(self._is_reading(self.s.cell(r, c)) for c in cols) >= 2
        ]
        if not antigen_rows:
            raise self.fail(fr, None, "no antigen rows under the serum rows")
        serum_cols = [
            c for c in cols if any(self._is_reading(self.s.cell(r, c)) for r in antigen_rows)
        ]
        if len(serum_cols) < 2:
            raise self.fail(fr, None, f"{len(serum_cols)} serum columns")
        if serum_cols != list(range(serum_cols[0], serum_cols[-1] + 1)):
            raise self.fail(fr, serum_cols[0], f"serum columns with gaps: {serum_cols}")
        filled = [r for r in range(fr - 2, -1, -1) if any(self.s.cell(r, c) for c in serum_cols)][
            :2
        ]
        if len(filled) != 2:
            raise self.fail(fr - 1, fc, "no two serum name rows above 'Passage'")
        return Header(fr, fc, serum_cols, antigen_rows[0], antigen_rows[-1], (filled[1], filled[0]))

    @staticmethod
    def _is_reading(text: str) -> bool:
        t = re.sub(r"\s+", "", text)
        return bool(MEASURED.fullmatch(t)) or t in ("<", ">", "ND", "*", "-")

    def _title_type(self) -> tuple[str, str]:
        t = self.title
        if re.search(r"A\s*\(\s*H1(N1)?\s*\)?\s*pdm|A\s*\(\s*H1\s*N1\s*\)", t, re.IGNORECASE):
            return "A(H1N1)", ""
        if re.search(r"A\s*\(\s*H3\s*N2\s*\)", t, re.IGNORECASE):
            return "A(H3N2)", ""
        if re.search(r"influenza\s+B\b", t, re.IGNORECASE):
            if re.search(r"Victoria", t, re.IGNORECASE):
                return "B", "VICTORIA"
            if re.search(r"Yamagata", t, re.IGNORECASE):
                return "B", "YAMAGATA"
            raise self.fail(0, None, f"B table with no lineage in the title {t[:90]!r}")
        raise self.fail(0, None, f"no virus type in the title {t[:90]!r}")

    def _test_date(self) -> dt.date:
        found = sorted(set(ISO_DATE.findall(self.title)))
        if len(found) != 1:
            message = f"{len(found)} dates in the title {self.title[:90]!r}"
            if not found:
                raise NoTestDate(f"{self.s.where(0)}: {message}")
            raise self.fail(0, None, message)
        day = dt.date.fromisoformat(found[0])
        # the sheet name, when it is a date, is DDMMYY: a second witness
        if m := re.search(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)", self.s.name):
            try:
                named = dt.date(2000 + int(m[3]), int(m[2]), int(m[1]))
            except ValueError:
                named = None
            if named != day:
                self.warnings.append(
                    f"{self.s.where(0)}: sheet name {self.s.name!r} is not the title's date {day}"
                )
        return day

    def _rbc(self) -> str:
        if self.assay != "HI":  # neutralisation: no red cells
            return ""
        if m := RBC_IN_TITLE.search(self.title):
            return RBC[re.sub(r"\s+", " ", m[1]).upper()]
        rule = self.rules.table_defaults.lookup(lab=self.lab, subtype=self.subtype, assay="HI")
        if rule is None:
            raise self.fail(0, None, "no red cells in the title and no table_defaults rule")
        return rule["rbc"]

    def _labelled_col(self, pattern: str, *, required: bool) -> int | None:
        h = self.header
        hits = {
            c
            for r in range(h.first_antigen_row)
            for c in range(h.serum_cols[0])
            if re.fullmatch(pattern, self.s.cell(r, c), re.IGNORECASE)
        }
        if len(hits) > 1 or (required and not hits):
            raise self.fail(0, None, f"{len(hits)} columns labelled {pattern!r}")
        return hits.pop() if hits else None

    # -- the table ------------------------------------------------------------------------

    def table(self) -> Table:
        rbc = self._rbc()
        prefix = {"A(H1N1)": "h1pdm", "A(H3N2)": "h3"}.get(self.subtype) or {
            "VICTORIA": "bvic",
            "YAMAGATA": "byam",
        }.get(self.lineage, "b")
        group = "-".join(p for p in (prefix, self.assay.lower(), rbc, self.lab.lower()) if p)
        legend = self._legend()
        antigens, rows = self._antigens()
        sera = [self._serum(c, antigens, legend) for c in self.header.serum_cols]
        titres = [
            [self._titre(r, c, sera[i], legend) for i, c in enumerate(self.header.serum_cols)]
            for r in rows
        ]
        # a serum whose every cell is "*" or "ND" (not tested; a ferret number "TBG")
        used = [i for i in range(len(sera)) if any(row[i] for row in titres)]
        self.dropped["sera: no readings"] += len(sera) - len(used)
        sera = [sera[i] for i in used]
        titres = [[row[i] for i in used] for row in titres]
        keep = [i for i, row in enumerate(titres) if any(row)]
        self.dropped["antigens: no readings"] += len(titres) - len(keep)
        return Table(
            table_id="",
            group=group,
            lab=self.lab,
            subtype=self.subtype,
            lineage=self.lineage,
            assay=self.assay,
            rbc=rbc,
            date=self.test_date.isoformat(),
            date_suffix=0,
            source_key=f"{self.lab} xlsx {self.s.path.name} [{self.s.name}]",
            antigens=[antigens[i] for i in keep],
            sera=sera,
            titres=[titres[i] for i in keep],
            meta={"file": self.s.path.name, "sheet": self.s.name},
            dropped=dict(sorted((k, v) for k, v in self.dropped.items() if v)),
            warnings=sorted(set(self.warnings)),
        )

    # -- footnotes ------------------------------------------------------------------------

    def _legend(self) -> dict[str, dict[str, str]]:
        """Footnote mark -> {"<": "<40"} and/or {"species": "SHEEP"}, from the legend
        below the antigens (and, on older sheets, beside the title)."""
        h = self.header
        legend: dict[str, dict[str, str]] = {}
        for r in range(len(self.s.rows)):
            if h.first_antigen_row <= r <= h.last_antigen_row:
                continue
            for c in range(h.serum_cols[0]):
                text = self.s.cell(r, c)
                if not text:
                    continue
                for mark, value in LESS_THAN.findall(text):
                    self._legend_entry(legend, mark, "<", value, r, c)
                for mark in SHEEP.findall(text):
                    self._legend_entry(legend, mark, "species", "SHEEP", r, c)
                if m := LESS_THAN_ALONE.fullmatch(text):
                    mark = self.s.cell(r, c - 1).strip() if c else ""
                    if not re.fullmatch(r"\d", mark):
                        raise self.fail(r, c, f"footnote {text!r} with no mark to its left")
                    self._legend_entry(legend, mark, "<", m[1], r, c)
        return legend

    def _legend_entry(
        self, legend: dict[str, dict[str, str]], mark: str, key: str, value: str, r: int, c: int
    ) -> None:
        value = re.sub(r"\s+", "", value)
        old = legend.setdefault(mark, {}).get(key)
        if old is not None and old != value:
            raise self.fail(r, c, f"footnote {mark} says {key} {old!r} and {value!r}")
        legend[mark][key] = value

    # -- sera -----------------------------------------------------------------------------

    def _serum(self, c: int, antigens: list[Antigen], legend: dict[str, dict[str, str]]) -> Serum:
        fr = self.header.ferret_row
        raw_id = self.s.cell(fr, c)
        serum_id, marks = _serum_id(raw_id)
        if unknown := [m for m in marks if m not in legend]:
            # an error only if a bare "<" needs it (see _titre)
            self.warnings.append(f"{self.s.where(fr, c)}: footnote {unknown} not in the legend")
        species = {legend[m]["species"] for m in marks if "species" in legend.get(m, {})}
        r1, r2 = self.header.name_rows
        n1, n2 = self.s.cell(r1, c), self.s.cell(r2, c)
        written = f"{n1} {n2}".strip()
        if not n1 or not n2:
            raise self.fail(r1, c, f"serum name {written!r}: expected two rows")
        name = self._serum_name(r1, c, n1, n2, antigens)
        passage, tags = self._serum_passage(fr - 1, c)
        return Serum(
            name=name.name,
            raw_name=written,
            serum_id=f"{self.lab} {serum_id}" if serum_id else "",
            passage=passage,
            passage_class=self.passages.passage_class(passage),
            species=species.pop() if species else "",
            reassortant=name.reassortant,
            annotations=tags,
            lineage=self.lineage,
            source={"column": c + 1, "id": raw_id, "marks": marks},
        )

    def _serum_passage(self, r: int, c: int) -> tuple[str, list[str]]:
        """``Egg``, ``MDCK``, ``Egg J141O``: the cell/egg word gives the passage, a variant
        tag after it is an annotation."""
        text = self.s.cell(r, c).strip()
        m = re.fullmatch(r"([A-Za-z]+)(?:\s+(.+))?", text)
        if m is None or m[1].upper() not in SERUM_WORD:
            raise self.fail(r, c, f"serum passage {text!r}: not one of {sorted(SERUM_WORD)}")
        tags, _ = self._identity(m[2] or "")
        if m[2] and not tags and not _candidates(m[2]):
            raise self.fail(r, c, f"serum passage {text!r}: {m[2]!r} is not a variant tag")
        return SERUM_WORD[m[1].upper()], tags

    def _serum_name(self, r: int, c: int, n1: str, n2: str, antigens: list[Antigen]):
        written = f"{n1} {n2}"
        rule = self.rules.strain_aliases.find(
            written, lab=self.lab, subtype=self.subtype, applies_to="serum"
        )
        if rule is not None:
            problems: list[str] = []
            name, _ = aliases.parse_name(
                self.rules,
                rule["canonical"],
                lab=self.lab,
                subtype=self.subtype,
                applies_to="serum",
                warnings=problems,
                not_after=self.test_date.year,
            )
            self.warnings.extend(f"{self.s.where(r, c)}: {p}" for p in problems)
            self.warnings.append(f"{self.s.where(r, c)}: serum {written!r} named by {rule.where}")
            return abbrev.Named(name.name, name.reassortant)
        reassortant = ""
        if "/" not in n1:  # "IVR-999" over "A/Exampleville/1/19"
            reassortant = self._reassortant(n1)
            if not reassortant:
                raise self.fail(r, c, f"serum name {written!r}: {n1!r} is not a reassortant")
            text = n2
        else:  # "B/Exa" over "60/08", "A/Exa/" over "/1/2019"
            text = f"{n1.rstrip('/ ')}/{n2.lstrip('/ ')}"
        parts = [p.strip() for p in text.split("/")]
        if len(parts) < 3:
            raise self.fail(r, c, f"serum name {written!r}: no location/isolate/year")
        if len(parts) == 3:  # no type written
            parts = ["", *parts]
        location, isolate, year = "".join(parts[1:-2]), parts[-2], parts[-1]
        references = [a for a in antigens if a.reference]
        found = abbrev.match(location, isolate, year, references, reassortant) or abbrev.match(
            location, isolate, year, antigens, reassortant
        )
        if len(found) != 1:
            raise self.fail(
                r,
                c,
                f"serum {written!r} matches {len(found)} antigen names {sorted(found)[:4]}:"
                " add a strain_aliases rule (applies_to serum)",
            )
        return next(iter(found.values()))

    def _identity(self, text: str) -> tuple[list[str], int]:
        """(annotations, number of tags left out) from a name's extra text, an
        Other-information cell or a serum's passage row. An isolate number is identity;
        a variant tag or clone number is identity only when an ``identity_tags`` rule (lab,
        subtype) names it; everything else is description."""
        out: list[str] = []
        ignored = 0
        if m := ISOLATE.search(text):
            out.append(f"ISOLATE {m[1]}")
        for tag in _candidates(text):
            rule = self.rules.identity_tags.find(tag, lab=self.lab, subtype=self.subtype)
            if rule is None:
                ignored += 1
                continue
            value = (
                re.sub(rule["pattern"], rule["annotation"], tag, flags=re.IGNORECASE)
                if rule["kind"] == "regex"
                else rule["annotation"]
            ).upper()
            if value not in out:
                out.append(value)
        return out, ignored

    def _reassortant(self, text: str) -> str:
        from .names import split_extra

        found, annotation = split_extra(text, self.rules.reassortants, self.lab)
        return found if not annotation else ""

    # -- antigens -------------------------------------------------------------------------

    def _antigens(self) -> tuple[list[Antigen], list[int]]:
        """The antigens and the sheet row of each (titres are read once the sera are)."""
        h = self.header
        name_col = self._labelled_col(r"\s*Viruses\s*", required=True)
        assert name_col is not None
        other_col = self._labelled_col(r"\s*Other(\s+information)?\s*", required=False)
        date_col = self._labelled_col(r"\s*Collection(\s+date)?\s*", required=True)
        assert date_col is not None
        passage_col = h.serum_cols[0] - 1
        antigens, rows = [], []
        reference = False
        for r in range(h.ferret_row + 1, h.last_antigen_row + 1):
            raw = self.s.cell(r, name_col)
            if re.fullmatch(r"\s*REFERENCE\s+VIRUS(ES)?\s*", raw, re.IGNORECASE):
                reference = True
                continue
            if re.fullmatch(r"\s*TEST\s+VIRUS(ES)?\s*", raw, re.IGNORECASE):
                reference = False
                continue
            readings = [self.s.cell(r, c) for c in h.serum_cols]
            has_readings = any(self._is_reading(v) for v in readings)
            if not raw:
                count = sum(self._is_reading(v) for v in readings)
                if count > 1:
                    raise self.fail(r, name_col, "row with titres but no strain name")
                if count == 1:
                    # the template's diagonal of homologous titres left under the last
                    # reference antigen: a lone cell that belongs to no virus
                    self.dropped["cells: no antigen"] += 1
                    self.warnings.append(f"{self.s.where(r)}: one titre in a row with no name")
                continue
            if not has_readings:
                if "/" in raw:
                    self.dropped["antigens: no readings"] += 1
                continue
            if self.rules.control_antigens.find(raw, lab=self.lab) is not None:
                self.dropped["antigens: control"] += 1
                continue
            other = self.s.cell(r, other_col) if other_col is not None else ""
            antigens.append(self._antigen(r, name_col, raw, other, date_col, passage_col))
            antigens[-1].reference = reference
            rows.append(r)
        return antigens, rows

    def _antigen(
        self, r: int, name_col: int, raw: str, other: str, date_col: int, passage_col: int
    ) -> Antigen:
        text = raw
        if m := REASSORTANT_FIRST.fullmatch(raw):  # "IVR-999 (A/Exampleville/1/2019)"
            text = f"{m[2]} {m[1]}"
        problems: list[str] = []
        name, renamed = aliases.parse_name(
            self.rules,
            text,
            lab=self.lab,
            subtype=self.subtype,
            applies_to="antigen",
            warnings=problems,
            not_after=self.test_date.year,
        )
        self.warnings.extend(f"{self.s.where(r, name_col)}: {p}" for p in problems)
        raw_passage = self.s.cell(r, passage_col)
        passage = self._passage(raw_passage, r, passage_col)
        collected = self.s.cell(r, date_col)
        # the name's extras ("Isolate 2 J141O") and the Other-information cell go through the
        # same filter, so a tag counts the same wherever Crick wrote it
        annotations: list[str] = []
        for text, in_name in [*((a, True) for a in name.annotations), (other, False)]:
            kept, ignored = self._identity(text)
            if in_name:  # the rest of a name's extra text stays, as for other labs
                rest = re.sub(r"[\s(),]+", " ", CLONE.sub("", TAG.sub("", ISOLATE.sub("", text))))
                kept += [rest.strip()] if rest.strip() else []
            annotations += [a for a in kept if a not in annotations]
            if ignored:
                self.dropped["tags: not identity"] += ignored
        lab_ids = [
            f"{self.lab}#{v}"
            for c in range(name_col)
            if re.fullmatch(r"\d{5,}", v := self.s.cell(r, c))
        ]
        antigen = Antigen(
            name=name.name,
            raw_name=raw,
            passage=passage,
            passage_class=self.passages.passage_class(passage),
            date=self._date(collected, r, date_col),
            lab_ids=lab_ids,
            reassortant=name.reassortant,
            annotations=annotations,
            lineage=self.lineage,
            source={"row": r + 1, "passage": raw_passage},
        )
        if other:
            antigen.source["other_information"] = other
        if renamed:
            antigen.source[aliases.SOURCE_KEY] = renamed
        return antigen

    def _passage(self, raw: str, r: int, c: int) -> str:
        try:
            read = self.passages.parse(_passage_text(raw, self.passages))
        except ValueError as err:
            raise self.fail(r, c, str(err)) from err
        self.warnings.extend(f"{self.s.where(r, c)}: {p}" for p in read.problems)
        return read.text

    def _date(self, text: str, r: int, c: int) -> str | None:
        if re.fullmatch(r"[\s/.-]*", text):  # "-": no date
            return None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T[\d:]+", text):  # an Excel date cell with a time
            text = text[:10]
        try:
            day, warning = dates.parse(text, self.date_order, not_after=self.test_date)
        except dates.DateError as err:
            raise self.fail(r, c, str(err)) from err
        if warning:
            self.warnings.append(f"{self.s.where(r, c)}: {warning}")
        if day > self.test_date.isoformat():
            self.warnings.append(f"{self.s.where(r, c)}: date {text!r} is after the test; left out")
            return None
        return day

    # -- titres ---------------------------------------------------------------------------

    def _titre(self, r: int, c: int, serum: Serum, legend: dict[str, dict[str, str]]) -> list[str]:
        raw = self.s.cell(r, c)
        if not raw:
            self.dropped["cells: blank"] += 1
            return []
        if (rule := self.rules.titre_tokens.find(raw, lab=self.lab, assay=self.assay)) is not None:
            return [] if rule["titre"] == "*" else [rule["titre"]]
        text = re.sub(r"\s+", "", raw)
        if text == "<":
            values = {legend[m]["<"] for m in serum.source["marks"] if "<" in legend.get(m, {})}
            if len(values) != 1:
                raise self.fail(
                    r, c, f"'<' for serum {serum.source['id']!r}: {len(values)} footnote values"
                )
            return [values.pop()]
        if (
            self.assay == "HI"
            and TITRE.fullmatch(text)
            and abbrev.is_dilution(int(text.lstrip("<>")))
        ):
            return [text]
        if self.assay != "HI" and MEASURED.fullmatch(text) and float(text.lstrip("<>")) > 0:
            return [self._whole(text)]
        raise self.fail(
            r, c, f"titre {raw!r} is not a dilution (10, 20, 40...) nor a titre_tokens rule"
        )

    def _whole(self, text: str) -> str:
        """A neutralisation titre as a whole number: Crick reports some interpolated reads
        with a decimal (``436.5``), and tables hold whole titres. Half rounds up, as ae did."""
        sign, number = (text[0], text[1:]) if text[0] in "<>" else ("", text)
        if "." not in number:
            return text
        self.dropped["cells: decimal rounded"] += 1
        return sign + str(int(decimal.Decimal(number).quantize(0, decimal.ROUND_HALF_UP)))


def _serum_id(raw: str) -> tuple[str, list[str]]:
    """``F12/18*2`` -> ("F12/18", ["2"]); ``Sh 5, 6, 7*1,3`` -> ("SH5/6/7", ["1", "3"]);
    ``NEW F12/26`` -> ("F12/26", []): ``NEW`` marks a serum new to the table."""
    text, marks = raw.strip(), []
    if m := MARKS.fullmatch(text):
        text, marks = m[1], re.split(r"\s*,\s*", m[2])
    text = re.sub(r"^NEW\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*", "/", text.upper())
    if re.fullmatch(r"SH\s*[\d/]+", text):  # a sheep pool: "SH 5/6/7" as ae wrote it
        text = re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text), marks


def _passage_text(raw: str, parser: PassageParser) -> str:
    """Crick's passage notation in the form the passage parser reads. Crick writes a count
    after a space (``MDCK 1``), as ``P<n>`` after the cell name (``SIAT P1``, ``MDCKP2``,
    ``MDCK-SIAT1 P2``) or as ``C<n>`` after a space (``MDCK C1``); ``+<n>`` for more of the
    same cells (``C2+1``); a bare step name for an
    unknown count (``P1/MDCK``); a space before a note (``E3 (Am1Al2)``); and the virus
    concentration after the passage (``E3/E1 10-3``), which is not part of it."""
    text = re.sub(r"\s+10-\d+.*$", "", raw.strip())
    text = re.sub(r"\s+\(", "(", text)
    out = []
    for part in (q.strip() for q in text.split("/")):
        part = re.sub(r"(?<=[A-Za-z])\s+(?=\d)", "", part)
        # "C2+1": one more passage of the same cells
        part = re.sub(r"([A-Za-z]+)(\d+)\+(\d+)", r"\1\2\1\3", part)
        m = re.fullmatch(r"(MDCK-SIAT1|[A-Za-z][A-Za-z-]*?[A-Za-z])(?:\s*P|\s+C)(\d+|[Xx])", part)
        if m and parser.is_step_name(m[1]):
            part = m[1] + m[2]
        elif parser.is_step_name(part):
            part += "X"  # a step with no count: an unknown number of passages
        out.append(part)
    return "/".join(out)


def _candidates(text: str) -> list[str]:
    """The variant tags and clone numbers written in ``text`` (upper case), in order."""
    found = [(m.start(), f"CLONE {m[1]}") for m in CLONE.finditer(text)]
    found += [(m.start(), m[1].upper()) for m in TAG.finditer(text)]
    return [t for _, t in sorted(found)]


def _file_date(path: Path) -> str | None:
    m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", path.name)
    if m is None:
        return None
    try:
        return dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat()
    except ValueError:
        return None
