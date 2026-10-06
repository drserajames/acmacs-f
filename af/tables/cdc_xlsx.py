"""Read CDC's HI workbooks (the "RUN nnnn" sheets CDC emails when a test is not in the TSV).

Sarah, 24 Sep 2026: these are read so the CDC chains are complete (10 committed tables
2024-2025 exist only in this form). A sheet is found by its labels, never by fixed positions,
because CDC moves columns between runs:

- the title "HEMAGGLUTINATION INHIBITION REACTIONS OF INFLUENZA <subtype> VIRUSES" (+ OSELTAMIVIR);
- "DATE TESTED: <date>" (CDC writes M/D/Y), and "RBCS USED" with the species below it;
- the serum letters row (A, B, C ... over the titre columns), found just above the
  "REFERENCE VIRUSES" label; the antigen label row (CDC ID#, PASSAGE, DATE COLLECTED ...);
- antigen rows up to "SERUM CONTROL"; the serum block under "REFERENCE ANTISERA", one row per
  letter.

Every titre column must have exactly one serum row and every serum row a column; every row
in the antigen block is either read, a section label, blank, or dropped by a rule (counted).
ae's extractor read these sheets positionally and, in h3-hi-guinea-pig-cdc-20240912, took the
CUID column for the antigen passage.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections import Counter
from pathlib import Path

from af.util.subtypes import subtypes

from . import aliases, dates
from .cdc import LAB, ReadResult, _titre_order
from .model import Antigen, Serum, Table
from .passage import PassageParser
from .rules import Rule, Rules
from .sheet import Sheet, SheetError, apply_cell_fixes, load_or_error

TITLE = r"HEMAGGLUTINATION INHIBITION REACTIONS OF INFLUENZA (.+) VIRUSES"
TITLE_SUBTYPES = {
    "H3": ("A(H3N2)", ""),
    "A(H3N2)": ("A(H3N2)", ""),
    "A(H1N1)PDM09": ("A(H1N1)", ""),
    "H1N1PDM09": ("A(H1N1)", ""),
    "B/VICTORIA": ("B", "VICTORIA"),
    "B VICTORIA LINEAGE": ("B", "VICTORIA"),
    "TYPE B VICTORIA LINEAGE": ("B", "VICTORIA"),  # 2013-19 layout
}
RBC = {"GUINEA PIG": "guinea-pig", "TURKEY": "turkey"}
# The passage cell: "S1(07/19/2024)<NY>", "E4/E1(9/13/2024)LOT#10", "S2".
PASSAGE_CELL = re.compile(
    r"(?P<passage>[^(<]*)\s*(?:\((?P<date>[^)]*)(?P<close>\))?)?\s*(?P<extra>.*)"
)
SITE = re.compile(r"<?([A-Z]{2})>?")  # the state lab that isolated it: kept in source
LOT = re.compile(r"LOT\s*#\s*\d+", re.IGNORECASE)  # distinguishes egg lots: an annotation
TITRE_LIKE = re.compile(r"[<>]?\s*\d+")
LETTER = re.compile(r"[A-Z]|([A-Z])\1")  # serum letters: A ... Z, AA, BB ...
# the BOOSTED column: Y/N since 2024; words in the 2013-19 sheets, where a pre-boost bleed is
# the ferret's serum before its boost, so not boosted
BOOSTED = {
    "Y": True,
    "BOOSTED": True,
    "N": False,
    "": False,
    "NOT BOOSTED": False,
    "UN BOOSTED": False,
    "PRE BOOST BLEED": False,
    "NOT BOOSTED (PRE BOOST BLEED)": False,
}
SECTION = re.compile(r"(REFERENCE|TEST) (VIRUSES|ANTIGENS)", re.IGNORECASE)
# the reference label: "REFERENCE VIRUSES"; the 2013-16 sheets say "REFERENCE ANTIGENS"
REFERENCE = r"REFERENCE (VIRUSES|ANTIGENS)"
TESTED = r"(?:DATE TESTED|Test Date):\s*(.+)|TESTED\s+(\d.+)"


class CDCSheetError(SheetError):
    pass


def _expand_lots(lot: str) -> str:
    """A pooled serum's lots as CDC's TSV and season files write them: "2013-029, 030" ->
    "2013-029,2013-030" (the 2013-19 sheets give later lots of the year as bare numbers)."""
    out, year = [], ""
    for part in (p.strip() for p in lot.split(",")):
        if m := re.fullmatch(r"(\d{4})-\d+", part):
            year = m[1]
        elif year and re.fullmatch(r"\d{3}", part):
            part = f"{year}-{part}"
        out.append(part)
    return ",".join(out)


def _is_dilution(n: int) -> bool:
    """10 x 2^k: a value off the series is a typing error (32 for 320, 2180 for 1280)."""
    return n >= 10 and n % 10 == 0 and (n // 10) & (n // 10 - 1) == 0


def cdc_passage(text: str, parser: PassageParser) -> str:
    """The 2013-19 sheets' passage notation in the form the parser reads. A comma joins
    steps ("CX,C4/C2" is written "CXC4/C2" in CDC's season files), and '+' adds passages, a
    bare count repeating the previous step ("E3+3/E2" is E3/E3/E2, as ae read it)."""
    text = re.sub(r"(?<=[A-Z0-9?])\s*,\s*(?=[A-Z]+\d|[A-Z]+X\b)", "", text, flags=re.IGNORECASE)
    if "+" not in text:
        return text
    out: list[str] = []
    for part in re.split(r"\s*[/+]\s*", text):
        if re.fullmatch(r"\d+", part):
            if not out or (name := parser.last_step_name(out[-1])) is None:
                raise ValueError(f"passage {text!r}: nothing before {part!r} to repeat")
            part = name + part
        out.append(part)
    return "/".join(out)


def _next_letter(letters: str) -> str:
    """CDC's serum letters: A ... Z, then doubled, AA, BB, CC."""
    if letters == "Z":
        return "AA"
    return chr(ord(letters[0]) + 1) * len(letters)


def read(paths: list[Path], rules: Rules) -> ReadResult:
    result = ReadResult(tables=[], rows=0)
    for path in paths:
        data = path.read_bytes()
        provenance = {
            "file": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "reader": "af.tables.cdc_xlsx",
        }
        sheets = load_or_error(path, result.errors)
        if sheets is None:
            continue
        for sheet in sheets:
            if not sheet.find(REFERENCE):
                result.skipped_tests.append(
                    f"{sheet.where(0)}: no REFERENCE VIRUSES label, not a titre sheet"
                )
                continue
            try:
                table = SheetReader(sheet, rules).table()
            except (SheetError, dates.DateError, ValueError) as err:
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
    def __init__(self, sheet: Sheet, rules: Rules):
        self.s = sheet
        self.rules = rules
        self.passages = PassageParser(rules.passage_tokens, LAB)
        # hand repairs to single cells, named and checked (af.tables.sheet.apply_cell_fixes)
        self.warnings: list[str] = apply_cell_fixes(sheet, rules.cell_fixes, LAB)
        self.dropped: Counter[str] = Counter()
        order = rules.lab_conventions.lookup(lab=LAB)
        if order is None:
            raise CDCSheetError(f"no lab_conventions rule for {LAB}")
        self.date_order = order["date_order"]

    def fail(self, r: int, c: int | None, message: str) -> CDCSheetError:
        return CDCSheetError(f"{self.s.where(r, c)}: {message}")

    def table(self) -> Table:
        subtype, lineage, prefix, protocol = self._title()
        test_date = self._test_date()
        rbc = self._rbc()
        group = "-".join((prefix, "hi", rbc, LAB.lower()))
        ref_row, _ = self.s.find_one(REFERENCE)
        columns = self._serum_columns(ref_row)
        label_row = self._label_row(ref_row)
        sera = self._sera(columns, subtype, test_date)
        last_letter = max(columns.values())
        columns = {letter: columns[letter] for letter, _ in sera}  # control sera removed
        antigens, titres = self._antigens(label_row, columns, last_letter, subtype, test_date)
        if lineage:
            # The sheet states the lineage once, in its title ("TYPE B VICTORIA LINEAGE"), and
            # every point on it carries that lineage, as in the other labs' readers. Counted
            # in the run report; a point of another lineage is a control, removed by rule.
            for antigen in antigens:
                antigen.lineage = lineage
            for _, serum in sera:
                serum.lineage = lineage
            self.warnings.append(
                f"{self.s.where(0)}: lineage {lineage} from the sheet title for "
                f"{len(antigens)} antigens and {len(sera)} sera"
            )
        return Table(
            table_id="",
            group=group,
            lab=LAB,
            subtype=subtype,
            lineage=lineage,
            assay="HI",
            rbc=rbc,
            date=test_date.isoformat(),
            date_suffix=0,
            source_key=f"CDC xlsx {self.s.path.name} [{self.s.name}]",
            antigens=antigens,
            sera=[sr for _, sr in sera],
            titres=titres,
            meta={"file": self.s.path.name, "sheet": self.s.name, "test_protocol": protocol},
            dropped=dict(sorted((k, v) for k, v in self.dropped.items() if v)),
            warnings=sorted(set(self.warnings)),
        )

    # -- table fields ---------------------------------------------------------------------

    def _title(self) -> tuple[str, str, str, str]:
        r, c = self.s.find_one(TITLE)
        raw = re.fullmatch(TITLE, self.s.cell(r, c), re.IGNORECASE)[1].upper()  # type: ignore[index]
        if raw not in TITLE_SUBTYPES:
            raise self.fail(r, c, f"unknown subtype {raw!r} in title")
        oseltamivir = bool(self.s.find(r".*OSELTAMIVIR.*", stop=r + 3))
        subtype, lineage = TITLE_SUBTYPES[raw]
        protocol = "hi_oseltamivir_protocol" if oseltamivir else "hi_protocol"
        return subtype, lineage, subtypes().group_prefix(subtype, lineage), protocol

    def _test_date(self) -> dt.date:
        found = set()
        # "DATE TESTED: 9/12/2024"; the 2013-19 sheets write "TESTED 2/11/2014"
        for r, c in self.s.find(TESTED):
            m = re.fullmatch(TESTED, self.s.cell(r, c), re.IGNORECASE)
            assert m is not None  # find() matched the same pattern
            text = m[1] or m[2]
            iso, warning = dates.parse(text, self.date_order)
            if warning:
                self.warnings.append(f"{self.s.where(r, c)}: {warning}")
            found.add(iso)
        if len(found) != 1:
            raise self.fail(0, None, f"expected one test date, found {sorted(found)}")
        return dt.date.fromisoformat(found.pop())

    def _rbc(self) -> str:
        r, c = self.s.find_one(r"RBC'?S USED")
        value = self.s.cell(r + 1, c).upper()
        if value not in RBC:
            raise self.fail(r + 1, c, f"RBC {value!r} not one of {sorted(RBC)}")
        return RBC[value]

    # -- layout ---------------------------------------------------------------------------

    def _serum_columns(self, ref_row: int) -> dict[str, int]:
        """Letter -> column, from the nearest letters row at or above REFERENCE VIRUSES: the
        first run of consecutive letters. The 2013-18 sheets continue a sequence across tests
        (M, N ... Z, AA, BB), so the run may start at any letter."""
        for r in range(ref_row, max(ref_row - 6, -1), -1):
            row = self.s.rows[r]
            for c0, first in enumerate(row):
                if not LETTER.fullmatch(first):
                    continue
                letters = {}
                expected = first
                for c in range(c0, len(row)):
                    if row[c] != expected:
                        break
                    letters[expected] = c
                    expected = _next_letter(expected)
                if len(letters) >= 2:
                    return letters
        raise self.fail(ref_row, None, "no serum letters row (A, B, C ...) above REFERENCE VIRUSES")

    def _label_row(self, ref_row: int) -> int:
        # the 2013-19 sheets put the serum abbreviations between the two rows
        for r in (ref_row, ref_row + 1, ref_row + 2):
            if any(v.upper() == "CDC ID#" for v in self.s.rows[r]):
                return r
        raise self.fail(
            ref_row, None, "no 'CDC ID#' label in the REFERENCE VIRUSES row or the one below"
        )

    def _label_columns(self, label_row: int, pattern: str) -> list[int]:
        rx = re.compile(pattern, re.IGNORECASE)
        above = self.s.rows[label_row - 1] if label_row else []
        out = []
        for c, v in enumerate(self.s.rows[label_row]):
            joined = f"{above[c] if c < len(above) else ''} {v}".strip()
            if rx.fullmatch(v) or rx.fullmatch(joined):
                out.append(c)
        return out

    # -- antigens -------------------------------------------------------------------------

    def _antigens(
        self,
        label_row: int,
        columns: dict[str, int],
        last_letter: int,
        subtype: str,
        test_date: dt.date,
    ) -> tuple[list[Antigen], list[list[list[str]]]]:
        titre_cols = list(columns.values())
        first_titre = min(titre_cols)
        end = self._block_end(label_row + 1, titre_cols)
        id_col = self._one_label(label_row, r"CDC ID#")
        date_col = self._one_label(label_row, r"(DATE )?COLLECTED|DATE COLLECTED")
        passage_cols = self._label_columns(label_row, r"PASSAGE")
        if not passage_cols:
            raise self.fail(label_row, None, "no PASSAGE column")
        name_col = self._name_column(label_row + 1, end, first_titre)
        self._no_unlettered_titres(label_row + 1, end, last_letter, passage_cols)
        antigens, titres = [], []
        for r in range(label_row + 1, end):
            name_raw = self.s.cell(r, name_col)
            cells = [self.s.cell(r, c) for c in titre_cols]
            if not name_raw and not any(cells):
                continue
            if SECTION.fullmatch(name_raw):
                continue
            if SECTION.fullmatch(self.s.cell(r, 0)) and not name_raw:
                # a section label; the 2016 sheets write notes on it ("RECEIVED AS ...")
                if notes := [v for v in self.s.rows[r][1:] if v]:
                    self.warnings.append(f"{self.s.where(r)}: note on the section row: {notes}")
                continue
            if self.rules.control_antigens.find(name_raw, lab=LAB) is not None:
                self.dropped["antigens: control"] += 1
                continue
            problems: list[str] = []
            name, renamed = aliases.parse_name(
                self.rules,
                name_raw,
                lab=LAB,
                subtype=subtype,
                applies_to="antigen",
                warnings=problems,
                not_after=test_date.year,
            )
            self.warnings.extend(f"{self.s.where(r, name_col)}: {p}" for p in problems)
            passage, harvest, annotations, site = self._passage_cell(r, passage_cols, test_date)
            collected = self._collected(r, date_col, test_date)
            antigens.append(
                Antigen(
                    name=name.name,
                    raw_name=name_raw,
                    passage=passage,
                    passage_class=self.passages.passage_class(passage),
                    passage_date=harvest,
                    date=collected,
                    lab_ids=[f"CDC#{self.s.cell(r, id_col)}"] if self.s.cell(r, id_col) else [],
                    reassortant=name.reassortant,
                    annotations=name.annotations + annotations,
                    source={
                        "row": r + 1,
                        "passage_cell": self.s.cell(r, passage_cols[0]),
                        "site": site,
                    },
                )
            )
            if renamed:
                antigens[-1].source[aliases.SOURCE_KEY] = renamed
            titres.append([self._titre(r, c) for c in titre_cols])
        antigens, titres = self._merge_repeats(antigens, titres)
        keep = [i for i, row in enumerate(titres) if any(row)]
        self.dropped["antigens: no readings"] += len(titres) - len(keep)
        return [antigens[i] for i in keep], [titres[i] for i in keep]

    def _collected(self, r: int, c: int, test_date: dt.date) -> str | None:
        """The collection date; one after the test date is a typing error (fix the cell)."""
        if not (text := self.s.cell(r, c)):
            return None
        try:
            iso, warning = dates.parse(text, self.date_order, not_after=test_date)
        except dates.DateError as err:
            raise self.fail(r, c, str(err)) from None
        if warning:
            self.warnings.append(f"{self.s.where(r, c)}: {warning}")
        if dt.date.fromisoformat(iso) > test_date:
            raise self.fail(r, c, f"collection date {iso} is after the test date {test_date}")
        return iso

    def _no_unlettered_titres(self, start: int, end: int, last: int, passages: list[int]) -> None:
        """Titres right of the last lettered column, before the PASSAGE column, belong to a
        serum with no letter (2017-03-21 lists two sera so): an error, never a silent loss."""
        right = [c for c in passages if c > last]
        for r in range(start, end):
            for c in range(last + 1, min(right) if right else last + 1):
                if TITRE_LIKE.fullmatch(self.s.cell(r, c)):
                    raise self.fail(r, c, "a titre in a column with no serum letter")

    def _merge_repeats(
        self, antigens: list[Antigen], titres: list[list[list[str]]]
    ) -> tuple[list[Antigen], list[list[list[str]]]]:
        """A sheet that lists one preparation on several rows (same name, passage cell,
        harvest date and CDC id) gives one antigen with every row's readings, as the TSV reader
        does for repeated rows of a test: both routes into a CDC table must agree, and a
        chart cannot hold two points with one identity."""
        first: dict[tuple[str, ...], int] = {}
        out_ag: list[Antigen] = []
        out_ti: list[list[list[str]]] = []
        for antigen, row in zip(antigens, titres, strict=True):
            key = (
                antigen.raw_name,
                antigen.source["passage_cell"],
                antigen.passage_date or "",
                ",".join(antigen.lab_ids),
            )
            if key not in first:
                first[key] = len(out_ag)
                out_ag.append(antigen)
                out_ti.append([list(cell) for cell in row])
                continue
            i = first[key]
            for cell, more in zip(out_ti[i], row, strict=True):
                cell.extend(more)
                cell.sort(key=_titre_order)
            out_ag[i].source.setdefault("repeated_rows", []).append(antigen.source["row"])
            self.dropped["antigens: repeated rows merged"] += 1
            self.warnings.append(
                f"{self.s.where(antigen.source['row'] - 1)}: {antigen.raw_name!r} repeats row "
                f"{out_ag[i].source['row']}; readings merged"
            )
        return out_ag, out_ti

    def _block_end(self, start: int, titre_cols: list[int]) -> int:
        """The antigen block ends at the SERUM CONTROL row; failing that, at the first blank row
        (blank rows can separate the reference and test viruses, so SERUM CONTROL wins)."""
        if hits := self.s.find(r"SERUM CONTROL", start=start):
            return hits[0][0]
        for r in range(start, len(self.s.rows)):
            if not any(self.s.rows[r][: max(titre_cols) + 1]):
                return r
        raise self.fail(start, None, "the antigen block has no end")

    def _one_label(self, label_row: int, pattern: str) -> int:
        cols = self._label_columns(label_row, pattern)
        if len(cols) != 1:
            raise self.fail(label_row, None, f"expected one {pattern!r} column, found {len(cols)}")
        return cols[0]

    def _name_column(self, start: int, stop: int, before: int) -> int:
        """The column left of the titres where most cells look like strain names."""
        counts = Counter(
            c
            for r in range(start, stop)
            for c in range(before)
            if re.match(r"[AB]/", self.s.cell(r, c), re.IGNORECASE)
        )
        if not counts:
            raise self.fail(start, None, "no column of strain names left of the titres")
        return counts.most_common(1)[0][0]

    def _passage_cell(
        self, r: int, cols: list[int], test_date: dt.date
    ) -> tuple[str, str | None, list[str], str]:
        """Some sheets repeat PASSAGE right of the titres; the copies must agree (the site tag
        aside)."""
        read = [
            self.passage_text(self.s.cell(r, c), r, c, test_date) for c in cols if self.s.cell(r, c)
        ]
        if not read:
            return "", None, [], ""
        if len({(x[0], x[1], tuple(x[2])) for x in read}) != 1:
            raise self.fail(
                r, cols[0], f"the PASSAGE columns disagree: {[self.s.cell(r, c) for c in cols]}"
            )
        if len({x[3] for x in read}) != 1:
            self.warnings.append(
                f"{self.s.where(r, cols[0])}: PASSAGE copies differ in the site tag only"
            )
        return read[0][:3] + (max(x[3] for x in read),)

    def passage_text(
        self, text: str, r: int, c: int, test_date: dt.date
    ) -> tuple[str, str | None, list[str], str]:
        """ "S1(07/19/2024)<NY>" -> ("SIAT1", "2024-07-19", [], "NY")."""
        m = PASSAGE_CELL.fullmatch(text)
        assert m is not None  # every group is optional
        try:
            passage = self.passages.parse(cdc_passage(m["passage"].strip(), self.passages))
        except ValueError as err:
            raise self.fail(r, c, str(err)) from None
        self.warnings.extend(f"{self.s.where(r, c)}: {p}" for p in passage.problems)
        harvest = None
        if m["date"] is not None and not m["close"]:
            self.warnings.append(
                f"{self.s.where(r, c)}: passage {text!r} has no closing parenthesis"
            )
        if m["date"]:
            try:
                harvest, warning = dates.parse(m["date"], self.date_order, not_after=test_date)
            except dates.DateError as err:
                raise self.fail(r, c, f"passage {text!r}: {err}") from None
            if warning:
                self.warnings.append(f"{self.s.where(r, c)}: {warning}")
            if harvest is not None and harvest > test_date.isoformat():
                # after the test: a typo, and nothing says which way (2015-12-03 "11/15/16")
                self.warnings.append(
                    f"{self.s.where(r, c)}: date {m['date']!r} is after the test; left out"
                )
                harvest = None
        extra, annotations, site = m["extra"].strip(), [], ""
        if (s := SITE.fullmatch(extra)) is not None:
            site = s[1]
        elif LOT.fullmatch(extra):
            annotations.append(re.sub(r"\s+", "", extra.upper()))
        elif extra:
            raise self.fail(r, c, f"passage {text!r}: cannot read {extra!r} after the passage")
        return passage.text, harvest, annotations, site

    def _titre(self, r: int, c: int) -> list[str]:
        raw = self.s.cell(r, c)
        if not raw:
            self.dropped["cells: blank"] += 1
            return []
        if (rule := self.rules.titre_tokens.find(raw, lab=LAB, assay="HI")) is not None:
            return [] if rule["titre"] == "*" else [rule["titre"]]
        if re.fullmatch(r"[<>]?[1-9]\d*", raw) and (raw[0] in "<>" or int(raw) >= 10):
            if not _is_dilution(int(raw.lstrip("<>"))):
                raise self.fail(r, c, f"titre {raw!r} is not a dilution (10 x 2^k): a typing error")
            return [raw]
        raise self.fail(r, c, f"titre {raw!r} matches no titre_tokens rule")

    # -- sera -----------------------------------------------------------------------------

    def _concentration(self, r: int, labels: dict[str, int]) -> list[str]:
        """The 2016-18 sheets' CONC. column ("2:1"): a concentrated serum is another
        preparation of the lot, so an annotation (identity), as ae read it."""
        if "CONC." not in labels or not (value := self.s.cell(r, labels["CONC."])):
            return []
        return [f"CONC {value}"]

    def _control_serum(self, **fields: str) -> Rule | None:
        """The first control_sera rule naming the serum by any of its fields (lot, name,
        species): the 2013-19 sheets list WHO kit sheep sera, goat plasmas and normal sera
        beside the ferret antisera."""
        for field, text in fields.items():
            if (rule := self.rules.control_sera.find(text, lab=LAB, field=field)) is not None:
                return rule
        return None

    def _sera(
        self, columns: dict[str, int], subtype: str, test_date: dt.date
    ) -> list[tuple[str, Serum]]:
        r0, _ = self.s.find_one(r"REFERENCE ANTISER(A|UM)")
        # "LOT" since 2024, "LOT #" in the 2013-19 sheets
        labels = {re.sub(r"\s*#$", "", v.upper()): c for c, v in enumerate(self.s.rows[r0]) if v}
        for need in ("LOT", "PASSAGE", "BOOSTED", "SPECIES"):
            if need not in labels:
                raise self.fail(r0, None, f"no {need} column in the antisera block")
        by_letter: dict[str, Serum] = {}
        r = r0 + 1
        while LETTER.fullmatch(self.s.cell(r, 0)):
            letter = self.s.cell(r, 0)
            raw = self.s.cell(r, 1)
            lot = self.s.cell(r, labels["LOT"])
            species = self.s.cell(r, labels["SPECIES"]).upper()
            rule = self._control_serum(lot=lot, name=raw, species=species)
            if rule is not None and rule["action"] == "drop":
                self.dropped["sera: control"] += 1
                by_letter[letter] = None  # type: ignore[assignment]
                r += 1
                continue
            problems: list[str] = []
            name, serum_renamed = aliases.parse_name(
                self.rules,
                raw,
                lab=LAB,
                subtype=subtype,
                applies_to="serum",
                warnings=problems,
                not_after=test_date.year,
            )
            self.warnings.extend(f"{self.s.where(r, 1)}: {p}" for p in problems)
            passage, harvest, annotations, _ = self.passage_text(
                self.s.cell(r, labels["PASSAGE"]), r, labels["PASSAGE"], test_date
            )
            boosted_cell = self.s.cell(r, labels["BOOSTED"])
            if (boosted := BOOSTED.get(re.sub(r"[\s-]+", " ", boosted_cell.upper()))) is None:
                raise self.fail(r, labels["BOOSTED"], f"BOOSTED {boosted_cell!r}")
            serum = Serum(
                name=name.name,
                raw_name=raw,
                serum_id=f"CDC {_expand_lots(lot)}" if lot else "",  # no lot: no identity
                passage=passage,
                passage_class=self.passages.passage_class(passage),
                passage_date=harvest,
                species="" if species == "FERRET" else species,
                reassortant=name.reassortant,
                annotations=name.annotations
                + annotations
                + (["BOOSTED"] if boosted else [])
                + self._concentration(r, labels),
                source={
                    "row": r + 1,
                    "letter": letter,
                    "lot": lot,
                    "pool": self.s.cell(r, labels["POOL"]) if "POOL" in labels else "",
                },
            )
            if serum_renamed:
                serum.source[aliases.SOURCE_KEY] = serum_renamed
            if boosted_cell.upper() not in ("Y", "N", ""):
                # the 2013-19 words, e.g. a pre-boost bleed read as not boosted, stay visible
                serum.source["boosted"] = boosted_cell
            if letter in by_letter:
                raise self.fail(r, 0, f"serum letter {letter} twice")
            by_letter[letter] = serum
            r += 1
        if set(by_letter) != set(columns):
            raise self.fail(
                r0, None, f"serum letters {sorted(by_letter)} != titre columns {sorted(columns)}"
            )
        out = []
        for letter in sorted(columns, key=columns.get):  # type: ignore[arg-type]
            serum = by_letter[letter]
            if serum is None:
                continue
            rule = self.rules.control_sera.find(serum.source["lot"], lab=LAB, field="lot")
            if rule is not None and rule["action"] == "species":
                serum.species = rule["value"]
            out.append((letter, serum))
        return out
