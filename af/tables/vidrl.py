"""Read VIDRL's HI and FRA workbooks: one table per workbook, sera described in rows above.

Layout (HI since at least 2019, FRA since 2022; FRA 2019-22 lacks some labels):

- ``Test Date: 16/09/2026`` (D/M/Y; older FRA sheets have a bare date cell), ``RBC Type:
  Turkey`` on HI sheets, ``Focus Reduction Assay`` on FRA sheets;
- the serum index row ``1 2 ... n`` over the serum columns (the anchor: the ``Reference
  Antisera`` label above it is missing on some sheets), then one row each of serum ids,
  serum passages and serum names, then clade rows;
- antigen rows: ``[index] | name | [clade] | titres ... | passage | sample date | ID # |
  comments``; the right-hand columns are found by their labels in the header rows.

The sheet names neither the flu type nor the lineage, so they come from the folder's config
entry (``subtype``, ``lineage``).

**Serum names are abbreviations** (``Exa/1234``, ``EXC07``, ``Exa City 272``): a serum is
the one antigen of the table whose isolate is the abbreviation's number and whose location
starts with its letters (spaces and hyphens ignored); a reassortant abbreviation
(``IVR-99``) is the antigen with that reassortant. No such antigen, or two different names,
is an error unless a ``strain_aliases`` rule (applies_to serum) names the virus. ae took the
first match and needed hand rules for the rest.

Checks: the serum index row is exactly ``1..n``; a workbook's sheets other than the one
whose test date is the file's are reported, not read (VIDRL keeps copies of earlier tests
in extra sheets); every row with titres has a strain name; titres are valid; control sera
(human pools) are dropped by ``control_sera`` rules on the serum id or name.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import aliases, dates
from .cdc import ReadResult
from .model import Antigen, Serum, Table
from .passage import PassageParser
from .rules import Rules
from .sheet import Sheet, SheetError, apply_cell_fixes, load

TEST_DATE = re.compile(r"\s*Test\s+Date\s*:\s*(.*)", re.IGNORECASE)
RBC_TYPE = re.compile(r"\s*RBC\s+Type\s*:\s*([A-Za-z ]+?)\s*(#\s*\d+)?\s*", re.IGNORECASE)
FRA = re.compile(
    r"\s*Focus\s+Reduction\s+Assay\s*(?P<date>\d{1,2}\s+[A-Za-z]+\s+\d{4})?\s*", re.IGNORECASE
)
RBC = {"TURKEY": "turkey", "GUINEA PIG": "guinea-pig", "CHICKEN": "chicken"}
TITRE = re.compile(r"[<>]?[1-9]\d*")
LABELS = {
    "passage": re.compile(r"Passage(\s+details)?", re.IGNORECASE),
    "date": re.compile(r"Sample(\s+date)?", re.IGNORECASE),
    "id": re.compile(r"ID\s*#|VW", re.IGNORECASE),
}
LAB_ID = re.compile(r"[A-Z]{1,3}\d{8}")  # VIDRL sample ids: two or three letters and eight digits
DATE_CELL = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}")
REASSORTANT_ONLY = re.compile(r"[A-Z]+-\d+[A-Z]?", re.IGNORECASE)


class VIDRLError(SheetError):
    pass


def read(
    paths: list[Path], rules: Rules, *, lab: str, subtype: str, lineage: str = ""
) -> ReadResult:
    """``subtype`` ("A(H1N1)", "A(H3N2)", "B") and ``lineage`` come from config: the sheets
    do not say."""
    result = ReadResult(tables=[], rows=0)
    for path in paths:
        data = path.read_bytes()
        provenance = {
            "file": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "reader": "af.tables.vidrl",
        }
        try:
            sheets = load(path)
        except Exception as err:  # openpyxl raises zip and XML errors of many kinds
            result.errors.append(f"{path.name}: not a readable workbook ({type(err).__name__})")
            continue
        file_date = _file_date(path)
        read_here, sheet_errors = 0, []
        for sheet in sheets:
            if not any(any(row) for row in sheet.rows):
                continue
            try:
                reader = SheetReader(sheet, rules, lab, subtype, lineage)
                if reader.test_date.isoformat() != file_date:
                    result.skipped_tests.append(
                        f"{sheet.where(0)}: test date {reader.test_date} is not the file's"
                        f" ({file_date}): another test's sheet, not read"
                    )
                    continue
                table = reader.table()
            except (SheetError, dates.DateError, LookupError, ValueError) as err:
                sheet_errors.append(str(err))
                continue
            read_here += 1
            table.provenance = dict(provenance)
            result.dropped.update(table.dropped)
            result.rows += len(table.antigens)
            if problems := table.check() or aliases.check_titres(table, rules):
                result.errors.extend(f"{sheet.where(0)}: {p}" for p in problems)
            result.tables.append(table)
        if read_here == 1:
            # VIDRL keeps comparisons and old tests in extra sheets: not tables of this file
            result.skipped_tests.extend(
                f"{e} (another sheet was this file's table)" for e in sheet_errors
            )
        else:
            result.errors.extend(sheet_errors)
            if not sheet_errors:
                result.errors.append(f"{path.name}: {read_here} sheets with the file's test date")
    return result


@dataclass
class Header:
    index_row: int
    serum_cols: list[int]
    first_antigen_row: int


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
            raise VIDRLError(f"no lab_conventions rule for {lab}")
        self.date_order = convention["date_order"]
        self.warnings: list[str] = apply_cell_fixes(sheet, rules.cell_fixes, lab)
        self.dropped: Counter[str] = Counter()
        self.header = self._header()
        self.test_date = self._test_date()

    def fail(self, r: int, c: int | None, message: str) -> VIDRLError:
        return VIDRLError(f"{self.s.where(r, c)}: {message}")

    # -- layout ---------------------------------------------------------------------------

    def _header(self) -> Header:
        for r, row in enumerate(self.s.rows[:25]):
            cols = [c for c, v in enumerate(row) if re.fullmatch(r"\d+", v)]
            if len(cols) >= 3 and row[cols[0]] == "1":
                numbers = [int(row[c]) for c in cols]
                if len(set(numbers)) != len(numbers):
                    raise self.fail(r, cols[0], f"serum index row repeats a number: {numbers}")
                if cols != list(range(cols[0], cols[0] + len(cols))):
                    raise self.fail(r, cols[0], "serum index row has gaps")
                first = next(
                    (
                        a
                        for a in range(r + 4, len(self.s.rows))
                        if self._has_titres(a, cols) and self._strain_left(a, cols[0])
                    ),
                    None,
                )
                if first is None:
                    raise self.fail(r, None, "no antigen rows under the serum index row")
                return Header(r, cols, first)
        raise self.fail(0, None, "no serum index row (1 2 ... n)")

    def _has_titres(self, r: int, cols: list[int]) -> bool:
        return any(TITRE.fullmatch(_clean_titre(self.s.cell(r, c))) for c in cols)

    def _strain_left(self, r: int, first_serum: int) -> bool:
        return any("/" in self.s.cell(r, c) for c in range(first_serum))

    def _test_date(self) -> dt.date:
        h = self.header.index_row
        labelled = [
            (r, c, m[1])
            for r in range(h)
            for c, v in enumerate(self.s.rows[r])
            if (m := TEST_DATE.fullmatch(v))
        ]
        if len(labelled) > 1:
            raise self.fail(0, None, f"{len(labelled)} 'Test Date' cells")
        if labelled:
            r, c, text = labelled[0]
        elif titled := [
            (r, c, m["date"])
            for r in range(h)
            for c, v in enumerate(self.s.rows[r])
            if (m := FRA.fullmatch(v)) and m["date"]
        ]:
            r, c, text = titled[0]
            try:
                return dt.datetime.strptime(re.sub(r"\s+", " ", text), "%d %B %Y").date()
            except ValueError as err:
                raise self.fail(r, c, f"test date {text!r} in the title") from err
        else:  # older FRA sheets: a bare date cell above the sera
            bare = [
                (r, c, v)
                for r in range(h)
                for c, v in enumerate(self.s.rows[r])
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}", v)
            ]
            if len(bare) != 1:
                raise self.fail(0, None, f"no 'Test Date:' and {len(bare)} bare date cells")
            r, c, text = bare[0]
            self.warnings.append(f"{self.s.where(r, c)}: test date from an unlabelled cell")
        day, warning = dates.parse(text, self.date_order)
        if warning:
            self.warnings.append(f"{self.s.where(r, c)}: {warning}")
        return dt.date.fromisoformat(day)

    def _assay(self) -> tuple[str, str]:
        h = self.header.index_row
        top = [(r, c, v) for r in range(h) for c, v in enumerate(self.s.rows[r]) if v]
        if any(FRA.fullmatch(v) for _, _, v in top):
            return "FRA", ""
        rbc = [(r, c, m[1]) for r, c, v in top if (m := RBC_TYPE.fullmatch(v))]
        if len(rbc) != 1:
            raise self.fail(
                0, None, f"neither 'Focus Reduction Assay' nor one 'RBC Type:' ({len(rbc)})"
            )
        r, c, text = rbc[0]
        species = RBC.get(text.strip().upper())
        if species is None:
            raise self.fail(r, c, f"RBC type {text!r} not known")
        return "HI", species

    def _right_columns(self) -> tuple[int | None, int | None, int | None]:
        """(passage, date, ID) columns right of the sera, found by what they hold: VIDRL's
        labels over them are shifted or missing on some sheets, and appear inside the data
        rows on others. A label decides only between columns whose contents qualify."""
        h = self.header
        width = max(len(r) for r in self.s.rows)
        rows = range(h.first_antigen_row, len(self.s.rows))
        scores = {}
        for c in range(h.serum_cols[-1] + 1, width):
            values = [v for r in rows if (v := self.s.cell(r, c)) and not _is_label(v)]
            if not values:
                continue
            half = len(values) / 2
            scores[c] = {
                "id": sum(bool(LAB_ID.fullmatch(v)) for v in values) > half,
                "date": sum(bool(DATE_CELL.fullmatch(v)) for v in values) > half,
                "passage": sum(self._reads_as_passage(v) for v in values) > half,
            }
        found: dict[str, int | None] = {}
        for key in ("id", "date", "passage"):
            cols = [c for c, sc in scores.items() if sc[key] and c not in found.values()]
            if len(cols) > 1:
                labelled = [c for c in cols if self._labelled(c, key)]
                cols = labelled if len(labelled) == 1 else cols[:1]
            found[key] = cols[0] if cols else None
        return found["passage"], found["date"], found["id"]

    def _labelled(self, c: int, key: str) -> bool:
        return any(
            LABELS[key].fullmatch(self.s.cell(r, c))
            for r in range(self.header.index_row, self.header.first_antigen_row)
        )

    def _reads_as_passage(self, text: str) -> bool:
        if not re.search(r"\d", text) or "/" in text and text.count("/") >= 3:
            return False
        try:
            return not self.passages.parse(_passage_text(text, self.passages)).problems
        except ValueError:
            return False

    def _name_column(self) -> int:
        h = self.header
        counts = Counter(
            c
            for r in range(h.first_antigen_row, len(self.s.rows))
            for c in range(h.serum_cols[0])
            if "/" in self.s.cell(r, c)
        )
        if not counts:
            raise self.fail(h.first_antigen_row, None, "no column of strain names")
        return counts.most_common(1)[0][0]

    # -- the table ------------------------------------------------------------------------

    def table(self) -> Table:
        assay, rbc = self._assay()
        prefix = {"A(H1N1)": "h1pdm", "A(H3N2)": "h3"}.get(self.subtype) or {
            "VICTORIA": "bvic",
            "YAMAGATA": "byam",
        }.get(self.lineage, "b")
        group = "-".join(p for p in (prefix, assay.lower(), rbc, self.lab.lower()) if p)
        serum_cols, sera_raw = self._sera()
        antigens, titres = self._antigens(serum_cols)
        sera = [self._serum(c, raw, antigens) for c, raw in zip(serum_cols, sera_raw, strict=True)]
        return Table(
            table_id="",
            group=group,
            lab=self.lab,
            subtype=self.subtype,
            lineage=self.lineage,
            assay=assay,
            rbc=rbc,
            date=self.test_date.isoformat(),
            date_suffix=0,
            source_key=f"{self.lab} xlsx {self.s.path.name} [{self.s.name}]",
            antigens=antigens,
            sera=sera,
            titres=titres,
            meta={"file": self.s.path.name, "sheet": self.s.name},
            dropped=dict(sorted((k, v) for k, v in self.dropped.items() if v)),
            warnings=sorted(set(self.warnings)),
        )

    # -- sera -----------------------------------------------------------------------------

    def _sera(self) -> tuple[list[int], list[tuple[str, str, str]]]:
        """(column, (id, passage, abbreviated name)) of the sera kept; human pools dropped."""
        h = self.header.index_row
        cols, raw = [], []
        for c in self.header.serum_cols:
            serum_id, passage, name = (self.s.cell(h + k, c) for k in (1, 2, 3))
            # VIDRL writes a pool's marks in any of the three rows ("SH 2029" as the name,
            # "sera pool" as the id, "post vax" as the passage)
            if any(
                self.rules.control_sera.find(v, lab=self.lab, field=f) is not None
                for f in ("id", "name", "passage")
                for v in (serum_id, name, passage)
            ):
                self.dropped["sera: control"] += 1
                continue
            if not name:
                raise self.fail(h + 3, c, f"serum column without a name (id {serum_id!r})")
            if not serum_id:
                self.warnings.append(f"{self.s.where(h + 1, c)}: serum {name!r} has no id")
            cols.append(c)
            raw.append((serum_id, passage, name))
        if not cols:
            raise self.fail(h, None, "no sera")
        return cols, raw

    def _serum(self, c: int, raw: tuple[str, str, str], antigens: list[Antigen]) -> Serum:
        h = self.header.index_row
        serum_id, raw_passage, abbreviation = raw
        name = self._serum_name(h + 3, c, abbreviation, antigens)
        passage = self._passage(raw_passage, h + 2, c)
        return Serum(
            name=name.name,
            raw_name=abbreviation,
            serum_id=f"{self.lab} {serum_id}" if serum_id else "",
            passage=passage,
            passage_class=self.passages.passage_class(passage),
            reassortant=name.reassortant,
            annotations=name.annotations,
            lineage=self.lineage,
            source={"column": c + 1},
        )

    def _serum_name(self, r: int, c: int, abbreviation: str, antigens: list[Antigen]):
        """The antigen this abbreviation names (see the module docstring)."""
        problems: list[str] = []
        if abbreviation.count("/") >= 3:  # written in full
            name, _ = aliases.parse_name(
                self.rules,
                abbreviation,
                lab=self.lab,
                subtype=self.subtype,
                applies_to="serum",
                warnings=problems,
                not_after=self.test_date.year,
            )
            self.warnings.extend(f"{self.s.where(r, c)}: {p}" for p in problems)
            return name
        rule = self.rules.strain_aliases.find(
            abbreviation, lab=self.lab, subtype=self.subtype, applies_to="serum"
        )
        if rule is not None:
            name, _ = aliases.parse_name(
                self.rules,
                rule["canonical"],
                lab=self.lab,
                subtype=self.subtype,
                applies_to="serum",
                warnings=problems,
                not_after=self.test_date.year,
            )
            self.warnings.append(
                f"{self.s.where(r, c)}: serum {abbreviation!r} named by {rule.where}"
            )
            return name
        # the serum's virus is among the numbered reference antigens: look there first
        references = [a for a in antigens if a.reference]
        found = self._match_antigens(abbreviation, references) or self._match_antigens(
            abbreviation, antigens
        )
        if len(found) > 1:
            # VIDRL numbers a serum as its homologous reference antigen (serum 3 = antigen 3)
            index = int(self.s.cell(self.header.index_row, c))
            same = {k: v for k, v in found.items() if v.antigen.source.get("index") == index}
            if len(same) == 1:
                found = same
        if len(found) != 1:
            raise self.fail(
                r,
                c,
                f"serum {abbreviation!r} matches {len(found)} antigen names "
                f"{sorted(found)[:4]}: add a strain_aliases rule (applies_to serum)",
            )
        (key,) = found
        return found[key]

    def _match_antigens(self, abbreviation: str, antigens: list[Antigen]) -> dict:
        text = abbreviation.strip()
        out = {}
        if REASSORTANT_ONLY.fullmatch(text):
            from .names import split_extra

            wanted, _ = split_extra(text, self.rules.reassortants, self.lab)
            for a in antigens:
                if wanted and a.reassortant == wanted:
                    out[(a.name, a.reassortant)] = _Named(a)
            return out
        prefix, subsequence = {}, {}
        for loc_text, iso_text in _splits(text):
            loc = _squash(loc_text)
            iso = iso_text.upper().lstrip("0") or iso_text.upper()
            for a in antigens:
                parts = a.name.split("/")
                if len(parts) != 4 or (parts[2].upper().lstrip("0") or parts[2]) != iso:
                    continue
                location = _squash(parts[1])
                if location.startswith(loc):
                    prefix[(a.name, a.reassortant)] = _Named(a)
                elif location[:1] == loc[:1] and _in_order(loc, location):
                    subsequence[(a.name, a.reassortant)] = _Named(a)
        # "Exa" for EXAMPLEVILLE, "Sth Exa" for SOUTH EXAMPLELAND: the letters in order
        found = prefix or subsequence
        # an abbreviation names no reassortant: the plain virus, when the table has it too
        plain = {k: v for k, v in found.items() if not k[1]}
        return plain or found

    # -- antigens -------------------------------------------------------------------------

    def _antigens(self, serum_cols: list[int]) -> tuple[list[Antigen], list[list[list[str]]]]:
        name_col = self._name_column()
        passage_col, date_col, id_col = self._right_columns()
        antigens, titres = [], []
        for r in range(self.header.first_antigen_row, len(self.s.rows)):
            raw = self.s.cell(r, name_col)
            has_titres = self._has_titres(r, serum_cols)
            if not raw:
                if has_titres:
                    raise self.fail(r, name_col, "row with titres but no strain name")
                continue
            if not has_titres and "/" not in raw:
                continue  # a note or section label
            if self.rules.control_antigens.find(raw, lab=self.lab) is not None:
                self.dropped["antigens: control"] += 1
                continue
            problems: list[str] = []
            name, renamed = aliases.parse_name(
                self.rules,
                raw,
                lab=self.lab,
                subtype=self.subtype,
                applies_to="antigen",
                warnings=problems,
                not_after=self.test_date.year,
            )
            self.warnings.extend(f"{self.s.where(r, name_col)}: {p}" for p in problems)
            raw_passage = self.s.cell(r, passage_col) if passage_col is not None else ""
            passage = self._passage(raw_passage, r, passage_col)
            collected = self.s.cell(r, date_col) if date_col is not None else ""
            lab_id = self.s.cell(r, id_col) if id_col is not None else ""
            collected, lab_id = ("" if _is_label(v) else v for v in (collected, lab_id))
            if re.fullmatch(r"[\s/.-]*", collected):  # "//": a date left blank
                collected = ""
            antigen = Antigen(
                name=name.name,
                raw_name=raw,
                passage=passage,
                passage_class=self.passages.passage_class(passage),
                date=self._date(collected, r, date_col) if collected else None,
                lab_ids=[f"{self.lab}#{lab_id}"] if lab_id else [],
                reference=any(re.fullmatch(r"\d+", self.s.cell(r, c)) for c in range(name_col)),
                reassortant=name.reassortant,
                annotations=name.annotations,
                lineage=self.lineage,
                source={"row": r + 1, "passage": raw_passage},
            )
            left = [self.s.cell(r, c) for c in range(name_col)]
            if numbers := [v for v in left if re.fullmatch(r"\d+", v)]:
                antigen.source["index"] = int(numbers[0])
            if renamed:
                antigen.source[aliases.SOURCE_KEY] = renamed
            antigens.append(antigen)
            titres.append([self._titre(r, c) for c in serum_cols])
        keep = [i for i, row in enumerate(titres) if any(row)]
        self.dropped["antigens: no readings"] += len(titres) - len(keep)
        return [antigens[i] for i in keep], [titres[i] for i in keep]

    def _passage(self, raw: str, r: int, c: int | None) -> str:
        """VIDRL separates passage steps with ',' as well as '/' ("MDCK3, MDCK1") and puts
        a space before a count ("MDCK 1")."""
        try:
            read = self.passages.parse(_passage_text(raw, self.passages))
        except ValueError as err:
            raise self.fail(r, c, str(err)) from err
        where = self.s.where(r, c)
        self.warnings.extend(f"{where}: {p}" for p in read.problems)
        return read.text

    def _date(self, text: str, r: int, c: int | None) -> str | None:
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

    def _titre(self, r: int, c: int) -> list[str]:
        raw = self.s.cell(r, c)
        if not raw:
            self.dropped["cells: blank"] += 1
            return []
        if (rule := self.rules.titre_tokens.find(raw, lab=self.lab, assay="*")) is not None:
            return [] if rule["titre"] == "*" else [rule["titre"]]
        text = _clean_titre(raw)
        if TITRE.fullmatch(text) and _is_dilution(int(text.lstrip("<>"))):
            return [text]
        raise self.fail(
            r, c, f"titre {raw!r} is not a dilution (10, 20, 40...) nor a titre_tokens rule"
        )


@dataclass
class _Named:
    """An antigen's parsed name fields, as a serum takes them."""

    antigen: Antigen

    @property
    def name(self) -> str:
        return self.antigen.name

    @property
    def reassortant(self) -> str:
        return self.antigen.reassortant

    @property
    def annotations(self) -> list[str]:
        return list(self.antigen.annotations)


def _splits(text: str) -> list[tuple[str, str]]:
    """(location letters, isolate) readings of an abbreviation: at its last '/' when it has
    one ("Sing/WUH4618"), else before the trailing number, with or without capital letters
    that start the isolate ("ExaABC1234", "EXC07", "Exa City 272")."""
    text = re.sub(r"^[AB]/", "", text.strip(), flags=re.IGNORECASE)
    if "/" in text:
        loc, _, iso = text.rpartition("/")
        return [(loc, iso)] if loc and iso else []
    out = []
    if (m := re.fullmatch(r"(.*?)\s*([0-9][0-9A-Z-]*)", text)) and m[1]:
        out.append((m[1], m[2]))
    if (m := re.fullmatch(r"(.*?[a-z])([A-Z]+[0-9][0-9A-Z-]*)", text)) and m[1]:
        out.append((m[1], m[2]))
    return out


def _in_order(letters: str, word: str) -> bool:
    it = iter(word)
    return all(ch in it for ch in letters)


def _passage_text(raw: str, parser: PassageParser | None = None) -> str:
    """VIDRL's passage notation in the form the passage parser reads. VIDRL separates steps
    with ',' as well as '/' and '+' ("MDCK3, MDCK1", "C1+1"), writes counts after a space,
    hyphen or '#' ("MDCK 1", "MDCK-1", "MDCK#1") or before the name ("P1 SIAT"), marks QMC
    passages for HI ("QMC2-HI"), and writes a bare count for another passage of the previous
    step ("C2, 2") and a bare name for an unknown count ("X, SIAT1")."""
    text = raw.strip()
    text = re.sub(r"-HI\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bP(\d+)\s+([A-Za-z]+)", r"\2\1", text)
    text = re.sub(r"(?<=[A-Za-z])\s*[-#]?\s*(?=\d)", "", text)
    parts = [q.strip() for q in re.split(r"\s*[,/+]\s*|(?<=\d)\s+(?=[A-Za-z])", text) if q.strip()]
    if parser is None:
        return "/".join(parts)
    out: list[str] = []
    for part in parts:
        if re.fullmatch(r"\d+", part):
            name = parser.last_step_name(out[-1]) if out else None
            if name is None:
                raise ValueError(f"passage {raw!r}: nothing before {part!r} to repeat")
            part = name + part
        elif parser.is_step_name(part):
            part += "X"  # a step with no count: an unknown number of passages
        out.append(part)
    return "/".join(out)


def _is_dilution(n: int) -> bool:
    """10 x 2^k: a value off the series is a typing error (604 for 640, 2506 for 2560)."""
    return n >= 10 and n % 10 == 0 and (n // 10) & (n // 10 - 1) == 0


def _is_label(text: str) -> bool:
    return any(rx.fullmatch(text) for rx in LABELS.values()) or text.casefold() in (
        "comments",
        "sample date",
        "passage details",
    )


def _squash(text: str) -> str:
    return re.sub(r"[\s\-_.]", "", text.upper())


def _clean_titre(raw: str) -> str:
    return re.sub(r"\s+", "", raw)


def _file_date(path: Path) -> str | None:
    m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", path.name)
    if m is None:
        return None
    try:
        return dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat()
    except ValueError:
        return None
