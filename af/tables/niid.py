"""Read NIID's HI workbooks: one table per sheet, labelled columns, sera named in the header.

Layout (NIID has used it unchanged since at least 2016):

- a title row naming the virus type and the red cells (``... influenza B viruses (Victoria
  lineage)-0.5%CRBC``), and ``HI test date:<date>`` somewhere above the header;
- the header row: ``NIID-ID``, ``Strains``, ``Passage History``, ``Sample date``, then one
  column per serum, headed by the serum's description (``EXAMPLECITY /12/29 Cell No.101``),
  then annotation columns (``Subclade``, ``Amino acid substitution in HA``, ...);
- antigen rows under section labels (``REF. Ag.``, ``TEST Ag.``); notes below them.

The sheet names neither the lab nor the flu type in a field of its own, so the type comes
from the title through the ``flu_types`` rules (lab NIID, whole-title regex) and the red
cells from the title's ``<n>% <letter>RBC``.

Checks: every column right of ``Sample date`` is a serum (its header reads as one), a
control serum dropped by rule (``control_sera``, field ``name``), or an annotation column,
and an annotation column must not hold titres (that would be a serum whose header did not
read); every row with titres has an NIID-ID; the test date equals the date in the file name;
titres are valid.

NIID conventions handled here:

- a serum header is ``[clade] <name> [pdm] [reassortant] [passage] [Cell|Egg|hCK|Cell&Egg]
  [NIID|CDC] No.<n>`` (or ``#<n>``, ``NIID-<n>``), read from the right. The serum id is
  ``NIID <TYPE> NO.<n>`` as ae wrote it (``NIID EGG #43`` for a ``#`` number), the lab that
  numbered it is an annotation, and the cell/egg word gives the passage (``MDCK?``, ``E?``,
  ``HCK?``). The name gets the table's type and a two-digit year is read against the test
  year; headers always abbreviate the year, so that is not warned;
- passages write a space before the count and ``+n`` for n more of the previous step
  (``MDCK 2 +1`` = MDCK2/MDCK1; ``hCK +2`` = HCK?/HCK2);
- NIID-IDs are written with and without spaces (``29/30 - 1``, ``29/30-2``): spaces are
  removed;
- a row NIID has commented out (every cell starts with ``#``) is dropped and counted;
- a fullwidth ``＜`` and spaces inside a titre (``< 10``) are NIID's typing, not data.
"""

from __future__ import annotations

import contextlib
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
from .sheet import Sheet, SheetError, load

LAB_ID = r"\s*NIID-ID\s*"
COLUMNS = ("Strains", "Passage History", "Sample date")
SECTION = re.compile(r"(REF|TEST)\s*\.?\s*Ag\s*\.?", re.IGNORECASE)
TEST_DATE = re.compile(r"HI\s+test\s+date\s*:\s*(.+)", re.IGNORECASE)
RBC = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*([A-Z])RBC", re.IGNORECASE)
RBC_LETTER = {"C": "chicken", "T": "turkey", "G": "guinea-pig"}
TITRE = re.compile(r"[<>]?[1-9]\d*")

# serum header, read from the right
_SERUM_ID = re.compile(
    r"\s(?:No\s*\.\s*(?P<no>[\d][\d\-]*)|#(?P<hash>\d+)|NIID-\s*(?P<niid>[\d][\d\-#]*)"
    r"|(?P<bare>\d{4}-\d+))$",
    re.IGNORECASE,
)
_SERUM_LAB = re.compile(r"\s(NIID|CDC)$", re.IGNORECASE)
_SERUM_TYPE = re.compile(r"\s(Cell&Egg|Cell|Egg|hCK)$", re.IGNORECASE)
_SERUM_PASSAGE = re.compile(r"\s([A-Z]+\d+(?:/[A-Z]+\d+)+)$", re.IGNORECASE)
_SERUM_REASSORTANT = re.compile(r"\s\(?([A-Z]+-\d+[A-Z]?)\)?$", re.IGNORECASE)
_SERUM_PDM = re.compile(r"\spdm$", re.IGNORECASE)
_SERUM_CLADE = re.compile(r"^\d+[A-Z]\s+")
NO_YEAR = re.compile(r"[AB]/[^/]+/[^/]+/?", re.IGNORECASE)
_TYPE_PASSAGE = {"CELL": "MDCK?", "EGG": "E?", "HCK": "HCK?", "CELL&EGG": ""}


class NIIDError(SheetError):
    pass


@dataclass
class SerumHeader:
    name: str  # "EXAMPLECITY/12/29", as written, spaces around '/' and '-' removed
    serum_id: str  # "CELL NO.122"
    passage: str  # the lab's passage text, or ""
    type_word: str  # "CELL", "EGG", ... or ""
    reassortant: str  # as written, e.g. "IVR-99"
    lab: str  # "NIID", "CDC" or ""


def parse_serum_header(text: str) -> SerumHeader | None:
    """None when ``text`` does not end in a serum number (then it is not a serum column)."""
    rest = " " + re.sub(r"\s+", " ", text).strip()
    m = _SERUM_ID.search(rest)
    if m is None:
        return None
    rest = rest[: m.start()]
    fields: dict[str, str] = {}
    parts = {
        "lab": _SERUM_LAB,
        "type_word": _SERUM_TYPE,
        "passage": _SERUM_PASSAGE,
        "reassortant": _SERUM_REASSORTANT,
        "pdm": _SERUM_PDM,
    }
    # NIID writes these in more than one order ("Egg SAN-9 #43", "(IVR-99) Egg No.101"):
    # peel whichever is last, each at most once
    while found_any := [
        (k, hit) for k, rx in parts.items() if k not in fields and (hit := rx.search(rest))
    ]:
        key, found = found_any[0]
        fields[key] = found[1] if found.groups() else found[0]
        rest = rest[: found.start()]
    if m["niid"] and fields.get("lab", "").upper() == "NIID":
        return None  # "NIID NIID-..." would be a new layout: refuse it
    name = _SERUM_CLADE.sub("", rest.strip())
    name = re.sub(r"\s*([/-])\s*", r"\1", name)
    if "/" not in name:
        return None
    # spaces after the location are line wrapping in the header cell ("PUN-NIV 323546/21")
    location, _, after = name.partition("/")
    name = location + "/" + re.sub(r"\s+", "", after)
    type_word = fields.get("type_word", "").upper()
    number = f"#{m['hash']}" if m["hash"] else "NO." + (m["no"] or m["niid"] or m["bare"])
    serum_id = f"{type_word} {number}" if type_word else number
    return SerumHeader(
        name=name,
        serum_id=serum_id,
        passage=fields.get("passage", ""),
        type_word=type_word,
        reassortant=fields.get("reassortant", ""),
        lab=fields.get("lab", "").upper(),
    )


def niid_passage(text: str, parser: PassageParser) -> str:
    """NIID's passage notation in the form the passage parser reads: spaces removed, '+'
    as a step separator, a bare count repeats the previous step's name, and a step name
    with no count (``hCK +2``) gets an unknown count."""
    parts = [p for p in re.split(r"[/+]", re.sub(r"\s+", "", text)) if p]
    out: list[str] = []
    for i, part in enumerate(parts):
        if re.fullmatch(r"(\d+|X)(\(.*\))?", part, re.IGNORECASE):
            if not out or (name := parser.last_step_name(out[-1])) is None:
                raise ValueError(f"passage {text!r}: nothing before {part!r} to repeat")
            part = name + part
        elif parser.is_step_name(part) and i + 1 < len(parts):
            part += "X"
        out.append(part)
    return "/".join(out)


def read(paths: list[Path], rules: Rules, *, lab: str) -> ReadResult:
    result = ReadResult(tables=[], rows=0)
    for path in paths:
        data = path.read_bytes()
        provenance = {
            "file": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "reader": "af.tables.niid",
        }
        for sheet in load(path):
            if not sheet.find(LAB_ID, stop=15):
                result.skipped_tests.append(f"{sheet.where(0)}: no 'NIID-ID' header")
                continue
            try:
                table = SheetReader(sheet, rules, lab).table()
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
    def __init__(self, sheet: Sheet, rules: Rules, lab: str):
        self.s = sheet
        self.rules = rules
        self.lab = lab
        self.passages = PassageParser(rules.passage_tokens, lab)
        convention = rules.lab_conventions.lookup(lab=lab)
        if convention is None:
            raise NIIDError(f"no lab_conventions rule for {lab}")
        self.date_order = convention["date_order"]
        self.warnings: list[str] = []
        self.dropped: Counter[str] = Counter()
        self.no_year: dict[int, str] = {}  # serum column -> name written without a year

    def fail(self, r: int, c: int | None, message: str) -> NIIDError:
        return NIIDError(f"{self.s.where(r, c)}: {message}")

    def table(self) -> Table:
        h, id_col = self.s.find_one(LAB_ID, stop=15)
        header = self.s.rows[h]
        cols = {}
        for label in COLUMNS:
            found = [c for c, v in enumerate(header) if v.casefold() == label.casefold()]
            if len(found) != 1:
                raise self.fail(h, None, f"expected one {label!r} column, found {len(found)}")
            cols[label] = found[0]
        title = self._title(h)
        flu = self.rules.flu_types.find(title, lab=self.lab)
        if flu is None:
            raise self.fail(0, None, f"title {title!r} matches no flu_types rule")
        subtype, lineage = flu["subtype"], "" if flu["lineage"] == "-" else flu["lineage"]
        rbc = self._rbc(title)
        test_date = self._test_date(h)
        prefix = {"A(H1N1)": "h1pdm", "A(H3N2)": "h3"}.get(subtype) or {
            "VICTORIA": "bvic",
            "YAMAGATA": "byam",
        }.get(lineage, "b")
        group = "-".join((prefix, "hi", rbc, self.lab.lower()))
        serum_cols, sera = self._sera(h, cols["Sample date"], subtype, lineage, test_date)
        antigens, titres = self._antigens(h, id_col, cols, serum_cols, subtype, lineage, test_date)
        self._complete_years(h, serum_cols, sera, antigens, subtype)
        return Table(
            table_id="",
            group=group,
            lab=self.lab,
            subtype=subtype,
            lineage=lineage,
            assay="HI",
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
                "title": title,
                **({"serum_label": lbl} if (lbl := self._serum_label(h, serum_cols)) else {}),
            },
            dropped=dict(sorted((k, v) for k, v in self.dropped.items() if v)),
            warnings=sorted(set(self.warnings)),
        )

    # -- header ---------------------------------------------------------------------------

    def _title(self, h: int) -> str:
        for r in range(h):
            for v in self.s.rows[r]:
                if RBC.search(v):
                    return v
        raise self.fail(
            0, None, "no title naming the red cells ('...% <letter>RBC') above the header"
        )

    def _rbc(self, title: str) -> str:
        m = RBC.search(title)
        assert m is not None
        rbc = RBC_LETTER.get(m[2].upper())
        if rbc is None:
            raise self.fail(0, None, f"red cells {m[0]!r} not known")
        return rbc

    def _test_date(self, h: int) -> dt.date:
        found = [
            (r, c, m[1])
            for r in range(h)
            for c, v in enumerate(self.s.rows[r])
            if (m := TEST_DATE.fullmatch(v))
        ]
        if len(found) != 1:
            raise self.fail(0, None, f"expected one 'HI test date:' cell, found {len(found)}")
        r, c, text = found[0]
        file_date = _file_date(self.s.path)
        readings = {}
        for order in dates.ORDERS:
            with contextlib.suppress(dates.DateError):
                readings[order] = dates.parse(text, order)[0]
        if not readings:
            raise self.fail(r, c, f"test date {text!r} is not a date")
        lab_reading = readings.get(self.date_order)
        if file_date is None:
            if lab_reading is None and len(set(readings.values())) > 1:
                raise self.fail(r, c, f"test date {text!r} is ambiguous and the file has no date")
            return dt.date.fromisoformat(lab_reading or next(iter(readings.values())))
        if lab_reading == file_date:
            return dt.date.fromisoformat(file_date)
        if file_date in readings.values():
            self.warnings.append(
                f"{self.s.where(r, c)}: test date {text!r} read as {file_date}, the file's date"
            )
            return dt.date.fromisoformat(file_date)
        raise self.fail(r, c, f"test date {text!r} is not the file's date {file_date}")

    def _serum_label(self, h: int, serum_cols: list[int]) -> str:
        """A label over the serum columns ("Rabbit serum"): kept in meta. The sera's species
        is not set from it; NIID stopped writing it while the same sera stayed in use."""
        labels = {
            v
            for r in range(h)
            for c in serum_cols[:1]
            if re.search(r"serum", v := self.s.cell(r, c), re.IGNORECASE)
        }
        return "; ".join(sorted(labels))

    # -- sera -----------------------------------------------------------------------------

    def _sera(
        self, h: int, last_fixed: int, subtype: str, lineage: str, test_date: dt.date
    ) -> tuple[list[int], list[Serum]]:
        header = self.s.rows[h]
        serum_cols, sera = [], []
        for c in range(last_fixed + 1, len(header)):
            text = header[c]
            if not text:
                continue
            if (rule := self.rules.control_sera.find(text, lab=self.lab, field="name")) is not None:
                if rule["action"] != "drop":
                    raise self.fail(h, c, f"control_sera action {rule['action']!r} not read here")
                self.dropped["sera: control"] += 1
                continue
            parsed = parse_serum_header(text)
            if parsed is None:
                self._check_annotation_column(h, c)
                continue
            serum_cols.append(c)
            sera.append(self._serum(h, c, parsed, subtype, lineage, test_date))
        if not sera:
            raise self.fail(h, None, "no serum columns")
        return serum_cols, sera

    def _serum(
        self, h: int, c: int, parsed: SerumHeader, subtype: str, lineage: str, test_date: dt.date
    ) -> Serum:
        text = (
            parsed.name
            if re.match(r"[AB]/", parsed.name, re.IGNORECASE)
            else (f"{subtype[0]}/{parsed.name}")
        )
        if NO_YEAR.fullmatch(text):
            self.no_year[c] = text.rstrip("/")
            text = f"{text.rstrip('/')}/{test_date.year}"  # placeholder: see _complete_years
        problems: list[str] = []
        name, renamed = aliases.parse_name(
            self.rules,
            text,
            lab=self.lab,
            subtype=subtype,
            applies_to="serum",
            warnings=problems,
            not_after=test_date.year,
        )
        self.warnings.extend(
            f"{self.s.where(h, c)}: {p}" for p in problems if "two-digit year" not in p
        )
        passage = _TYPE_PASSAGE.get(parsed.type_word, "")
        if parsed.passage:
            read = self.passages.parse(niid_passage(parsed.passage, self.passages))
            self.warnings.extend(f"{self.s.where(h, c)}: {p}" for p in read.problems)
            passage = read.text
        reassortant = name.reassortant
        if parsed.reassortant:
            found, annotation = _split_reassortant(parsed.reassortant, self.rules, self.lab)
            if not found:
                self.warnings.append(
                    f"{self.s.where(h, c)}: {parsed.reassortant!r} is not a known reassortant"
                )
            reassortant = found or reassortant
            if annotation:
                name.annotations.append(annotation)
        serum = Serum(
            name=name.name,
            raw_name=self.s.cell(h, c),
            serum_id=f"{self.lab} {parsed.serum_id}",
            passage=passage,
            reassortant=reassortant,
            annotations=[*name.annotations, *([parsed.lab] if parsed.lab else [])],
            lineage=lineage,
            source={"column": c + 1},
        )
        if renamed:
            serum.source[aliases.SOURCE_KEY] = renamed
        return serum

    def _complete_years(
        self,
        h: int,
        serum_cols: list[int],
        sera: list[Serum],
        antigens: list[Antigen],
        subtype: str,
    ) -> None:
        """A serum header written without a year takes it from the table's own antigen of
        that name (the serum's virus is almost always among the reference antigens). No such
        antigen, or two with different years, is an error: the year is never guessed."""
        for c, text in self.no_year.items():
            serum = sera[serum_cols.index(c)]
            stem = serum.name.rsplit("/", 1)[0]
            years = sorted(
                {a.name.rsplit("/", 1)[1] for a in antigens if a.name.rsplit("/", 1)[0] == stem}
            )
            if len(years) != 1:
                raise self.fail(
                    h, c, f"serum {text!r} has no year and the antigens give {years or 'none'}"
                )
            serum.name = f"{stem}/{years[0]}"
            self.warnings.append(
                f"{self.s.where(h, c)}: serum {text!r} has no year; {years[0]} from its antigen"
            )

    def _check_annotation_column(self, h: int, c: int) -> None:
        values = [v for r in range(h + 1, len(self.s.rows)) if (v := self.s.cell(r, c))]
        titres = [v for v in values if TITRE.fullmatch(_clean_titre(v))]
        if values and len(titres) > len(values) / 2:
            raise self.fail(
                h, c, f"column {self.s.cell(h, c)!r} holds titres but does not read as a serum"
            )

    # -- antigens -------------------------------------------------------------------------

    def _antigens(
        self,
        h: int,
        id_col: int,
        cols: dict[str, int],
        serum_cols: list[int],
        subtype: str,
        lineage: str,
        test_date: dt.date,
    ) -> tuple[list[Antigen], list[list[list[str]]]]:
        name_col, passage_col, date_col = (cols[k] for k in COLUMNS)
        antigens, titres = [], []
        for r in range(h + 1, len(self.s.rows)):
            row = self.s.rows[r]
            if not any(row):
                continue
            lab_id = self.s.cell(r, id_col)
            raw = self.s.cell(r, name_col)
            cells = [self.s.cell(r, c) for c in serum_cols]
            if lab_id.startswith("#") and all(v.startswith("#") for v in row if v):
                self.dropped["antigens: commented out by the lab (#)"] += 1
                self.warnings.append(f"{self.s.where(r, id_col)}: row commented out (#), dropped")
                continue
            if not lab_id:
                if any(TITRE.fullmatch(_clean_titre(v)) for v in cells):
                    raise self.fail(r, id_col, f"row with titres but no NIID-ID: {row[:4]}")
                if raw and not SECTION.fullmatch(raw) and raw.count("/") >= 3:
                    raise self.fail(r, name_col, f"strain {raw!r} without an NIID-ID")
                continue  # a section label or a note under the table
            if self.rules.control_antigens.find(raw, lab=self.lab) is not None:
                self.dropped["antigens: control"] += 1
                continue
            problems: list[str] = []
            name, renamed = aliases.parse_name(
                self.rules,
                raw,
                lab=self.lab,
                subtype=subtype,
                applies_to="antigen",
                warnings=problems,
                not_after=test_date.year,
            )
            self.warnings.extend(f"{self.s.where(r, name_col)}: {p}" for p in problems)
            raw_passage = self.s.cell(r, passage_col)
            try:
                passage = self.passages.parse(niid_passage(raw_passage, self.passages))
            except ValueError as err:
                raise self.fail(r, passage_col, str(err)) from err
            self.warnings.extend(f"{self.s.where(r, passage_col)}: {p}" for p in passage.problems)
            collected = self.s.cell(r, date_col)
            compact_id = re.sub(r"\s+", "", lab_id)
            antigen = Antigen(
                name=name.name,
                raw_name=raw,
                passage=passage.text,
                date=self._date(collected, r, date_col, test_date) if collected else None,
                lab_ids=[f"{self.lab}#{compact_id}"],
                reassortant=name.reassortant,
                annotations=name.annotations,
                lineage=lineage,
                source={"row": r + 1, "passage": raw_passage},
            )
            if renamed:
                antigen.source[aliases.SOURCE_KEY] = renamed
            antigens.append(antigen)
            titres.append([self._titre(r, c) for c in serum_cols])
        keep = [i for i, row in enumerate(titres) if any(row)]
        self.dropped["antigens: no readings"] += len(titres) - len(keep)
        return [antigens[i] for i in keep], [titres[i] for i in keep]

    def _date(self, text: str, r: int, c: int, test_date: dt.date) -> str | None:
        day, warning = dates.parse(text, self.date_order, not_after=test_date)
        if warning:
            self.warnings.append(f"{self.s.where(r, c)}: {warning}")
        if day > test_date.isoformat():
            self.warnings.append(f"{self.s.where(r, c)}: date {text!r} is after the test; left out")
            return None
        return day

    def _titre(self, r: int, c: int) -> list[str]:
        raw = self.s.cell(r, c)
        if not raw:
            self.dropped["cells: blank"] += 1
            return []
        if (rule := self.rules.titre_tokens.find(raw, lab=self.lab, assay="HI")) is not None:
            return [] if rule["titre"] == "*" else [rule["titre"]]
        text = _clean_titre(raw)
        if TITRE.fullmatch(text) and (text[0] in "<>" or int(text) >= 10):
            return [text]
        raise self.fail(r, c, f"titre {raw!r} matches no titre_tokens rule")


def _clean_titre(raw: str) -> str:
    return re.sub(r"\s+", "", raw.replace("＜", "<").replace("＞", ">"))


def _split_reassortant(text: str, rules: Rules, lab: str) -> tuple[str, str]:
    from .names import split_extra

    return split_extra(text, rules.reassortants, lab)


def _file_date(path: Path) -> str | None:
    m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", path.name)
    if m is None:
        return None
    try:
        return dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat()
    except ValueError:
        return None
