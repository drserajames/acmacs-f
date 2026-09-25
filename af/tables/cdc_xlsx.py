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

from . import aliases, dates
from .cdc import LAB, ReadResult
from .model import Antigen, Serum, Table
from .passage import PassageParser
from .rules import Rules
from .sheet import Sheet, SheetError, load

TITLE = r"HEMAGGLUTINATION INHIBITION REACTIONS OF INFLUENZA (.+) VIRUSES"
TITLE_SUBTYPES = {
    "H3": ("A(H3N2)", "", "h3"),
    "A(H3N2)": ("A(H3N2)", "", "h3"),
    "A(H1N1)PDM09": ("A(H1N1)", "", "h1pdm"),
    "H1N1PDM09": ("A(H1N1)", "", "h1pdm"),
    "B/VICTORIA": ("B", "VICTORIA", "bvic"),
    "B VICTORIA LINEAGE": ("B", "VICTORIA", "bvic"),
}
RBC = {"GUINEA PIG": "guinea-pig", "TURKEY": "turkey"}
# The passage cell: "S1(07/19/2024)<NY>", "E4/E1(9/13/2024)LOT#10", "S2".
PASSAGE_CELL = re.compile(
    r"(?P<passage>[^(<]*)\s*(?:\((?P<date>[^)]*)(?P<close>\))?)?\s*(?P<extra>.*)"
)
SITE = re.compile(r"<?([A-Z]{2})>?")  # the state lab that isolated it: kept in source
LOT = re.compile(r"LOT\s*#\s*\d+", re.IGNORECASE)  # distinguishes egg lots: an annotation
SECTION = re.compile(r"(REFERENCE|TEST) VIRUSES", re.IGNORECASE)


class CDCSheetError(SheetError):
    pass


def read(paths: list[Path], rules: Rules) -> ReadResult:
    result = ReadResult(tables=[], rows=0)
    for path in paths:
        data = path.read_bytes()
        provenance = {
            "file": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "reader": "af.tables.cdc_xlsx",
        }
        for sheet in load(path):
            if not sheet.find(r"REFERENCE VIRUSES"):
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
        self.warnings: list[str] = []
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
        ref_row, _ = self.s.find_one(r"REFERENCE VIRUSES")
        columns = self._serum_columns(ref_row)
        label_row = self._label_row(ref_row)
        sera = self._sera(columns, subtype, test_date)
        columns = {letter: columns[letter] for letter, _ in sera}  # control sera removed
        antigens, titres = self._antigens(label_row, columns, subtype, test_date)
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
        return (*TITLE_SUBTYPES[raw], "hi_oseltamivir_protocol" if oseltamivir else "hi_protocol")

    def _test_date(self) -> dt.date:
        found = set()
        for r, c in self.s.find(r"(DATE TESTED|Test Date):\s*(.+)"):
            text = re.fullmatch(
                r"(?:DATE TESTED|Test Date):\s*(.+)", self.s.cell(r, c), re.IGNORECASE
            )[1]  # type: ignore[index]
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
        """Letter -> column, from the nearest letters row at or above REFERENCE VIRUSES."""
        for r in range(ref_row, max(ref_row - 6, -1), -1):
            row = self.s.rows[r]
            if "A" in row:
                c0 = row.index("A")
                letters = {}
                for c in range(c0, len(row)):
                    expected = chr(ord("A") + c - c0)
                    if row[c] != expected:
                        break
                    letters[expected] = c
                if len(letters) >= 2:
                    return letters
        raise self.fail(ref_row, None, "no serum letters row (A, B, C ...) above REFERENCE VIRUSES")

    def _label_row(self, ref_row: int) -> int:
        for r in (ref_row, ref_row + 1):
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
        self, label_row: int, columns: dict[str, int], subtype: str, test_date: dt.date
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
        antigens, titres = [], []
        for r in range(label_row + 1, end):
            name_raw = self.s.cell(r, name_col)
            cells = [self.s.cell(r, c) for c in titre_cols]
            if not name_raw and not any(cells):
                continue
            if SECTION.fullmatch(name_raw) or (
                SECTION.fullmatch(self.s.cell(r, 0)) and not any(cells)
            ):
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
            )
            self.warnings.extend(f"{self.s.where(r, name_col)}: {p}" for p in problems)
            passage, harvest, annotations, site = self._passage_cell(r, passage_cols, test_date)
            collected = self.s.cell(r, date_col)
            antigens.append(
                Antigen(
                    name=name.name,
                    raw_name=name_raw,
                    passage=passage,
                    passage_class=self.passages.passage_class(passage),
                    passage_date=harvest,
                    date=dates.parse(collected, self.date_order)[0] if collected else None,
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
        keep = [i for i, row in enumerate(titres) if any(row)]
        self.dropped["antigens: no readings"] += len(titres) - len(keep)
        return [antigens[i] for i in keep], [titres[i] for i in keep]

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
        passage = self.passages.parse(m["passage"].strip())
        self.warnings.extend(f"{self.s.where(r, c)}: {p}" for p in passage.problems)
        harvest = None
        if m["date"] is not None and not m["close"]:
            self.warnings.append(
                f"{self.s.where(r, c)}: passage {text!r} has no closing parenthesis"
            )
        if m["date"]:
            harvest, warning = dates.parse(m["date"], self.date_order, not_after=test_date)
            if warning:
                self.warnings.append(f"{self.s.where(r, c)}: {warning}")
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
            return [raw]
        raise self.fail(r, c, f"titre {raw!r} matches no titre_tokens rule")

    # -- sera -----------------------------------------------------------------------------

    def _sera(
        self, columns: dict[str, int], subtype: str, test_date: dt.date
    ) -> list[tuple[str, Serum]]:
        r0, _ = self.s.find_one(r"REFERENCE ANTISER(A|UM)")
        labels = {v.upper(): c for c, v in enumerate(self.s.rows[r0]) if v}
        for need in ("LOT", "PASSAGE", "BOOSTED", "SPECIES"):
            if need not in labels:
                raise self.fail(r0, None, f"no {need} column in the antisera block")
        by_letter: dict[str, Serum] = {}
        r = r0 + 1
        while re.fullmatch(r"[A-Z]", self.s.cell(r, 0)):
            letter = self.s.cell(r, 0)
            raw = self.s.cell(r, 1)
            lot = self.s.cell(r, labels["LOT"])
            rule = self.rules.control_sera.find(lot, lab=LAB)
            if rule is not None and rule["action"] == "drop":
                self.dropped["sera: control"] += 1
                by_letter[letter] = None  # type: ignore[assignment]
                r += 1
                continue
            problems: list[str] = []
            name, serum_renamed = aliases.parse_name(
                self.rules, raw, lab=LAB, subtype=subtype, applies_to="serum", warnings=problems
            )
            self.warnings.extend(f"{self.s.where(r, 1)}: {p}" for p in problems)
            passage, harvest, annotations, _ = self.passage_text(
                self.s.cell(r, labels["PASSAGE"]), r, labels["PASSAGE"], test_date
            )
            boosted = self.s.cell(r, labels["BOOSTED"]).upper()
            if boosted not in ("Y", "N", ""):
                raise self.fail(r, labels["BOOSTED"], f"BOOSTED {boosted!r}")
            species = self.s.cell(r, labels["SPECIES"]).upper()
            serum = Serum(
                name=name.name,
                raw_name=raw,
                serum_id=f"CDC {lot}",
                passage=passage,
                passage_class=self.passages.passage_class(passage),
                passage_date=harvest,
                species="" if species == "FERRET" else species,
                reassortant=name.reassortant,
                annotations=name.annotations
                + annotations
                + (["BOOSTED"] if boosted == "Y" else []),
                source={
                    "row": r + 1,
                    "letter": letter,
                    "lot": lot,
                    "pool": self.s.cell(r, labels["POOL"]) if "POOL" in labels else "",
                },
            )
            if serum_renamed:
                serum.source[aliases.SOURCE_KEY] = serum_renamed
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
            rule = self.rules.control_sera.find(serum.source["lot"], lab=LAB)
            if rule is not None and rule["action"] == "species":
                serum.species = rule["value"]
            out.append((letter, serum))
        return out
