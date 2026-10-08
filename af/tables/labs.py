"""The WHO CC lab codes af's table readers can produce (``rules/tables/labs.tsv``).

One parser of that file for every consumer (serology, sequence matching): the codes a reader
could produce, not the labs a store happens to hold, so a lab is valid before its tables
arrive. The file is a rule table like the others (code, name, reader, evidence, added_by,
added_on, optional); a file without those columns, or with a code twice, is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .rules import RuleTable

SERUM_ID_PRINT = ("with-lab", "bare")


@dataclass(frozen=True)
class Lab:
    code: str  # as in Table.lab, e.g. "CDC"
    name: str
    reader: str  # the af module that reads the lab's source, or "none yet"
    # How this lab's serum ids are printed (Sarah, 8 Oct, Q123: as ae printed them; the stored
    # id stays "<LAB> <id>"): "with-lab" as stored, "bare" without the leading "<code> "
    serum_id_print: str


def _rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    rows = [dict(r.values) for r in RuleTable(path, scope=(), required=required).rules]
    codes = [r["code"] for r in rows]
    if dupes := sorted({c for c in codes if codes.count(c) > 1}):
        raise ValueError(f"{path}: lab code(s) listed twice: {dupes}")
    if bad := [c for c in codes if not c or c != c.upper() or " " in c]:
        raise ValueError(f"{path}: lab codes must be upper case without spaces: {bad}")
    return rows


def read_labs(path: Path) -> list[Lab]:
    rows = _rows(path, ("code", "name", "reader", "serum_id_print"))
    if bad := [
        (r["code"], r["serum_id_print"]) for r in rows if r["serum_id_print"] not in SERUM_ID_PRINT
    ]:
        raise ValueError(f"{path}: serum_id_print must be one of {SERUM_ID_PRINT}: {bad}")
    return [Lab(r["code"], r["name"], r["reader"], r["serum_id_print"]) for r in rows]


def read_lab_codes(path: Path) -> list[str]:
    """Only the codes. Callers that need no more (sequence matching) do not depend on the
    display columns, so their own synthetic files need not carry them."""
    return [r["code"] for r in _rows(path, ("code", "name", "reader"))]
