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


@dataclass(frozen=True)
class Lab:
    code: str  # as in Table.lab, e.g. "CDC"
    name: str
    reader: str  # the af module that reads the lab's source, or "none yet"


def read_labs(path: Path) -> list[Lab]:
    table = RuleTable(path, scope=(), required=("code", "name", "reader"))
    labs = [Lab(r["code"], r["name"], r["reader"]) for r in table.rules]
    codes = [lab.code for lab in labs]
    if dupes := sorted({c for c in codes if codes.count(c) > 1}):
        raise ValueError(f"{path}: lab code(s) listed twice: {dupes}")
    if bad := [c for c in codes if not c or c != c.upper() or " " in c]:
        raise ValueError(f"{path}: lab codes must be upper case without spaces: {bad}")
    return labs


def read_lab_codes(path: Path) -> list[str]:
    return [lab.code for lab in read_labs(path)]
