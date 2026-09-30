"""Does a GISAID record's passage match an antigen preparation's?

Used for one thing only: a preparation whose table rows name two different GISAID records of
the same virus takes the record whose passage matches its own (Sarah, Q81, 30 Sep 2026:
"Passage-matched record"). **Passage stays part of antigen identity**: egg and cell
preparations are different antigens and are never merged here. This chooses which sequence
colours one preparation, among records its own rows already name.

Both passages are written canonically with the tables' passage-token rules
(``rules/tables/passage_tokens.tsv``, :class:`af.tables.passage.PassageParser`), taking only
the rules every lab shares (GISAID's text is not one lab's): GISAID's "S2 (2024-03-19)" and a
table's "SIAT2 (2024-03-19)" are the same passage. A date in brackets is the harvest date;
GISAID records write it several ways ("2024-03-19", "12/11/2024", "102121").

A record scores 2 when its steps and its harvest date both equal the preparation's, 1 when
its steps equal and one side has no date, 0 otherwise (different steps, different dates, or
a passage that does not read).
"""

from __future__ import annotations

import re
from pathlib import Path

from af.tables.passage import PassageParser
from af.tables.rules import RuleTable

#: A scope no lab has, so that only the passage-token rules every lab shares apply.
SHARED_RULES_ONLY = "*shared*"
_BRACKET = re.compile(r"\(([^()]*)\)")
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_US = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_MMDDYY = re.compile(r"(\d{2})(\d{2})(\d{2})")


class PassageMatcher:
    def __init__(self, tokens: RuleTable) -> None:
        self.parser = PassageParser(tokens, SHARED_RULES_ONLY)

    @classmethod
    def read(cls, path: Path) -> PassageMatcher:
        return cls(
            RuleTable(path, scope=("lab",), required=("kind", "pattern", "canonical", "class"))
        )

    def canonical(self, text: str) -> tuple[str | None, str]:
        """(canonical steps or None if unreadable, ISO harvest date or "")."""
        found = _BRACKET.search(text)
        steps = (text[: found.start()] if found else text).strip()
        date = _iso_date(found[1]) if found else ""
        steps = re.sub(r"\s*,\s*", "/", steps)  # "C2, MDCK2": two segments
        passage = self.parser.parse(steps)
        return (None if passage.problems or not passage.text else passage.text), date

    def score(self, preparation: str, record: str) -> int:
        mine, my_date = self.canonical(preparation)
        theirs, their_date = self.canonical(record)
        if mine is None or theirs is None or mine != theirs:
            return 0
        if my_date and their_date:
            return 2 if my_date == their_date else 0
        return 1


def _iso_date(text: str) -> str:
    """A bracketed date as ISO; "" when it is not a date (a lab's note)."""
    text = text.strip()
    if m := _ISO.fullmatch(text):
        return text
    if m := _US.fullmatch(text):
        return f"{m[3]}-{int(m[1]):02d}-{int(m[2]):02d}"
    if m := _MMDDYY.fullmatch(text):
        return f"20{m[3]}-{m[1]}-{m[2]}"
    return ""
