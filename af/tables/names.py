"""Strain-name normalisation for table antigens and sera: the mechanical part only.

Mechanics: upper-case, collapsed whitespace, the subtype prefix taken from the table, isolate
leading zeros stripped, a hyphen inside a location written as a space (acmacs conventions),
and whatever follows the year split off (a reassortant or an annotation, see ``split_extra``).

Location *spelling* (KYIV or KIEV, a CJK name, a GISAID alias) is curation, not mechanics. ae
rewrote it through locdb; af leaves the lab's spelling, so that names stay equal to the GISAID
names of the same viruses, until the shared location lookup (workstream 2, task 2.4) exists.

A name that is not type/location/isolate/year is kept, upper-cased, with the problem reported;
never reinterpreted (ae guesses silently, D-ingestion T22).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .rules import RuleTable

_SPACES = re.compile(r"\s+")
_YEAR_AND_EXTRA = re.compile(r"(\d{4})(?:[\s-]+(.+))?")  # "2019", "2019 X-345", "2019-CDC-LV25B"
_PAREN = re.compile(r"\((.*)\)")
_TRAILING_PAREN = re.compile(
    r"\s*\(([^()]*)\)\s*$"
)  # "B/EXAMPLETOWN/1/2021 (23/228)": may contain '/'


@dataclass
class Name:
    name: str
    reassortant: str = ""
    annotations: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


_TYPE_HOMOGLYPHS = str.maketrans({"\u0410": "A", "\u0412": "B"})  # Cyrillic А, В
_TWO_DIGIT_YEAR = re.compile(r"(\d{2})(?:[\s-]+(.+))?")


def parse(
    raw: str, subtype: str, reassortants: RuleTable, lab: str, *, not_after: int | None = None
) -> Name:
    """``subtype`` is the table's: "A(H1N1)", "A(H3N2)" or "B".

    ``not_after`` (the table's test year) allows a two-digit year: 20yy unless that is later
    than the test, then 19yy. Without it a two-digit year is reported, never guessed from
    today's date.
    """
    text = _SPACES.sub(" ", raw.strip().upper())
    # A trailing "(...)" is split off first: it can itself contain a '/' (eu-d5, 38 GISAID
    # names such as "(23/228)"). A second trailing group stays in the name and is reported.
    paren = ""
    if (m := _TRAILING_PAREN.search(text)) is not None and "/" in text[: m.start()]:
        paren, text = m[1].strip(), text[: m.start()]
    text = re.sub(r"\s*/\s*", "/", text)
    parts = text.split("/")
    if len(parts) < 4:
        return Name(text, problems=[f"name {raw!r}: expected type/location/isolate/year"])
    flu_type, *middle, last = parts
    problems = []
    if (latin := flu_type.translate(_TYPE_HOMOGLYPHS)) != flu_type:
        # Cyrillic letters that look like A or B (seen in names CNIC relays from Russian labs)
        problems.append(f"name {raw!r}: Cyrillic letter in the type read as {latin!r}")
        flu_type = latin
    prefix = subtype
    if flu_type not in ("A", "B") and not flu_type.startswith("A("):
        problems.append(f"name {raw!r}: unknown type {flu_type!r}")
        prefix = flu_type
    elif flu_type[0] != subtype[0] or (flu_type.startswith("A(") and flu_type != subtype):
        problems.append(f"name {raw!r}: {flu_type} virus on a {subtype} table")
        prefix = flu_type  # keep what the lab wrote; never relabel (D-ingestion T40)
    m = _YEAR_AND_EXTRA.fullmatch(last)
    short = _TWO_DIGIT_YEAR.fullmatch(last) if m is None and not_after is not None else None
    if m is not None:
        year, extra = m[1], m[2] or ""
    elif short is not None and not_after is not None:
        full = 2000 + int(short[1])
        year, extra = str(full if full <= not_after else full - 100), short[2] or ""
        problems.append(f"name {raw!r}: two-digit year {short[1]!r} read as {year}")
    else:
        problems.append(f"name {raw!r}: {last!r} does not start with a four-digit year")
        year, extra = last, ""
    if len(middle) == 2:
        location, isolate = middle
    else:
        problems.append(f"name {raw!r}: {len(middle)} fields between type and year")
        location, isolate = middle[0], "-".join(middle[1:])
    location = location.replace("-", " ").strip()
    isolate = isolate.lstrip("0") or isolate  # "02" -> "2"
    name = Name("/".join((prefix, location, isolate, year)), problems=problems)
    for part in (extra, paren):
        if part:
            reassortant, annotation = split_extra(part, reassortants, lab)
            if reassortant and name.reassortant:
                name.problems.append(
                    f"name {raw!r}: two reassortants {name.reassortant!r}, {reassortant!r}"
                )
            name.reassortant = name.reassortant or reassortant
            if annotation:
                name.annotations.append(annotation)
    return name


def split_extra(extra: str, reassortants: RuleTable, lab: str) -> tuple[str, str]:
    """Text after the year -> (reassortant, annotation). One of the two is empty."""
    text = extra.strip()
    if (m := _PAREN.fullmatch(text)) is not None:
        text = m[1].strip()
    rule = reassortants.find(text, lab=lab)
    if rule is None:
        return "", text
    regex = re.compile(rule["pattern"], re.IGNORECASE)
    return regex.sub(rule["canonical"], text).upper(), ""
