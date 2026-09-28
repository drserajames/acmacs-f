"""Abbreviated serum names resolved to a table's own antigens (Crick; VIDRL's reader has its
own variant without the year).

Labs write a serum's virus short (``B/Exa`` over ``60/08``, ``Sth Exa 272``). The serum is
the table's antigen whose isolate (and year, when written) is the abbreviation's and whose
location starts with its letters, or failing that has them in order ("Sth Exa" for SOUTH
EXAMPLELAND). Spaces, hyphens, dots, underscores and apostrophes are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model import Antigen


@dataclass(frozen=True)
class Named:
    """The name fields a serum takes from its antigen."""

    name: str
    reassortant: str = ""


def match(
    location: str, isolate: str, year: str, antigens: list[Antigen], reassortant: str = ""
) -> dict[tuple[str, str], Named]:
    """Distinct (name, reassortant) of the antigens the abbreviation can name. With no
    reassortant written, the plain virus wins when the table has it too."""
    loc = squash(location)
    iso = _isolate(isolate)
    prefix: dict[tuple[str, str], Named] = {}
    ordered: dict[tuple[str, str], Named] = {}
    for a in antigens:
        parts = a.name.split("/")
        if len(parts) != 4 or _isolate(parts[2]) != iso or not _same_year(year, parts[3]):
            continue
        if reassortant and a.reassortant != reassortant:
            continue
        key = (a.name, a.reassortant)
        full = squash(parts[1])
        if full.startswith(loc):
            prefix[key] = Named(a.name, a.reassortant)
        elif loc and full[:1] == loc[:1] and in_order(loc, full):
            ordered[key] = Named(a.name, a.reassortant)
    found = prefix or ordered
    if reassortant:
        return found
    plain = {k: v for k, v in found.items() if not k[1]}
    return plain or found


def squash(text: str) -> str:
    return re.sub(r"[\s\-_.'’]", "", text.upper())


def in_order(letters: str, word: str) -> bool:
    it = iter(word)
    return all(ch in it for ch in letters)


def is_dilution(n: int) -> bool:
    """10 x 2^k: a value off the series is a typing error (604 for 640, 2506 for 2560)."""
    return n >= 10 and n % 10 == 0 and (n // 10) & (n // 10 - 1) == 0


def _isolate(text: str) -> str:
    t = squash(text)
    return t.lstrip("0") or t


def _same_year(written: str, full: str) -> bool:
    w = written.strip()
    if not w:
        return True
    if len(w) == 2:
        return full[-2:] == w
    return full == w
