"""Strain-name normalisation, shared by the sequence store and the table parsers.

One implementation, because a sequence and a table antigen for the same virus must
produce the *same* string or the two never match. Workstream 3 wrote the mechanical
rules for table names (tidy, one lab's house style); this module is those rules
hardened against GISAID names (submitted by thousands of labs, and much messier).

What it does, and nothing more:

- upper-case, collapse whitespace, drop spaces around ``/``;
- split into type / location / isolate / year;
- strip the isolate's leading zeros (``02`` -> ``2``);
- write a hyphen inside a location as a space;
- split off whatever follows the year, for the caller to interpret.

What it deliberately does not do:

- **Rewrite a location's spelling.** ae resolved every location through locdb, which
  meant two accepted aliases of one place produced two different strain names, and the
  name depended on a database that is hard to audit. af keeps what the submitter wrote,
  so a sequence name and a table name agree by construction. The location *lookup*
  (coordinates, country, region) is separate and lives in :mod:`af.seq.locations`.
- **Guess at a name it cannot parse.** A name of the wrong shape is returned as it
  came, with a problem recorded. ae reinterpreted such names silently, which turned an
  extra slash into a wrong isolate and, in one shape, read part of the isolate as the
  year. Reporting beats guessing: an unparsable name is usually a submitter's typo that
  someone should see.

Problem codes are stable identifiers, not prose, so a caller can count them per rule
and a store can report them per pull.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Problem codes. Stable: they appear in counts, reports and provenance.
SHAPE = "name.shape"  # fewer than four /-separated parts
TYPE_UNKNOWN = "name.type"  # first part is not A, A(HxNy) or B
TYPE_MISMATCH = "name.type_mismatch"  # name's type disagrees with the source's
YEAR = "name.year"  # last part does not start with a four-digit year
FIELDS = "name.fields"  # more than two parts between type and year
EXTRA_SLASH_IN_PAREN = "name.extra_slash_in_paren"  # trailing "(...)" held a slash

_SPACES = re.compile(r"\s+")
_AROUND_SLASH = re.compile(r"\s*/\s*")
_YEAR_AND_EXTRA = re.compile(r"(\d{4})(?:[\s-]+(.+))?")
_PAREN_TAIL = re.compile(r"\s*\(([^()]*)\)\s*$")
_TYPE = re.compile(r"A\([^)]*\)|[ABCD]")


@dataclass
class Name:
    """A normalised name, what was left over, and what went wrong.

    ``extra`` is the text after the year, uninterpreted: the caller decides whether it
    is a reassortant or an annotation, because that needs a rule table this module does
    not own. ``problems`` holds :data:`SHAPE` and friends, in the order found.
    """

    name: str
    extra: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def normalise(raw: str, subtype: str | None = None) -> Name:
    """Normalise one strain name.

    ``subtype`` is what the *source* says the virus is ("A(H1N1)", "A(H3N2)", "B"):
    the table it came from, or the GISAID query that returned it. When the name
    disagrees, the name wins and :data:`TYPE_MISMATCH` is recorded — relabelling a
    virus to match the file it arrived in loses the evidence that something is wrong.
    Pass ``None`` when the source does not say.
    """
    text = _AROUND_SLASH.sub("/", _SPACES.sub(" ", raw.strip()).upper())
    problems: list[str] = []

    # A trailing "(...)" comes off BEFORE the split: it can itself contain a slash,
    # e.g. a name ending "(23/228)", and splitting first turns the year into part of
    # the isolate. Measured on real pulls: 38 names, of which 35 are repaired by this.
    paren = ""
    if (match := _PAREN_TAIL.search(text)) is not None:
        paren = match.group(1).strip()
        if "/" in paren:
            problems.append(EXTRA_SLASH_IN_PAREN)
        text = _PAREN_TAIL.sub("", text)

    parts = text.split("/")
    if len(parts) < 4:
        return Name(_rejoin(text, paren), problems=[SHAPE, *problems])

    flu_type, *middle, last = parts

    if _TYPE.fullmatch(flu_type) is None:
        problems.append(TYPE_UNKNOWN)
    elif subtype is not None and flu_type != subtype and flu_type != subtype[0]:
        problems.append(TYPE_MISMATCH)

    if (match := _YEAR_AND_EXTRA.fullmatch(last)) is None:
        problems.append(YEAR)
        year, extra = last, ""
    else:
        year, extra = match.group(1), match.group(2) or ""

    if len(middle) == 2:
        location, isolate = middle
    else:
        # A lab or centre code between the location and the isolate is the common
        # cause. Joining with "-" is what the old pipeline needed doing by hand.
        problems.append(FIELDS)
        location, isolate = middle[0], "-".join(middle[1:])

    location = location.replace("-", " ").strip()
    isolate = _strip_padding(isolate)

    if paren:
        extra = f"{extra} ({paren})".strip()
    return Name("/".join((flu_type, location, isolate, year)), extra, problems)


def _strip_padding(isolate: str) -> str:
    """Drop leading zeros from every all-digit part of an isolate.

    Labs pad isolate numbers, so ``02`` and ``2`` are one virus and must give one
    name. The padding can sit in the last part of a joined isolate as easily as in a
    plain one, and leaving it there would reintroduce exactly the duplicate the rule
    exists to prevent. A part that is not all digits is left alone: its zeros may be
    part of a code.
    """
    parts = [(part.lstrip("0") or "0") if part.isdigit() else part for part in isolate.split("-")]
    return "-".join(parts)


def _rejoin(text: str, paren: str) -> str:
    """Put a parenthesised tail back on a name that could not be parsed."""
    return f"{text} ({paren})" if paren else text
