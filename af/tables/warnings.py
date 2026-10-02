"""Table warnings, sorted so the run report can be read.

A reader warns about anything it read that is not plain: a rule that fired, a value it could
not read and kept as written, a choice it made between two readings. The store holds thousands
of them (about 3,500 in Oct 2026), most of them legitimate, and a real problem among them is
invisible: a curated name 'B/\\1' was flagged by the parser on 1 Oct 2026, among 1,700
legitimate warnings about CDC sample ids, and published (notes/tables/WARNINGS-CENSUS.md).

So each warning gets a class:

- **applied**: a named rule or a documented reader decision fired; the warning records it;
- **expected**: a label that is not a strain name, by design (CDC's sample ids, serum pools):
  an unparseable name that does not even begin like one (``A/``, ``B/``);
- **tolerated**: a source defect kept as written (a passage the parser cannot read, a sample
  date after the test);
- **review**: anything else, including an unparseable name that begins like a strain name.

The report counts the first two classes and lists the last two in full, so a new kind of
warning arrives in front of a reader instead of under the noise. The patterns match the
messages the readers write; a reader that changes a message must change its pattern here.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from .model import Table

APPLIED = {
    "name rewritten by a name_rewrites rule": r"rewritten .* by name_rewrites",
    "two-digit year read within the test year": r"two-digit year .* read as",
    "repeated rows of one sample merged": r"repeats row \d+; readings merged",
    "serum named by a strain_aliases rule": r"named by strain_aliases",
    "cell repaired by a cell_fixes rule": r"fixed as .* by cell_fixes",
    "name renamed by a strain_aliases rule": r"renamed .* by strain_aliases",
    "serum lot restored by a serum_ids rule": r"restored as .* by serum_ids",
    "Cyrillic letter in the type": r"Cyrillic letter in the type",
    "Chinese location kept (locationdb names it, no spelling)": r"Chinese location kept",
    "footnote mark written without '*'": r"footnote \d+ written without",
    "Excel serial date": r"is an Excel serial number",
    "serum year taken from its antigen": r"has no year; \d+ from its antigen",
    "serum column order taken from the index row": r"followed the index row|reorders the columns",
    "test date read as the file's date": r"read as .*, the file's date",
    "passage read from the ID column": r"the passage is in the ID column",
    "reference passage taken from its serum": r"no passage written; .* from its serum",
    "antigen merges two CDC isolate ids": r"antigen merges two CDC isolate ids",
    "note on a section row": r"note on the section row",
    "QC block under the table not read": r"block not read",
    "point lineage from the sheet title": r"lineage \w+ from the sheet title for",
    "table read from ae's .ace (no workbook)": r"ae \.ace, no workbook, read by ace_imports",
}
TOLERATED = {
    "name with extra fields": r"\d+ fields between type and year",
    "passage kept as written": r"passage .*: cannot read",
    "serum index row not 1..n; columns by position": r"columns taken by position",
    "antigen row without an index number": r"antigen row without an index number",
    "passage copies differ in the site tag only": r"differ in the site tag only",
    "one titre in a row with no name": r"one titre in a row with no name",
    "footnote not in the legend": r"not in the legend",
    "sheet name not the title's date": r"is not the title's date",
    "sample date after the test, left out": r"is after the test; left out",
    "no four-digit year": r"does not start with a four-digit year",
    "serum with no id": r"has no id",
    "unknown type": r"unknown type",
    "passage with no closing parenthesis": r"has no closing parenthesis",
    "date read in the other order": r"is not [DMY]{3}; read as",
    "test date from an unlabelled cell": r"test date from an unlabelled cell",
}
UNPARSED = re.compile(r"name '(.*)': expected type/location/isolate/year")
STRAIN_LIKE = re.compile(r"\s*[AB]\s*/", re.IGNORECASE)


def classify(warning: str) -> tuple[str, str]:
    """(class, kind): class is applied, expected, tolerated or review."""
    if (m := UNPARSED.search(warning)) is not None:
        if STRAIN_LIKE.match(m[1]):
            return "review", "a name that begins like a strain name and does not parse"
        return "expected", "a label that is not a strain name (sample ids, pools)"
    for kind, pattern in APPLIED.items():
        if re.search(pattern, warning):
            return "applied", kind
    for kind, pattern in TOLERATED.items():
        if re.search(pattern, warning):
            return "tolerated", kind
    return "review", "unclassified"


def report(tables: Iterable[Table]) -> list[str]:
    """Report lines: counts for applied and expected, every tolerated and review warning."""
    counts: Counter[tuple[str, str]] = Counter()
    listed: dict[str, list[str]] = {"tolerated": [], "review": []}
    for table in tables:
        for warning in table.warnings:
            cls, kind = classify(warning)
            counts[(cls, kind)] += 1
            if cls in listed:
                listed[cls].append(f"[{kind}] {table.table_id or table.source_key}: {warning}")
    total = Counter(cls for cls, _ in counts.elements())
    lines = [
        "warnings: "
        + ", ".join(f"{cls} {total[cls]}" for cls in ("applied", "expected", "tolerated", "review"))
    ]
    for cls in ("applied", "expected"):
        kinds = sorted(((n, k) for (c, k), n in counts.items() if c == cls), reverse=True)
        lines.extend(f"  {cls}: {n} {k}" for n, k in kinds)
    for cls in ("review", "tolerated"):
        lines.append(f"  {cls} ({len(listed[cls])}), each:")
        lines.extend(f"    {line}" for line in sorted(listed[cls]))
    contradictions = date_contradictions(tables)
    lines.append(
        f"  review, harvest date before the collection date ({len(contradictions)}), each:"
    )
    lines.extend(f"    {line}" for line in contradictions)
    return lines


def date_contradictions(tables: Iterable[Table]) -> list[str]:
    """Antigens whose harvest date precedes their own collection date: a fact the table itself
    contradicts (a CDC antigen harvested two years before it was collected). Found by a
    consumer of the store on 1 Oct 2026; the check is the store's own now. Listed, not fixed:
    which date is wrong is the lab's to say."""
    out = []
    for table in tables:
        for antigen in table.antigens:
            harvest, collected = antigen.passage_date, antigen.date
            if harvest and collected and harvest < collected:
                out.append(
                    f"{table.table_id or table.source_key}: {antigen.name} {antigen.passage} "
                    f"harvested {harvest}, collected {collected}"
                )
    return sorted(out)


RULE_OUTPUT = re.compile(r"name '.*' (?:rewritten|renamed) '(.*)' by (\S+)")
NAME_PROBLEM = re.compile(r"name '(.*)': (?!.* by )")  # a parse problem with a name, not a rule
FIXED_CELL = re.compile(r"^(\S+\[[^\]]*\]![A-Z]+\d+): .* fixed as .* by (cell_fixes\.tsv:\d+)")
CELL = re.compile(r"^(\S+\[[^\]]*\]![A-Z]+\d+): ")


def curated_errors(tables: list[Table]) -> list[str]:
    """Located errors for a curated rule whose own output reads badly. A lab's raw text may
    warn, but a value a rule wrote must read cleanly: 'B/\\1' was a warning among 1,700 on
    1 Oct 2026 and was published (notes/tables/WARNINGS-CENSUS.md).

    - a name written by name_rewrites or strain_aliases that the parser then flags;
    - a cell written by cell_fixes that draws a tolerated or review warning.

    A serum id restored by serum_ids cannot be checked against the tables: the lab lost that
    lot everywhere, which is why the rule exists. Its evidence is the source's history, and
    the rules loader refuses a blank or substituting canonical (af.tables.rules).
    """
    errors: list[str] = []
    for table in tables:
        where = table.table_id or table.source_key
        outputs = {m[1]: m[2] for w in table.warnings if (m := RULE_OUTPUT.search(w))}
        fixed = {m[1]: m[2] for w in table.warnings if (m := FIXED_CELL.match(w))}
        for warning in table.warnings:
            cls, _ = classify(warning)
            if cls == "applied":
                continue
            if (m := NAME_PROBLEM.search(warning)) and m[1] in outputs:
                errors.append(
                    f"{where}: {outputs[m[1]]} wrote a name that does not parse: {warning}"
                )
            if (m := CELL.match(warning)) and m[1] in fixed:
                errors.append(f"{where}: {fixed[m[1]]} wrote a cell that reads badly: {warning}")
    return errors
