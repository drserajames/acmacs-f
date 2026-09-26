"""Lab id shapes: an antigen or serum id of the wrong shape is a flag, not a silent value.

CNIC's 2021-22 B/Victoria tables carried 6-digit serum codes in the reference antigens' id
column (Sarah, 26 Sep 2026: year-serial for antigens, 6-digit for sera), found only by
reading git history. ``id_shapes.tsv`` says, per lab, field (antigen or serum) and test-date
range, which shapes an id may take; older formats get dated rows of their own. Every id is
checked against the rows for its lab, field and table date: no matching row is flagged, a
missing id is flagged separately, and a lab or field with no rows is not checked.

The flags go to the run report, not into the tables: they describe the source, and a table's
content (and hashes) should not change because a check was added.
"""

from __future__ import annotations

from collections import Counter

from .model import Table
from .rules import Rules


def check(tables: list[Table], rules: Rules) -> tuple[Counter[str], list[str]]:
    """(counts by lab/field/outcome, one line per flagged id)."""
    counts: Counter[str] = Counter()
    lines: list[str] = []
    for table in tables:
        for field, values in (
            (
                "antigen",
                [(a.name, _strip(a.lab_ids[0], "#") if a.lab_ids else "") for a in table.antigens],
            ),
            ("serum", [(s.name, _strip(s.serum_id, " ")) for s in table.sera]),
        ):
            in_force = [
                r
                for r in rules.id_shapes.rules
                if rules.id_shapes.in_scope(r, lab=table.lab)
                and r["field"] == field
                and r["date_from"] <= table.date <= r["date_to"]
            ]
            if not in_force:
                continue
            for name, value in values:
                key = f"{table.lab} {field}"
                if not value:
                    counts[f"{key} id missing"] += 1
                    lines.append(f"{table.table_id}: {field} {name} has no id")
                    continue
                rule = next((r for r in in_force if r.matches(value)), None)
                if rule is None:
                    counts[f"{key} id of an unexpected shape"] += 1
                    lines.append(
                        f"{table.table_id}: {field} {name} id {value!r} fits no id_shapes row"
                    )
                else:
                    rule.hits += 1
                    counts[f"{key} ok"] += 1
    return counts, lines


def _strip(value: str, sep: str) -> str:
    """ "CNIC#2021-12345" -> "2021-12345"; "CNIC 300142" -> "300142"."""
    return value.split(sep, 1)[1] if sep in value else value
