"""Named renames of strain names a lab got wrong, each checked against the titres.

A rename is curation, so it is data (``strain_aliases.tsv``), counted in every run and
never silent (design rule 9). ae relabelled such names without a word (D-ingestion T40).

A rename must also agree with the assay. Each rule carries ``min_titre`` and
``min_fraction``: after the rename, at least ``min_fraction`` of the renamed antigen's
cells (against the table's sera; for a serum, of its column) must read ``min_titre`` or
more. Example: a virus written "B/..." on an H3 table, renamed "A/...", must actually be
inhibited by the H3 ferret sera. A B virus against H3 antisera reads <10 or 20 at most, so
a threshold of 40 (two twofold dilutions above the <10 floor) on at least half the sera
separates the two. A rule whose titres disagree fails the run, loudly.
"""

from __future__ import annotations

import re

from . import names
from .locations import ChineseLocations
from .model import Table
from .names import Name
from .rules import Rule, Rules

SOURCE_KEY = "alias_rule"  # set in Antigen/Serum.source when a rule renamed it
SERUM_ID_KEY = "serum_id_rule"  # set in Serum.source when a serum_ids rule changed its id


def resolve(
    rules: Rules, raw: str, *, lab: str, subtype: str, applies_to: str
) -> tuple[str, Rule | None]:
    """The name to parse: the rule's canonical spelling if one matches, else ``raw``."""
    rule = rules.strain_aliases.find(raw, lab=lab, subtype=subtype, applies_to=applies_to)
    if rule is None:
        return raw, None
    return rule["canonical"], rule


def check_titres(table: Table, rules: Rules) -> list[str]:
    """Problems with renamed antigens or sera whose titres contradict the rename."""
    by_where = {r.where: r for r in (*rules.strain_aliases.rules, *rules.serum_ids.rules)}
    problems = []
    for no, antigen in enumerate(table.antigens):
        if (where := antigen.source.get(SOURCE_KEY)) is not None:
            problems += _check(table, by_where[where], antigen.name, table.titres[no])
    for no, serum in enumerate(table.sera):
        for key in (SOURCE_KEY, SERUM_ID_KEY):
            if (where := serum.source.get(key)) is not None:
                column = [row[no] for row in table.titres]
                problems += _check(table, by_where[where], serum.name, column)
    return problems


def _check(table: Table, rule: Rule, name: str, cells: list[list[str]]) -> list[str]:
    min_titre, min_fraction = int(rule["min_titre"]), float(rule["min_fraction"])
    tested = [cell for cell in cells if cell]
    reacting = [cell for cell in tested if max(_value(t) for t in cell) >= min_titre]
    if tested and len(reacting) >= min_fraction * len(tested):
        return []
    return [
        f"{table.table_id or table.source_key}: {name} renamed by {rule.where}, but only "
        f"{len(reacting)}/{len(tested)} cells read >= {min_titre} (rule needs {min_fraction:.0%})"
    ]


def _value(titre: str) -> int:
    """A '<N' counts as below N (never reacting); '>N' as N; plain N as N."""
    return 0 if titre.startswith("<") else int(titre.lstrip(">"))


def serum_lot(rules: Rules, raw_lot: str, *, lab: str, ferret: str) -> tuple[str, Rule | None]:
    """A serum id a lab lost or garbled, restored by a ``serum_ids`` rule. The rule applies
    only when its ``ferret`` guard equals the lab's ferret id for the row: the ferret is the
    evidence that this is the same serum (a lot like "No Lot <timestamp>" says nothing)."""
    for rule in rules.serum_ids.rules:
        if (
            rules.serum_ids.in_scope(rule, lab=lab)
            and rule.matches(raw_lot)
            and rule["ferret"] == ferret
        ):
            rule.hits += 1
            return rule["canonical"], rule
    return raw_lot, None


def parse_name(
    rules: Rules,
    raw: str,
    *,
    lab: str,
    subtype: str,
    applies_to: str,
    warnings: list[str],
    locations: ChineseLocations | None = None,
    not_after: int | None = None,
) -> tuple[Name, str | None]:
    """Parse ``raw`` after the lab's rewrites, any named rename and Chinese locations.

    Order: ``name_rewrites`` (systematic spellings, e.g. a lab's "BV/" for "B/"), then a
    ``strain_aliases`` rename (returned, so the titre check can find it), then a Chinese
    location romanised through ``locations``. Every change is recorded as a warning, so it
    shows wherever the table is reviewed. An unmapped Chinese location raises LookupError.
    """
    text = raw
    if (rewrite := rules.name_rewrites.find(raw, lab=lab)) is not None:
        text = re.compile(rewrite["pattern"], re.IGNORECASE).sub(rewrite["replacement"], raw)
        warnings.append(f"name {raw!r} rewritten {text!r} by {rewrite.where}")
    text, rule = resolve(rules, text, lab=lab, subtype=subtype, applies_to=applies_to)
    if locations is not None and text.count("/") >= 3:
        head, location, rest = text.split("/", 2)
        if (romanised := locations.romanise(location)) is not None:
            text = f"{head}/{romanised.text}/{rest}"
            if romanised.source == "locdb name (kept)":
                warnings.append(
                    f"name {raw!r}: Chinese location kept (locationdb names it, no spelling)"
                )
    name = names.parse(text, subtype, rules.reassortants, lab, not_after=not_after)
    warnings.extend(name.problems)
    if rule is None:
        return name, None
    warnings.append(f"name {raw!r} renamed {name.name!r} by {rule.where}")
    return name, rule.where
