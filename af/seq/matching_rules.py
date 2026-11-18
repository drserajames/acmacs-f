"""The rule tables the antigen -> sequence matcher reads, loaded once for every consumer.

Geo, stat and the antigenic maps all join table antigens to sequences through
:mod:`af.seq.matching`, and must join them the same way: a virus drawn in one colour on a map
and another on a geo figure would come from two readings of the rules (Sarah, Q46). So there
is one loader, :func:`matching_rules`, and the consumers take its result rather than a list of
paths each of them could leave out differently.

Every table is required. Leaving one out would switch its rule off silently (no own-lab
tie-break, no number rule), which is a different join, not a smaller configuration. The lab
keys of every table are checked against the lab codes the table readers produce
(``rules/tables/labs.tsv``) when a :class:`MatchingRules` is made, so one keyed by another
spelling of a lab is refused before any store is read. Checks that need the store (a
submitter or a GISAID spelling that no stored sequence has) are the join's:
:func:`af.serology.joins.link_from_store`.

The tree selection config (``config/selection.toml``) is read too, for one thing only: the
``exclude_list`` rules marked ``withhold = true``. Their records do not represent the virus they
are named for, so they must not represent an antigen either: excluding them from the tree alone
left them free to be matched and painted with their fallback clade (OWED 57). Other exclusions
are about the tree, not the sequence, and leave the matcher alone.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from af.seq.matching import (
    Equivalent,
    NumberRule,
    PassageRule,
    check_lab_codes,
    read_lab_submitters,
    read_location_equivalents,
    read_number_rules,
    read_passage_rules,
)
from af.seq.passage_match import PassageMatcher
from af.seq.select import listed_epi_isls, parse_rules
from af.store import ExternalInput
from af.tables.labs import read_lab_codes

#: Where each table lives under the acmacs-f-data root.
PASSAGE_CLASSES = Path("rules/sequences/gisaid_passage_classes.tsv")
LAB_SUBMITTERS = Path("rules/sequences/lab_submitters.tsv")
NUMBER_RULES = Path("rules/sequences/number_rules.tsv")
LOCATION_EQUIVALENTS = Path("rules/sequences/location-equivalents.tsv")
LABS = Path("rules/tables/labs.tsv")
PASSAGE_TOKENS = Path("rules/tables/passage_tokens.tsv")
TABLES = (PASSAGE_CLASSES, LAB_SUBMITTERS, NUMBER_RULES, LOCATION_EQUIVALENTS, LABS, PASSAGE_TOKENS)
#: The tree selection config; the matcher reads only its withholding rules (:func:`read_withheld`).
SELECTION = Path("config/selection.toml")


@dataclass(frozen=True)
class MatchingRules:
    """Everything the matcher is told, checked for consistency among itself.

    ``inputs`` are the files read, with content hashes, for the provenance of whatever the
    join feeds: a figure drawn after a rule is added must not look up-to-date against one
    drawn before. A value built directly (tests) may carry none.
    """

    passage: tuple[PassageRule, ...]
    lab_codes: frozenset[str]
    submitters: Mapping[str, frozenset[str]]  # lab -> the exact GISAID submitting-lab names
    number_rules: Mapping[str, NumberRule]  # lab -> rule
    equivalents: tuple[Equivalent, ...]
    inputs: tuple[ExternalInput, ...]
    # which of a preparation's two named records has its passage (Sarah, Q81, 30 Sep)
    passages: PassageMatcher
    # EPI_ISL -> "<subtype>.<rule id>" of the selection rule withholding it: never a candidate
    withheld: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.passage:
            raise ValueError("matching rules without passage rules")
        if not self.lab_codes:
            raise ValueError("matching rules without lab codes")
        check_lab_codes(dict(self.submitters), self.lab_codes, "lab_submitters")
        check_lab_codes(dict(self.number_rules), self.lab_codes, "number_rules")
        check_lab_codes(
            {row.lab: row for row in self.equivalents}, self.lab_codes, "location equivalents"
        )

    def provenance(self) -> dict[str, str]:
        """Each file read -> its sha256."""
        return {str(item.path): item.sha256 for item in self.inputs}

    def counts(self) -> dict[str, int]:
        """Rows per table, for a report: a table that lost its rows shows as a changed count.

        ``withheld`` (EPI_ISLs) only when some rule withholds: a configuration without one
        reports exactly what it did before the key existed.
        """
        counts = {
            "passage": len(self.passage),
            "lab_codes": len(self.lab_codes),
            "submitters": sum(len(names) for names in self.submitters.values()),
            "number_rules": len(self.number_rules),
            "equivalents": len(self.equivalents),
        }
        if self.withheld:
            counts["withheld"] = len(self.withheld)
        return counts


def matching_rules(af_data: Path) -> MatchingRules:
    """Read the matcher's tables from the acmacs-f-data checkout at ``af_data``.

    A missing file is fatal (design rule 4); so is a table keyed by a lab code the table
    readers do not produce.
    """
    if missing := [str(af_data / p) for p in (*TABLES, SELECTION) if not (af_data / p).is_file()]:
        raise FileNotFoundError(f"matching rule tables missing: {', '.join(missing)}")
    withheld, lists = read_withheld(af_data / SELECTION)
    inputs = tuple(ExternalInput.of(af_data / p) for p in TABLES)
    inputs += tuple(ExternalInput.of(path) for path in lists)
    lab_codes = read_lab_codes(af_data / LABS)
    return MatchingRules(
        passage=tuple(read_passage_rules(af_data / PASSAGE_CLASSES)),
        lab_codes=frozenset(lab_codes),
        submitters=read_lab_submitters(af_data / LAB_SUBMITTERS),
        number_rules=read_number_rules(af_data / NUMBER_RULES, labs=lab_codes),
        equivalents=tuple(read_location_equivalents(af_data / LOCATION_EQUIVALENTS)),
        inputs=inputs,
        passages=PassageMatcher.read(af_data / PASSAGE_TOKENS),
        withheld=withheld,
    )


def read_withheld(selection: Path) -> tuple[dict[str, str], list[Path]]:
    """EPI_ISL -> ``"<subtype>.<rule id>"`` for every withholding rule, and their list files.

    Each subtype table is parsed as the tree selection parses it (:func:`af.seq.select
    .parse_rules`), so a rule means one thing to both. A withholding rule must be an
    ``exclude_list`` whose file exists and names at least one EPI_ISL (design rule 1). An
    EPI_ISL two rules withhold keeps the first, in subtype and config order.

    The provenance records the withholding rules' list files, not ``selection.toml``: the join
    depends on which records are withheld, which those files and the flags fix, and every
    other edit to the tree selection (a date floor, a weekly-tree list) would otherwise look
    like a change to every map and geo figure.
    """
    data = tomllib.loads(selection.read_text())
    withheld: dict[str, str] = {}
    lists: list[Path] = []
    for subtype, table in data.items():
        if not (isinstance(table, dict) and "dataset" in table):
            continue
        for rule in parse_rules(table, selection.parent).rules:
            if not rule.withhold:
                continue
            name = f"{subtype}.{rule.id}"
            if rule.kind != "exclude_list" or rule.file is None or not rule.file.is_file():
                raise ValueError(
                    f"{selection}: {name}: only an exclude_list with its file can withhold"
                )
            if not (listed := listed_epi_isls(rule)):
                raise ValueError(f"{selection}: {name}: withholds nothing")
            lists.append(rule.file)
            for epi_isl in sorted(listed):
                withheld.setdefault(epi_isl, name)
    return withheld, lists
