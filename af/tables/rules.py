"""Curation rules as data: TSV tables in acmacs-f-data, one row per rule, each with evidence.

Why a table and not code: ae kept these rules as regexes in per-directory Python scripts,
applied unanchored and in chain, so duplicated keys never fired and rules could hit inside
longer strings (D-ingestion §0.4). Here:

- matching is whole-field: ``exact`` compares case-folded strings, ``regex`` must match the
  entire field (``re.fullmatch``, case-insensitive);
- the first matching row wins (rows are tried in file order);
- every row counts its hits, and a row with no hits after a full read is reported unless
  its ``optional`` column says ``yes``: a rule that matches nothing is either dead or
  pointing at data that has changed, and both need a person to look.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

REQUIRED = ("evidence", "added_by", "added_on")
WILDCARD = "*"


@dataclass
class Rule:
    table: str  # file stem, e.g. "titre_tokens"
    line: int  # 1-based line in the TSV, for messages
    values: dict[str, str]
    hits: int = 0
    _regex: re.Pattern[str] | None = field(default=None, repr=False)

    def __getitem__(self, key: str) -> str:
        return self.values[key]

    @property
    def optional(self) -> bool:
        return self.values.get("optional", "").lower() in ("yes", "true", "1")

    def matches(self, text: str) -> bool:
        kind = self.values["kind"]
        if kind == "exact":
            return text.casefold() == self.values["pattern"].casefold()
        if kind == "regex":
            if self._regex is None:
                self._regex = re.compile(self.values["pattern"], re.IGNORECASE)
            return self._regex.fullmatch(text) is not None
        raise ValueError(f"{self.where}: unknown kind {kind!r}")

    @property
    def where(self) -> str:
        return f"{self.table}.tsv:{self.line}"


class RuleTable:
    """One TSV of rules. ``scope`` columns (lab, assay, ...) accept ``*`` as a wildcard."""

    def __init__(self, path: Path, scope: tuple[str, ...], required: tuple[str, ...]):
        self.name = path.stem
        self.path = path
        self.scope = scope
        self.rules: list[Rule] = []
        with path.open(newline="", encoding="utf-8") as f:
            numbered = [
                (no, ln)
                for no, ln in enumerate(f, start=1)
                if ln.strip() and not ln.startswith("#")
            ]
        lines = [ln for _, ln in numbered]
        file_line = [no for no, _ in numbered]  # rule messages name the real line in the file
        reader = csv.DictReader(lines, delimiter="\t")
        missing = [c for c in (*scope, *required, *REQUIRED) if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for index, row in enumerate(reader, start=1):
            no = file_line[index]
            values = {k: (v or "").strip() for k, v in row.items() if k is not None}
            if None in row:
                raise ValueError(f"{path}:{no}: more cells than columns")
            empty = [c for c in (*scope, *REQUIRED) if not values[c]]
            if empty:
                raise ValueError(f"{path}:{no}: empty {empty}")
            rule = Rule(table=self.name, line=no, values=values)
            if "kind" in values and values["kind"] == "regex":
                re.compile(values["pattern"])  # fail at load, not at first use
            self.rules.append(rule)

    def in_scope(self, rule: Rule, **scope: str) -> bool:
        return all(
            rule[k] == WILDCARD or rule[k].casefold() == v.casefold() for k, v in scope.items()
        )

    def find(self, text: str, **scope: str) -> Rule | None:
        """First rule in scope whose pattern matches ``text``; counts the hit."""
        for rule in self.rules:
            if self.in_scope(rule, **scope) and rule.matches(text):
                rule.hits += 1
                return rule
        return None

    def lookup(self, **scope: str) -> Rule | None:
        """First rule whose scope columns match (no pattern); counts the hit."""
        for rule in self.rules:
            if self.in_scope(rule, **scope):
                rule.hits += 1
                return rule
        return None

    def unmatched(self) -> list[Rule]:
        return [r for r in self.rules if r.hits == 0 and not r.optional]


class Rules:
    """All rule tables for table ingestion, loaded from ``<data>/rules/tables/``."""

    def __init__(self, directory: Path):
        if not directory.is_dir():
            raise FileNotFoundError(f"rules directory not found: {directory}")
        self.directory = directory
        self.titre_tokens = RuleTable(
            directory / "titre_tokens.tsv",
            scope=("lab", "assay"),
            required=("kind", "pattern", "titre"),
        )
        self.control_sera = RuleTable(
            directory / "control_sera.tsv",
            scope=("lab",),
            required=("field", "kind", "pattern", "action", "value"),
        )
        self.table_defaults = RuleTable(
            directory / "table_defaults.tsv", scope=("lab", "subtype", "assay"), required=("rbc",)
        )
        self.reassortants = RuleTable(
            directory / "reassortants.tsv",
            scope=("lab",),
            required=("kind", "pattern", "canonical"),
        )
        self.passage_tokens = RuleTable(
            directory / "passage_tokens.tsv",
            scope=("lab",),
            required=("kind", "pattern", "canonical", "class"),
        )

        self.control_antigens = RuleTable(
            directory / "control_antigens.tsv",
            scope=("lab",),
            required=("kind", "pattern", "action"),
        )
        self.lab_conventions = RuleTable(
            directory / "lab_conventions.tsv",
            scope=("lab",),
            required=("date_order", "passage_plus"),
        )
        self.strain_aliases = RuleTable(
            directory / "strain_aliases.tsv",
            scope=("lab", "subtype", "applies_to"),
            required=("kind", "pattern", "canonical", "min_titre", "min_fraction"),
        )
        self.name_rewrites = RuleTable(
            directory / "name_rewrites.tsv",
            scope=("lab",),
            required=("kind", "pattern", "replacement"),
        )
        self.flu_types = RuleTable(
            directory / "flu_types.tsv",
            scope=("lab",),
            required=("kind", "pattern", "subtype", "lineage"),
        )
        self.serum_ids = RuleTable(
            directory / "serum_ids.tsv",
            scope=("lab",),
            required=("kind", "pattern", "ferret", "canonical", "min_titre", "min_fraction"),
        )
        self.season_files = RuleTable(
            directory / "season_files.tsv",
            scope=("lab",),
            required=("file", "subtype", "date_from", "date_to", "table_key"),
        )

    def tables(self) -> list[RuleTable]:
        return [
            self.titre_tokens,
            self.control_sera,
            self.table_defaults,
            self.reassortants,
            self.passage_tokens,
            self.control_antigens,
            self.lab_conventions,
            self.strain_aliases,
            self.season_files,
            self.name_rewrites,
            self.flu_types,
            self.serum_ids,
        ]

    def usage_report(self) -> list[str]:
        """One line per rule table, then one line per rule that never matched."""
        lines = []
        for table in self.tables():
            hit = sum(1 for r in table.rules if r.hits)
            lines.append(f"{table.name}: {hit}/{len(table.rules)} rules matched")
            for rule in table.unmatched():
                shown = {k: v for k, v in rule.values.items() if k not in REQUIRED}
                lines.append(f"  UNMATCHED {rule.where} {shown}")
        return lines
