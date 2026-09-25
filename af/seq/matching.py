"""Which stored sequence belongs with a table antigen.

Two ways in, in this order:

1. **EPI_ISL**, when the lab paired the antigen with a GISAID isolate (CDC does). The pairing
   is the lab's statement, so it wins; a name that disagrees with it is flagged, not used.
2. **Name**, for everything else: the location/isolate/year part of the normalised name
   (:mod:`af.seq.names`), within the datasets of the antigen's subtype. The type prefix is
   left out because GISAID names carry a bare ``A`` where table names carry ``A(H3N2)``.

A name often has several sequences: one virus deposited more than once, typically as the
original specimen and as an egg or cell isolate (about 4,000 shared names per A subtype in
the definitive set). The antigen's passage chooses among them: an egg antigen takes an egg
sequence, a cell antigen a cell sequence, and failing that the original specimen. What
cannot be chosen by those rules is **flagged, never taken silently** (T31): an egg antigen
with only cell sequences is still matched, but flagged doubtful; a tie between different
sequences is not matched at all.

Passage classes of GISAID's free-text passages come from a rule table
(``gisaid_passage_classes.tsv``, acmacs-f-data); table antigens bring their own class from
the table parsers.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from af.store import Store

EGG, CELL, ORIGINAL, MIXED, UNKNOWN = "egg", "cell", "original", "mixed", "unknown"

# Flags. Those in DOUBTFUL make a match doubtful: a caller must not use it unreviewed.
EPI_NOT_IN_STORE = "match.epi-not-in-store"
EPI_NAME_DIFFERS = "match.epi-name-differs"
SEVERAL_ACCESSIONS = "match.several-accessions"
EGG_WITHOUT_EGG_SEQUENCE = "match.egg-antigen-non-egg-sequence"
CELL_FROM_ORIGINAL = "match.cell-antigen-original-sequence"
IDENTICAL_DUPLICATES = "match.identical-duplicates"
AMBIGUOUS = "match.ambiguous"
REASSORTANT = "match.reassortant"
NO_MATCH = "match.none"
DOUBTFUL = frozenset({EPI_NAME_DIFFERS, SEVERAL_ACCESSIONS, EGG_WITHOUT_EGG_SEQUENCE,
                      AMBIGUOUS, REASSORTANT})  # fmt: skip


@dataclass(frozen=True)
class PassageRule:
    pattern: re.Pattern[str]
    passage_class: str
    reason: str


def read_passage_rules(path: Path) -> list[PassageRule]:
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    rules = [
        PassageRule(re.compile(row["pattern"], re.IGNORECASE), row["class"], row["reason"])
        for row in csv.DictReader(lines, delimiter="\t")
    ]
    bad = [r.passage_class for r in rules if r.passage_class not in (EGG, CELL, ORIGINAL)]
    if not rules or bad:
        raise ValueError(f"{path}: no rules, or unknown classes {bad}")
    return rules


def passage_class(text: str, rules: Sequence[PassageRule]) -> str:
    found = {rule.passage_class for rule in rules if rule.pattern.search(text)}
    if EGG in found and CELL in found:
        return MIXED
    for kind in (EGG, CELL, ORIGINAL):
        if kind in found:
            return kind
    return UNKNOWN


def name_key(name: str) -> tuple[str, str, str] | None:
    """location/isolate/year of a normalised name; None for a name of another shape."""
    parts = name.split("/")
    return (parts[1], parts[2], parts[3]) if len(parts) == 4 else None


@dataclass(frozen=True)
class Candidate:
    epi_isl: str
    accession: str
    dataset: str
    name: str
    passage: str
    passage_class: str
    seq_hash: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.epi_isl, self.accession)


@dataclass(frozen=True)
class Match:
    method: str | None  # "epi_isl", "name", or None (no sequence)
    chosen: Candidate | None
    candidates: tuple[Candidate, ...]
    flags: tuple[str, ...] = ()

    @property
    def doubtful(self) -> bool:
        return any(flag in DOUBTFUL for flag in self.flags)


@dataclass
class SequenceIndex:
    """Candidates by EPI_ISL and by name key, over one or more sequence datasets."""

    by_epi: dict[str, list[Candidate]] = field(default_factory=lambda: defaultdict(list))
    by_name: dict[tuple[str, str, str], list[Candidate]] = field(
        default_factory=lambda: defaultdict(list)
    )
    counts: Counter[str] = field(default_factory=Counter)

    def add(self, candidate: Candidate) -> None:
        self.by_epi[candidate.epi_isl].append(candidate)
        key = name_key(candidate.name)
        if key is not None:
            self.by_name[key].append(candidate)

    def match(
        self, name: str, antigen_class: str, *, epi_isl: str = "", reassortant: str = ""
    ) -> Match:
        """The sequence for one antigen; ``antigen_class`` is egg, cell, original or unknown."""
        flags: list[str] = [REASSORTANT] if reassortant else []
        if epi_isl:
            found = self.by_epi.get(epi_isl, [])
            if found:
                result = self._from_epi(found, name, flags)
                self.counts.update(result.flags or ["match.clean"])
                return result
            flags.append(EPI_NOT_IN_STORE)
        key = name_key(name)
        found = self.by_name.get(key, []) if key is not None else []
        result = self._from_name(found, antigen_class, flags)
        self.counts.update(result.flags or ["match.clean"])
        return result

    def _from_epi(self, found: list[Candidate], name: str, flags: list[str]) -> Match:
        if any(name_key(c.name) != name_key(name) for c in found):
            flags.append(EPI_NAME_DIFFERS)
        chosen = _one_sequence(found, flags, SEVERAL_ACCESSIONS)
        return Match("epi_isl", chosen, tuple(found), tuple(flags))

    def _from_name(self, found: list[Candidate], antigen_class: str, flags: list[str]) -> Match:
        if not found:
            return Match(None, None, (), (*flags, NO_MATCH))
        tier = _preferred(found, antigen_class, flags)
        chosen = _one_sequence(tier, flags, AMBIGUOUS)
        return Match("name", chosen, tuple(found), tuple(flags))


def _preferred(found: list[Candidate], antigen_class: str, flags: list[str]) -> list[Candidate]:
    """The candidates of the best passage tier for this antigen."""
    by_class: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in found:
        by_class[candidate.passage_class].append(candidate)
    if antigen_class == EGG:
        if by_class[EGG]:
            return by_class[EGG]
        flags.append(EGG_WITHOUT_EGG_SEQUENCE)
        return by_class[ORIGINAL] or by_class[CELL] or found
    if antigen_class == CELL:
        if by_class[CELL]:
            return by_class[CELL]
        if by_class[ORIGINAL]:
            flags.append(CELL_FROM_ORIGINAL)
            return by_class[ORIGINAL]
        return found
    return by_class[antigen_class] or found


def _one_sequence(tier: list[Candidate], flags: list[str], ambiguous_flag: str) -> Candidate | None:
    """One candidate, or several that carry the same sequence; otherwise none, flagged."""
    if len(tier) == 1:
        return tier[0]
    if len({c.seq_hash for c in tier}) == 1:
        flags.append(IDENTICAL_DUPLICATES)
        return min(tier, key=lambda c: c.key)
    flags.append(ambiguous_flag)
    return None


def index_from_store(
    store: Store, datasets: Iterable[str], passage_rules: Sequence[PassageRule]
) -> SequenceIndex:
    """Candidates from the CURRENT versions of ``sequences/<dataset>``."""
    index = SequenceIndex()
    for dataset in datasets:
        version = store.resolve(store.current("sequences", dataset))
        rows = duckdb.execute(
            "select i.epi_isl, i.accession, i.name, i.passage, s.seq_hash"
            " from read_parquet(?) i join read_parquet(?) s using (epi_isl, accession)",
            [str(version / "isolates" / "*" / "*.parquet"),
             str(version / "sequences" / "*" / "*.parquet")],
        ).fetchall()  # fmt: skip
        for epi_isl, accession, name, passage, seq_hash in rows:
            kind = passage_class(passage, passage_rules)
            index.counts[f"passage.{kind}"] += 1
            index.add(Candidate(epi_isl, accession, dataset, name, passage, kind, seq_hash))
    return index
