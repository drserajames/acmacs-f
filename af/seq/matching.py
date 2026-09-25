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
sequence, a cell antigen a cell sequence, and failing that the original specimen. A tie
left after that is often one virus registered as a separate isolate by each centre that
grew it; when exactly one sequence of the tie was submitted by the antigen's own lab, that
one is taken and flagged ``match.own-lab``. Labs map to GISAID submitters through a named
table (``lab_submitters.tsv``, acmacs-f-data), never a pattern: submitters rename (NIID is
now JIHS) and similar names belong to other institutes. What cannot be chosen by those
rules is **flagged, never taken silently** (T31): an egg antigen with only cell sequences is
still matched, but flagged doubtful; a tie between different sequences is not matched.

Last, for a lab that numbers its viruses uniquely (CNIC: one number per virus per year,
Sarah Q52), a name that found nothing or left a tie may be matched by **number**: the lab's
own deposit with the same isolate number and year, within the same province (the first word
of the location) unless the rule's scope is national. The district spelling is what may
differ. It is taken only when that key has one district among the lab's deposits, flagged
``match.lab-number``; a key with several is flagged ``match.lab-number-collision`` and left.

Passage classes of GISAID's free-text passages come from a rule table
(``gisaid_passage_classes.tsv``, acmacs-f-data); table antigens bring their own class from
the table parsers.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
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
OWN_LAB = "match.own-lab"
LAB_NUMBER = "match.lab-number"
LAB_NUMBER_COLLISION = "match.lab-number-collision"
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
    submitting_lab: str = ""

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


@dataclass(frozen=True)
class NumberRule:
    """Match by (province, isolate number, year) among ``lab``'s own deposits."""

    lab: str
    scope: str = "province"  # or "national": the number is unique across the country

    def key(self, location: str, isolate: str, year: str) -> tuple[str, str, str]:
        if self.scope not in ("province", "national"):
            raise ValueError(f"number rule scope {self.scope!r}: province or national")
        province = location.split(" ")[0] if self.scope == "province" else ""
        return (province, isolate, year)


@dataclass
class SequenceIndex:
    """Candidates by EPI_ISL and by name key, over one or more sequence datasets."""

    by_epi: dict[str, list[Candidate]] = field(default_factory=lambda: defaultdict(list))
    by_name: dict[tuple[str, str, str], list[Candidate]] = field(
        default_factory=lambda: defaultdict(list)
    )
    counts: Counter[str] = field(default_factory=Counter)
    submitters: dict[str, frozenset[str]] = field(default_factory=dict)  # lab -> GISAID names
    number_rules: dict[str, NumberRule] = field(default_factory=dict)  # lab -> rule
    _by_number: dict[str, dict[tuple[str, str, str], list[Candidate]]] | None = None

    def add(self, candidate: Candidate) -> None:
        self._by_number = None
        self.by_epi[candidate.epi_isl].append(candidate)
        key = name_key(candidate.name)
        if key is not None:
            self.by_name[key].append(candidate)

    def match(
        self,
        name: str,
        antigen_class: str,
        *,
        epi_isl: str = "",
        reassortant: str = "",
        lab: str = "",
    ) -> Match:
        """The sequence for one antigen; ``antigen_class`` is egg, cell, original or unknown.

        ``lab`` is the lab whose table the antigen is in; it breaks a name tie only through
        ``submitters``.
        """
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
        result = self._from_name(found, antigen_class, flags, self.submitters.get(lab, frozenset()))
        if result.chosen is None and key is not None and lab in self.number_rules:
            result = self._from_number(result, key, lab, antigen_class)
        self.counts.update(result.flags or ["match.clean"])
        return result

    def _from_epi(self, found: list[Candidate], name: str, flags: list[str]) -> Match:
        if any(name_key(c.name) != name_key(name) for c in found):
            flags.append(EPI_NAME_DIFFERS)
        chosen = _one_sequence(found, flags, SEVERAL_ACCESSIONS)
        return Match("epi_isl", chosen, tuple(found), tuple(flags))

    def _from_name(
        self, found: list[Candidate], antigen_class: str, flags: list[str], own: frozenset[str]
    ) -> Match:
        if not found:
            return Match(None, None, (), (*flags, NO_MATCH))
        tier = _preferred(found, antigen_class, flags)
        mine = [c for c in tier if c.submitting_lab in own]
        if len({c.seq_hash for c in tier}) > 1 and len({c.seq_hash for c in mine}) == 1:
            flags.append(OWN_LAB)  # a tie only the antigen's own lab's submission settles
            tier = mine
        chosen = _one_sequence(tier, flags, AMBIGUOUS)
        return Match("name", chosen, tuple(found), tuple(flags))

    def _from_number(
        self, result: Match, key: tuple[str, str, str], lab: str, antigen_class: str
    ) -> Match:
        """The lab's own deposit with this number: only for a name with no match or a tie."""
        rule = self.number_rules[lab]
        found = self._numbered(lab).get(rule.key(*key), [])
        if not found:
            return result
        flags = [f for f in result.flags if f != NO_MATCH]
        if len({c.name.split("/")[1] for c in found}) > 1:  # districts of the key
            return replace(result, flags=(*result.flags, LAB_NUMBER_COLLISION))
        tier = _preferred(found, antigen_class, flags)
        chosen = _one_sequence(tier, flags, AMBIGUOUS)
        if chosen is None:
            return result  # the number finds the same tie: nothing gained
        flags = [f for f in flags if f != AMBIGUOUS]
        return Match("number", chosen, tuple(found), (*flags, LAB_NUMBER))

    def _numbered(self, lab: str) -> dict[tuple[str, str, str], list[Candidate]]:
        if self._by_number is None:
            self._by_number = {}
        if lab not in self._by_number:
            own, rule = self.submitters.get(lab, frozenset()), self.number_rules[lab]
            table: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
            for key, cands in self.by_name.items():
                for c in cands:
                    if c.submitting_lab in own:
                        table[rule.key(*key)].append(c)
            self._by_number[lab] = dict(table)
        return self._by_number[lab]


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
        isolates = str(version / "isolates" / "*" / "*.parquet")
        # A store without submitters gives none: the own-lab rule then never applies, and
        # check_lab_submitters reports every named submitter as missing.
        columns = {c for (c, *_) in duckdb.execute(
            "describe select * from read_parquet(?)", [isolates]).fetchall()}  # fmt: skip
        submitter = "coalesce(i.submitting_lab, '')" if "submitting_lab" in columns else "''"
        rows = duckdb.execute(
            "select i.epi_isl, i.accession, i.name, i.passage, s.seq_hash, " + submitter +
            " from read_parquet(?) i join read_parquet(?) s using (epi_isl, accession)",
            [isolates, str(version / "sequences" / "*" / "*.parquet")],
        ).fetchall()  # fmt: skip
        for epi_isl, accession, name, passage, seq_hash, submitter in rows:
            kind = passage_class(passage, passage_rules)
            index.counts[f"passage.{kind}"] += 1
            index.add(
                Candidate(epi_isl, accession, dataset, name, passage, kind, seq_hash, submitter)
            )
    return index


def read_lab_submitters(path: Path) -> dict[str, frozenset[str]]:
    """``lab_submitters.tsv``: lab, the exact GISAID submitting-lab name, and a reason."""
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t"))
    if unexplained := [r["submitting_lab"] for r in rows if not (r.get("reason") or "").strip()]:
        raise ValueError(f"{path}: rows without a reason: {unexplained}")
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        out[row["lab"].strip().lower()].add(row["submitting_lab"])
    return {lab: frozenset(names) for lab, names in out.items()}


def check_lab_submitters(
    store: Store, datasets: Iterable[str], submitters: dict[str, frozenset[str]]
) -> None:
    """Every submitter named in the table must submit something in these store datasets.

    A name that matches nothing has been renamed or mistyped, and would silently stop
    breaking ties.
    """
    paths = [
        str(store.resolve(store.current("sequences", d)) / "isolates" / "*" / "*.parquet")
        for d in datasets
    ]
    seen = {
        name
        for (name,) in duckdb.execute(
            "select distinct submitting_lab from read_parquet(?)", [paths]
        ).fetchall()
    }
    missing = sorted(n for names in submitters.values() for n in names if n not in seen)
    if missing:
        raise ValueError(f"lab submitters not in the sequence store: {missing}")
