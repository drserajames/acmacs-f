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

Between the two, a **location equivalent** (``location-equivalents.tsv``, acmacs-f-data) may
join a lab's own location spelling to GISAID's, one way, keyed by (lab, table location): a
name that found nothing is looked up again under each GISAID spelling the table lists for it,
flagged ``match.location-equivalent``. It reaches deposits by any centre, which the number
rule, keyed on the lab's own deposits, cannot.

Passage classes of GISAID's free-text passages come from a rule table
(``gisaid_passage_classes.tsv``, acmacs-f-data); table antigens bring their own class from
the table parsers.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
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
LOCATION_EQUIVALENT = "match.location-equivalent"
NAME_SPACING = "match.name-spacing"  # same name but for spaces/punctuation in the location
REASSORTANT = "match.reassortant"
NO_MATCH = "match.none"
# How a refused name tie was coloured (Sarah, Q81): every tied candidate gave the same colour,
# so none had to be chosen; or they split and ae's rank chose one (see :func:`ae_ranked`).
TIE_AGREES = "match.tie-agrees"
TIE_RANKED = "match.tie-ranked"
DOUBTFUL = frozenset({EPI_NAME_DIFFERS, SEVERAL_ACCESSIONS, EGG_WITHOUT_EGG_SEQUENCE,
                      AMBIGUOUS, REASSORTANT})  # fmt: skip
# Doubts colouring accepts, flagged and counted (Sarah, Q81 D: "Yes, flagged + counted"): ae uses
# these matches. A match with any other doubt (several accessions, several datasets) is not used.
USABLE_DOUBTS = frozenset({EGG_WITHOUT_EGG_SEQUENCE, REASSORTANT, EPI_NAME_DIFFERS})


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


def location_forms_key(key: tuple[str, str, str]) -> tuple[str, str, str]:
    """A name key with the location's spaces and punctuation dropped.

    Labs and GISAID space and punctuate one location differently ("SHANDONG RENCHENG" /
    "SHANDONGRENCHENG", "COTE DIVOIRE" / "COTE D'IVOIRE"), and af.seq.names deliberately does not
    rewrite location spellings, so the two forms never meet on the exact key. Only the isolate
    number and year stay exact: they are what makes the key a virus rather than a place.
    """
    return (re.sub(r"[^A-Z0-9]", "", key[0].upper()), key[1], key[2])


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
    # A refused name tie (``match.ambiguous``): the candidates it was among, and the one ae's
    # rank would take. Nothing is chosen; colouring may use them (Sarah, Q81).
    tied: tuple[Candidate, ...] = ()
    ranked: Candidate | None = None

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
    # (lab, table location) -> GISAID locations (read_location_equivalents)
    equivalents: dict[tuple[str, str], tuple[str, ...]] = field(default_factory=dict)
    #: The same candidates under :func:`location_forms_key`. Kept beside ``by_name`` rather than
    #: replacing it: the number rule reads ``by_name``'s keys and takes the province from the
    #: location's first word, which a key without spaces could not give.
    by_location_forms: dict[tuple[str, str, str], list[Candidate]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _by_number: dict[str, dict[tuple[str, str, str], list[Candidate]]] | None = None

    def add(self, candidate: Candidate) -> None:
        self._by_number = None
        self.by_epi[candidate.epi_isl].append(candidate)
        key = name_key(candidate.name)
        if key is not None:
            self.by_name[key].append(candidate)
            self.by_location_forms[location_forms_key(key)].append(candidate)

    def match(
        self,
        name: str,
        antigen_class: str,
        *,
        epi_isl: str = "",
        reassortant: str = "",
        lab: str = "",
        passage: str = "",
    ) -> Match:
        """The sequence for one antigen; ``antigen_class`` is egg, cell, original or unknown.

        ``lab`` is the lab whose table the antigen is in; it breaks a name tie only through
        ``submitters``. ``passage`` (the antigen's own) only ranks a tie that stays refused.
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
        own = self.submitters.get(lab, frozenset())
        result = self._from_name(found, antigen_class, flags, own, passage)
        if not found and key is not None:
            result = self._from_location_forms(result, key, antigen_class, own, passage)
        if result.method is None and key is not None and (lab, key[0]) in self.equivalents:
            result = self._from_equivalent(result, key, lab, antigen_class, own, passage)
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
        self,
        found: list[Candidate],
        antigen_class: str,
        flags: list[str],
        own: frozenset[str],
        passage: str = "",
    ) -> Match:
        if not found:
            return Match(None, None, (), (*flags, NO_MATCH))
        tier = _preferred(found, antigen_class, flags)
        mine = [c for c in tier if c.submitting_lab in own]
        if len({c.seq_hash for c in tier}) > 1 and len({c.seq_hash for c in mine}) == 1:
            flags.append(OWN_LAB)  # a tie only the antigen's own lab's submission settles
            tier = mine
        chosen = _one_sequence(tier, flags, AMBIGUOUS)
        if chosen is None:
            ranked = ae_ranked(tier, passage, antigen_class)
            return Match("name", None, tuple(found), tuple(flags), tuple(tier), ranked)
        return Match("name", chosen, tuple(found), tuple(flags))

    def _from_location_forms(
        self,
        result: Match,
        key: tuple[str, str, str],
        antigen_class: str,
        own: frozenset[str],
        passage: str,
    ) -> Match:
        """The same name with the location's spaces and punctuation dropped, when it found none.

        A different spelling of one place, not a different place: the isolate number and year
        still have to match exactly. Measured on the live store (29 Sep 2026), this joins 235
        pairs of name forms, and every one is the same virus; where two forms carry different
        sequences the usual tie rules still refuse them rather than choose.
        """
        found = self.by_location_forms.get(location_forms_key(key), [])
        if not found:
            return result
        flags = [f for f in result.flags if f != NO_MATCH]
        loose = self._from_name(found, antigen_class, flags, own, passage)
        return Match("location-forms", loose.chosen, loose.candidates, (*loose.flags, NAME_SPACING))

    def _from_equivalent(
        self,
        result: Match,
        key: tuple[str, str, str],
        lab: str,
        antigen_class: str,
        own: frozenset[str],
        passage: str,
    ) -> Match:
        """The name again under each GISAID spelling of the lab's location, when it found none."""
        location, isolate, year = key
        found = [
            c
            for gisaid in self.equivalents[(lab, location)]
            for c in self.by_name.get((gisaid, isolate, year), [])
        ]
        if not found:
            return result
        flags = [f for f in result.flags if f != NO_MATCH]
        found_result = self._from_name(found, antigen_class, flags, own, passage)
        return replace(
            found_result, method="equivalent", flags=(*found_result.flags, LOCATION_EQUIVALENT)
        )

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


_STEP = re.compile(r"([A-Z]+)\s*(\d+|X|\?)?")


def passage_steps(text: str) -> tuple[tuple[str, str], ...]:
    """A passage as (name, count) steps, read loosely from free text ("MDCK1/SIAT2", "E3").

    Only for ranking a tie: GISAID passages are free text, and a string this cannot read
    fully still yields its last step, which is what the rank compares.
    """
    return tuple((name, count or "") for name, count in _STEP.findall(text.upper()))


def ae_passage_rank(antigen: str, antigen_class: str, candidate: Candidate) -> int:
    """ae's passage similarity (``SeqdbSelected::filter_name``): lower is closer.

    Identical 0; either unknown 90; same last step (name and count) 10; same last step name 20;
    same egg/non-egg class 50; otherwise 90. The egg test uses af's classes, so a GISAID "C1"
    and a table's "MDCK1" rank as same-class (50) where ae, knowing C is MDCK, ranks them 10.
    """
    mine, theirs = passage_steps(antigen), passage_steps(candidate.passage)
    if mine and mine == theirs:
        return 0
    if not mine or not theirs:
        return 90
    if mine[-1] == theirs[-1]:
        return 10
    if mine[-1][0] == theirs[-1][0]:
        return 20
    if (antigen_class == EGG) == (candidate.passage_class == EGG):
        return 50
    return 90


def ae_ranked(tier: Sequence[Candidate], passage: str, antigen_class: str) -> Candidate | None:
    """The candidate ae would take from a tie: best passage rank, then the lowest EPI_ISL number
    and accession, so the choice never depends on input order (design rule 8; ae's
    std::sort breaks equal ranks arbitrarily). ae's reassortant term is left out: af keys
    candidates by name, and a reassortant's GISAID name never shares a key with its parent's.
    """
    if not tier:
        return None
    return min(
        tier,
        key=lambda c: (
            ae_passage_rank(passage, antigen_class, c),
            epi_order(c.epi_isl),
            c.accession,
        ),
    )


def epi_order(epi_isl: str) -> tuple[int, str]:
    """Sort key for EPI_ISL ids by their number (EPI_ISL_9 before EPI_ISL_10); others last."""
    digits = epi_isl.rsplit("_", 1)[-1]
    return (int(digits), epi_isl) if digits.isdigit() else (1 << 62, epi_isl)


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
        out[row["lab"].strip()].add(row["submitting_lab"])  # the tables' lab code, exactly
    return {lab: frozenset(names) for lab, names in out.items()}


def read_number_rules(path: Path, labs: Iterable[str] | None = None) -> dict[str, NumberRule]:
    """``number_rules.tsv``: lab, scope, evidence: labs whose isolate numbers identify a virus.

    Which labs number their viruses uniquely is a fact about each lab, so it is data with its
    evidence, not a lab written into code. With ``labs`` (the table readers' lab codes, from
    config) every row's lab is checked, as :func:`check_lab_codes` checks the other rule tables.
    """
    lines = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    if missing := [c for c in ("lab", "scope", "evidence") if c not in (reader.fieldnames or [])]:
        raise ValueError(f"{path}: missing columns {missing}")
    rules: dict[str, NumberRule] = {}
    for row in reader:
        lab, scope = (row["lab"] or "").strip(), (row["scope"] or "").strip()
        if not (row["evidence"] or "").strip():
            raise ValueError(f"{path}: {lab}: no evidence")
        if scope not in ("province", "national"):
            raise ValueError(f"{path}: {lab}: scope {scope!r}: province or national")
        if lab in rules:
            raise ValueError(f"{path}: {lab} has two rows")
        rules[lab] = NumberRule(lab, scope)
    if labs is not None:
        check_lab_codes(rules, labs, str(path))
    return rules


def check_lab_codes(table: Mapping[str, object], labs: Iterable[str], what: str) -> None:
    """Every lab a rule table names must be a lab code the tables use, exactly as written.

    A lab key the tables never use (a different case, a typo) makes its rule silently do
    nothing: the matcher looks the antigen's lab up exactly.
    """
    known = set(labs)
    if unknown := sorted(set(table) - known):
        raise ValueError(f"{what}: lab(s) {unknown} are not table lab codes {sorted(known)}")


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


@dataclass(frozen=True)
class Equivalent:
    lab: str
    table_location: str
    gisaid_location: str
    optional: bool
    line: int


def read_location_equivalents(path: Path) -> list[Equivalent]:
    """``location-equivalents.tsv``: lab, table_location, gisaid_location, evidence, optional.

    One way (table -> GISAID). A key may list several GISAID spellings (one lab spelling that
    GISAID holds under two romanisations); a repeated (key, GISAID spelling) is an error.
    """
    with path.open(encoding="utf-8") as handle:
        numbered = [
            (n, line) for n, line in enumerate(handle, 1) if line.strip() and line[0] != "#"
        ]
    reader = csv.DictReader([line for _, line in numbered], delimiter="\t")
    required = ("lab", "table_location", "gisaid_location", "evidence")
    if missing := [c for c in required if c not in (reader.fieldnames or [])]:
        raise ValueError(f"{path}: missing columns {missing}")
    out: list[Equivalent] = []
    seen: set[tuple[str, str, str]] = set()
    for (line, _), row in zip(numbered[1:], reader, strict=True):
        cells = {k: (v or "").strip() for k, v in row.items() if k is not None}
        lab, table, gisaid = cells["lab"], cells["table_location"], cells["gisaid_location"]
        if not (lab and table and gisaid and cells["evidence"]):
            raise ValueError(f"{path}:{line}: lab, table_location, gisaid_location and evidence")
        if table == gisaid:
            raise ValueError(f"{path}:{line}: {table!r} is its own equivalent")
        if (lab, table, gisaid) in seen:
            raise ValueError(f"{path}:{line}: ({lab}, {table}) -> {gisaid} is listed twice")
        seen.add((lab, table, gisaid))
        optional = cells.get("optional", "")
        if optional not in ("", "optional"):
            raise ValueError(f"{path}:{line}: optional is 'optional' or blank, not {optional!r}")
        out.append(Equivalent(lab, table, gisaid, optional == "optional", line))
    return out


def equivalents_table(rows: Iterable[Equivalent]) -> dict[tuple[str, str], tuple[str, ...]]:
    """(lab, table location) -> GISAID spellings, the shape :class:`SequenceIndex` takes."""
    table: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        table[(row.lab, row.table_location)].append(row.gisaid_location)
    return {key: tuple(spellings) for key, spellings in table.items()}


def check_equivalents(
    store: Store, datasets: Iterable[str], rows: Sequence[Equivalent]
) -> dict[tuple[str, str, str], int]:
    """Stored sequences under each row's GISAID spelling; a row with none is an error.

    A spelling GISAID no longer uses (renamed, or a typo in the table) would otherwise do
    nothing, silently. Rows marked optional may match nothing; they are counted, not refused.
    """
    paths = [
        str(store.resolve(store.current("sequences", d)) / "isolates" / "*" / "*.parquet")
        for d in datasets
    ]
    held = Counter(
        dict(
            duckdb.execute(
                "select split_part(name, '/', 2), count(*) from read_parquet(?) group by 1",
                [paths],
            ).fetchall()
        )
    )
    counts = {(r.lab, r.table_location, r.gisaid_location): held[r.gisaid_location] for r in rows}
    idle = [r for r in rows if not held[r.gisaid_location] and not r.optional]
    if idle:
        raise ValueError(
            "location equivalents whose GISAID spelling no stored sequence has: "
            + ", ".join(
                f"line {r.line} ({r.lab}, {r.table_location}) -> {r.gisaid_location}" for r in idle
            )  # fmt: skip
        )
    return counts
