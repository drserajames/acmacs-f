"""The matcher's rule tables, read once for geo, stat and maps (af.seq.matching_rules)."""

from pathlib import Path

import pytest

from af.seq.matching import Equivalent, NumberRule, read_passage_rules
from af.seq.matching_rules import TABLES, MatchingRules, matching_rules
from af.seq.passage_match import PassageMatcher
from af.util.artefacts import sha256_path

PASSAGE = "pattern\tclass\treason\nSIAT\tcell\ttest\nMDCK\tcell\ttest\nE\\d\tegg\ttest\n"
LABS_HEADER = "code\tname\treader\tevidence\tadded_by\tadded_on\toptional\n"
SUBMITTERS_HEADER = "lab\tsubmitting_lab\treason\n"
NUMBER_HEADER = "lab\tscope\tevidence\tadded_by\tadded_on\n"
# the shared passage steps (lab "*"), as rules/tables/passage_tokens.tsv writes them
PASSAGE_TOKENS = (
    "lab\tkind\tpattern\tcanonical\tclass\tevidence\tadded_by\tadded_on\toptional\n"
    + "".join(
        f"*\texact\t{pattern}\t{canonical}\t{klass}\ttest\ttest\t2026-01-01\t\n"
        for pattern, canonical, klass in (
            ("C", "MDCK", "cell"),
            ("MDCK", "MDCK", "cell"),
            ("S", "SIAT", "cell"),
            ("SIAT", "SIAT", "cell"),
            ("E", "E", "egg"),
            ("OR", "OR", "original"),
        )
    )
)
EQUIVALENTS_HEADER = (
    "lab\ttable_location\tgisaid_location\tchinese\tcheck\tevidence\tadded_by\tadded_on\toptional\n"
)


def write_af_data(
    root: Path,
    *,
    submitters: str = "LABX\tLab X Institute\ttest\n",
    number: str = "LABY\tprovince\ttest\ttest\t2026-01-01\n",
    equivalents: str = "LABY\tOLDTOWN\tNEWTOWN\t\thand\ttest\ttest\t2026-01-01\t\n",
) -> Path:
    """An invented acmacs-f-data tree with the five tables the matcher reads."""
    files = {
        "rules/sequences/gisaid_passage_classes.tsv": PASSAGE,
        "rules/tables/labs.tsv": LABS_HEADER
        + "LABX\tLab X\tnone yet\ttest\ttest\t2026-01-01\t\n"
        + "LABY\tLab Y\tnone yet\ttest\ttest\t2026-01-01\t\n",
        "rules/sequences/lab_submitters.tsv": SUBMITTERS_HEADER + submitters,
        "rules/sequences/number_rules.tsv": NUMBER_HEADER + number,
        "rules/sequences/location-equivalents.tsv": EQUIVALENTS_HEADER + equivalents,
        "rules/tables/passage_tokens.tsv": PASSAGE_TOKENS,
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def passage_matcher(tmp_path: Path) -> PassageMatcher:
    """The invented shared passage steps, for tests that build a MatchingRules directly."""
    path = tmp_path / "passage_tokens.tsv"
    path.write_text(PASSAGE_TOKENS)
    return PassageMatcher.read(path)


def test_the_tables_load_together_with_their_hashes(tmp_path: Path) -> None:
    root = write_af_data(tmp_path)
    rules = matching_rules(root)
    assert rules.lab_codes == {"LABX", "LABY"}
    assert rules.submitters == {"LABX": frozenset({"Lab X Institute"})}
    assert rules.number_rules == {"LABY": NumberRule("LABY", "province")}
    assert [(e.lab, e.table_location, e.gisaid_location) for e in rules.equivalents] == [
        ("LABY", "OLDTOWN", "NEWTOWN")
    ]
    assert rules.counts() == {
        "passage": 3, "lab_codes": 2, "submitters": 1, "number_rules": 1, "equivalents": 1,
    }  # fmt: skip
    # every file read is in the provenance, by content hash
    assert rules.provenance() == {str(root / p): sha256_path(root / p) for p in TABLES}


def test_a_missing_table_is_fatal_not_a_rule_switched_off(tmp_path: Path) -> None:
    root = write_af_data(tmp_path)
    (root / "rules/sequences/number_rules.tsv").unlink()
    with pytest.raises(FileNotFoundError, match="number_rules.tsv"):
        matching_rules(root)


def test_empty_tables_are_allowed_when_present(tmp_path: Path) -> None:
    """A lab with no number rule yet is a table with no rows, not a missing table."""
    rules = matching_rules(write_af_data(tmp_path, submitters="", number="", equivalents=""))
    assert rules.counts()["number_rules"] == rules.counts()["equivalents"] == 0


@pytest.mark.parametrize(
    "table",
    [
        {"submitters": "labx\tLab X Institute\ttest\n"},
        {"number": "LABZ\tprovince\ttest\ttest\t2026-01-01\n"},
        {"equivalents": "labY\tOLDTOWN\tNEWTOWN\t\thand\ttest\ttest\t2026-01-01\t\n"},
    ],
)
def test_every_table_is_keyed_by_the_readers_lab_codes_exactly(
    tmp_path: Path, table: dict[str, str]
) -> None:
    """A key in another spelling (or a lab no reader produces) would switch its rule off."""
    with pytest.raises(ValueError, match="not table lab codes"):
        matching_rules(write_af_data(tmp_path, **table))


def test_a_value_built_directly_is_checked_too(tmp_path: Path) -> None:
    passages = tmp_path / "passage.tsv"
    passages.write_text(PASSAGE)
    passage = tuple(read_passage_rules(passages))
    matcher = passage_matcher(tmp_path)
    ok = MatchingRules(passage, frozenset({"LABX"}), {}, {}, (), (), matcher)
    assert ok.provenance() == {}
    with pytest.raises(ValueError, match="not table lab codes"):
        elsewhere = (Equivalent("LABQ", "A", "B", False, 3),)
        MatchingRules(passage, frozenset({"LABX"}), {}, {}, elsewhere, (), matcher)
    with pytest.raises(ValueError, match="without passage rules"):
        MatchingRules((), frozenset({"LABX"}), {}, {}, (), (), matcher)
    with pytest.raises(ValueError, match="without lab codes"):
        MatchingRules(passage, frozenset(), {}, {}, (), (), matcher)


def test_the_real_tables_load(af_data: Path) -> None:
    rules = matching_rules(af_data)
    counts = rules.counts()
    assert all(counts[k] > 0 for k in ("passage", "lab_codes", "submitters", "number_rules"))
    assert len(rules.provenance()) == len(TABLES)
