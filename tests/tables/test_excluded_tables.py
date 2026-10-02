"""A workbook held out by an excluded_tables rule is not read, and says so in every run."""

from __future__ import annotations

from pathlib import Path

from af.tables.rules import Rules
from af.tables.update import AC21Inputs, _held_out

from .synthetic_rules import write_rules

HEADER = "lab\tkind\tpattern\tevidence\tadded_by\tadded_on\toptional"


def rules_with(tmp_path: Path, row: str) -> Rules:
    directory = write_rules(tmp_path / "rules")
    (directory / "excluded_tables.tsv").write_text(f"{HEADER}\n{row}\n")
    return Rules(directory)


def test_a_held_out_workbook_is_reported_with_its_reason(tmp_path):
    rules = rules_with(
        tmp_path, "LABC\texact\tlabc-20300102.xlsx\tREVISIT: decided so\ttest\t2030-01-01\t"
    )
    files = [tmp_path / "labc-20300101.xlsx", tmp_path / "labc-20300102.xlsx"]
    inputs = AC21Inputs(lab="LABC", dir=tmp_path, start="2030-01-01")
    report: list[str] = []
    kept = _held_out(files, inputs, rules, report)
    assert kept == files[:1]
    (line,) = report
    assert "HELD EXCLUSION labc-20300102.xlsx" in line and "REVISIT: decided so" in line
    assert rules.excluded_tables.unmatched() == []


def test_another_labs_rule_holds_nothing(tmp_path):
    rules = rules_with(tmp_path, "LABX\texact\tlabc-20300102.xlsx\treason\ttest\t2030-01-01\t")
    files = [tmp_path / "labc-20300102.xlsx"]
    report: list[str] = []
    assert (
        _held_out(files, AC21Inputs(lab="LABC", dir=tmp_path, start="2030-01-01"), rules, report)
        == files
    )
    assert report == [] and len(rules.excluded_tables.unmatched()) == 1  # an error in a run


def test_no_file_means_no_exclusions(tmp_path):
    assert Rules(write_rules(tmp_path / "rules")).excluded_tables.rules == []
