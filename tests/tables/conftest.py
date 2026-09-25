"""A synthetic rules directory for the table tests.

The real rule tables live in the private acmacs-f-data repo (``rules/tables/``). These
are the same shapes with invented evidence, enough for the readers to run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables.rules import Rules

META = "\tinvented for tests\ttest\t2030-01-01\t"

RULES: dict[str, list[str]] = {
    "titre_tokens": [
        "lab\tassay\tkind\tpattern\ttitre\tevidence\tadded_by\tadded_on\toptional",
        "CDC\tHINT\tregex\t[0-9]\t<10" + META,
        "CDC\t*\texact\t5\t<10" + META,
        "CDC\tHI\texact\tUnable to test\t*" + META + "yes",
    ],
    "control_sera": [
        "lab\tfield\tkind\tpattern\taction\tvalue\tevidence\tadded_by\tadded_on\toptional",
        "CDC\tlot\tregex\t.*POOL.*\tdrop\thuman pool" + META,
        "CDC\tlot\tregex\t[0-9]{2}MouseS[0-9]+\tspecies\tMOUSE" + META + "yes",
    ],
    "table_defaults": [
        "lab\tsubtype\tassay\trbc\tevidence\tadded_by\tadded_on\toptional",
        "CDC\tA(H3N2)\tHI\tguinea-pig" + META,
        "CDC\t*\tHI\tturkey" + META + "yes",
        "CDC\t*\tHINT\t-" + META + "yes",
        "CDC\t*\tFRA\t-" + META + "yes",
    ],
    "reassortants": [
        "lab\tkind\tpattern\tcanonical\tevidence\tadded_by\tadded_on\toptional",
        "*\tregex\tB?X-?([0-9]+[A-Z]?(?:-CL)?)\tNYMC-\\1" + META + "yes",
        "*\tregex\t(IVR|CVR)-?([0-9]+[A-Z]?)\t\\1-\\2" + META + "yes",
        "*\tregex\tCNIC-?([0-9]+[A-Z]?)\tCNIC-\\1" + META + "yes",
    ],
    "passage_tokens": [
        "lab\tkind\tpattern\tcanonical\tclass\tevidence\tadded_by\tadded_on\toptional",
        *(
            f"*\texact\t{token}\t{canonical}\t{cls}" + META + "yes"
            for token, canonical, cls in [
                ("C", "MDCK", "cell"),
                ("S", "SIAT", "cell"),
                ("QMC", "QMC", "cell"),
                ("HCK", "HCK", "cell"),
                ("AX4", "AX4", "cell"),
                ("M", "MK", "cell"),
                ("NC", "NC", "unknown"),
                ("E", "E", "egg"),
                ("SPF", "SPF", "egg"),
                ("D", "D", "egg"),
                ("X", "X", "unknown"),
            ]
        ),
    ],
    "control_antigens": [
        "lab\tkind\tpattern\taction\tevidence\tadded_by\tadded_on\toptional",
        "CDC\tregex\tKIT-[0-9]+ .*\tdrop" + META + "yes",
    ],
    "strain_aliases": [
        "lab\tsubtype\tapplies_to\tkind\tpattern\tcanonical\tmin_titre\tmin_fraction"
        "\tevidence\tadded_by\tadded_on\toptional",
        "CDC\tA(H3N2)\tantigen\texact\tB/EXAMPLETYPO/7/2029\tA/EXAMPLETYPO/7/2029\t40\t0.5"
        + META
        + "yes",
    ],
    "season_files": [
        "lab\tfile\tsubtype\tdate_from\tdate_to\ttable_key\tevidence\tadded_by\tadded_on\toptional",
        "CDC\tseason.tsv\tH1 swl\t2029-08-01\t2029-08-31\tassay_date+subtype+assay-type"
        + META
        + "yes",
    ],
    "lab_conventions": [
        "lab\tdate_order\tevidence\tadded_by\tadded_on\toptional",
        "CDC\tMDY" + META + "yes",
    ],
}


def write_rules(directory: Path, *, all_optional: bool = False) -> Path:
    """``all_optional`` for tests whose tiny inputs can't exercise every rule, since a
    non-optional rule that matches nothing fails a full re-read (design rule 1)."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, lines in RULES.items():
        if all_optional:
            lines = [
                lines[0],
                *(line if line.endswith("\tyes") else line + "yes" for line in lines[1:]),
            ]
        (directory / f"{name}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


@pytest.fixture
def rules_dir(tmp_path: Path) -> Path:
    return write_rules(tmp_path / "rules")


@pytest.fixture
def rules(rules_dir: Path) -> Rules:
    return Rules(rules_dir)
