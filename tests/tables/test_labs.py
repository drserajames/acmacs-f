"""labs.tsv: the lab codes af's readers can produce."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.tables.labs import read_lab_codes, read_labs

HEADER = "code\tname\treader\tserum_id_print\tevidence\tadded_by\tadded_on\toptional\n"
META = "\tinvented for tests\ttest\t2030-01-01\t\n"


def write(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    path = tmp_path / "labs.tsv"
    path.write_text("# comment\n" + header + "".join(r + META for r in rows))
    return path


def test_reads_codes_in_file_order(tmp_path):
    path = write(
        tmp_path,
        "LABX\tExample lab X\taf.tables.x\twith-lab",
        "LABY\tExample lab Y\tnone yet\tbare",
    )
    assert read_lab_codes(path) == ["LABX", "LABY"]
    assert read_labs(path)[1].reader == "none yet"


def test_a_code_twice_is_refused(tmp_path):
    path = write(tmp_path, "LABX\tA\taf.tables.x\tbare", "LABX\tB\taf.tables.x\tbare")
    with pytest.raises(ValueError, match="listed twice"):
        read_lab_codes(path)


def test_a_file_without_a_code_column_is_refused(tmp_path):
    path = write(tmp_path, "Example lab X\taf.tables.x\tbare", header=HEADER.replace("code\t", ""))
    with pytest.raises(ValueError, match="missing columns"):
        read_lab_codes(path)


def test_lower_case_code_is_refused(tmp_path):
    path = write(tmp_path, "labx\tExample\taf.tables.x\tbare")
    with pytest.raises(ValueError, match="upper case"):
        read_lab_codes(path)


def test_the_real_file(af_data):
    path = af_data / "rules" / "tables" / "labs.tsv"
    if not path.is_file():
        pytest.skip("no labs.tsv in the data repo")
    codes = read_lab_codes(path)
    assert {"CDC", "CNIC", "NIID"} <= set(codes)
    assert {lab.serum_id_print for lab in read_labs(path)} <= {"with-lab", "bare"}


def test_serum_id_print_is_read(tmp_path):
    path = write(tmp_path, "LABX\tExample lab X\taf.tables.x\twith-lab", "LABY\tY\tnone yet\tbare")
    assert [lab.serum_id_print for lab in read_labs(path)] == ["with-lab", "bare"]


def test_an_unknown_serum_id_print_is_refused(tmp_path):
    path = write(tmp_path, "LABX\tExample lab X\taf.tables.x\tplain")
    with pytest.raises(ValueError, match="serum_id_print must be one of"):
        read_labs(path)


def test_the_codes_alone_do_not_need_the_display_column(tmp_path):
    header = HEADER.replace("serum_id_print\t", "")
    path = write(tmp_path, "LABX\tExample lab X\taf.tables.x", header=header)
    assert read_lab_codes(path) == ["LABX"]
    with pytest.raises(ValueError, match="missing columns"):
        read_labs(path)
