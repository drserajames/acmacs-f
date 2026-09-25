"""All names, ids and sequences here are invented."""

from __future__ import annotations

from pathlib import Path

import pytest

from af.seq import gisaid
from af.seq.dates import Precision
from af.seq.gisaid import PullError, join, parse_defline, read_fasta

DEFLINE = (
    "A/EXAMPLETOWN/7/2021_|_a=EPI_ISL_1_|_b=A / H3N2_|_c=MDCK1_|_d=_|_e=2021-03-17"
    "_|_f=_|_g=_|_h=_|_i=2021-04-01_|_j=Example Lab_|_k=Example Lab_|_l=HA_|_m=4"
    "_|_n=EXAMPLE_HA_|_o=EPI900001_|_p=DNA INSDC_|_q=EXAMPLECLADE_|_"
)


def row(**over: str) -> dict[str, str]:
    base = {
        "Isolate_Id": "EPI_ISL_1",
        "Collection_Date": "2021-03-17",
        "Subtype": "A / H3N2",
        "Lineage": "",
        "Host": "Human",
        "Passage_History": "MDCK1",
        "Location": "Europe / Exampleland / Exampleshire",
        "Originating_Lab": "Example Lab",
        "Submitting_Lab": "Example Lab",
        "Submission_Date": "2021-04-01",
    }
    base.update(over)
    return base


class TestDefline:
    def test_underscore_pipe_separator(self) -> None:
        name, fields = parse_defline(DEFLINE)
        assert name == "A/EXAMPLETOWN/7/2021"
        assert fields["a"] == "EPI_ISL_1"
        assert fields["o"] == "EPI900001"

    def test_bare_pipe_separator_is_also_accepted(self) -> None:
        """GISAID is asked for `_|_` and sometimes returns `|`."""
        name, fields = parse_defline("A/EXAMPLETOWN/7/2021|a=EPI_ISL_1|o=EPI900001")
        assert (name, fields["o"]) == ("A/EXAMPLETOWN/7/2021", "EPI900001")

    def test_leading_gt_is_optional(self) -> None:
        assert parse_defline(">A/EXAMPLETOWN/7/2021|a=EPI_ISL_1")[0] == "A/EXAMPLETOWN/7/2021"


class TestFasta:
    def test_multi_line_sequences(self, tmp_path) -> None:
        path = tmp_path / "in.fas"
        path.write_text(">one\nACGT\nACGT\n>two\nTTTT\n")
        assert list(read_fasta(path)) == [("one", "ACGTACGT"), ("two", "TTTT")]

    def test_trailing_newline_is_not_a_record(self, tmp_path) -> None:
        path = tmp_path / "in.fas"
        path.write_text(">one\nACGT\n\n")
        assert list(read_fasta(path)) == [("one", "ACGT")]

    def test_brotli_fasta_is_read_without_decompressing_to_disk(self, tmp_path) -> None:
        import brotli

        path = tmp_path / "in.fas.br"
        path.write_bytes(brotli.compress(b">one\r\nACGT\r\nAC\r\n>two\nTT\n"))
        assert list(read_fasta(path)) == [("one", "ACGTAC"), ("two", "TT")]
        assert list(tmp_path.iterdir()) == [path]


class TestJoin:
    def test_metadata_comes_from_the_workbook(self) -> None:
        records, counts = join([(DEFLINE, "ACGT")], [row()])
        assert len(records) == 1
        record = records[0]
        assert record.key == ("EPI_ISL_1", "EPI900001")
        assert record.host == "Human"
        assert record.location == "Europe / Exampleland / Exampleshire"
        assert counts.sequences == 1 and counts.isolates == 1

    def test_the_workbooks_date_wins_over_the_deflines(self) -> None:
        """The whole reason this module reads the workbook.

        The defline says 1 January because GISAID expands a partial date; the workbook
        says the submitter stated only a year. Keeping the defline's value would make
        this record indistinguishable from one collected on 1 January.
        """
        records, counts = join(
            [(DEFLINE.replace("e=2021-03-17", "e=2021-01-01"), "ACGT")],
            [row(Collection_Date="2021")],
        )
        date = records[0].collection_date
        assert date is not None
        assert str(date) == "2021"
        assert date.precision is Precision.YEAR
        assert counts.defline_date_differs == 1
        assert counts.date_precision["year"] == 1

    def test_an_agreeing_defline_is_not_counted_as_differing(self) -> None:
        _, counts = join([(DEFLINE, "ACGT")], [row()])
        assert counts.defline_date_differs == 0
        assert counts.date_precision["day"] == 1

    def test_an_unreadable_date_is_flagged_not_defaulted(self) -> None:
        records, counts = join([(DEFLINE, "ACGT")], [row(Collection_Date="unknown")])
        assert records[0].collection_date is None
        assert "date.unreadable" in records[0].problems
        assert counts.unreadable_date == 1

    def test_an_excel_serial_date_is_flagged_and_kept_as_stated(self) -> None:
        """A bare year typed into an Excel date cell: 2024 shows as 1905-07-16."""
        records, counts = join([(DEFLINE, "ACGT")], [row(Collection_Date="1905-07-16")])
        assert str(records[0].collection_date) == "1905-07-16"
        assert "date.excel-serial" in records[0].problems
        assert counts.excel_serial_date == 1

    def test_uracil_is_converted_and_counted(self) -> None:
        """Nextclade drops a sequence containing U from every output while exiting 0."""
        records, counts = join([(DEFLINE, "ACGU")], [row()])
        assert records[0].nucleotides == "ACGT"
        assert counts.uracil_converted == 1

    def test_lower_case_sequence_is_upper_cased(self) -> None:
        records, _ = join([(DEFLINE, "acgt")], [row()])
        assert records[0].nucleotides == "ACGT"

    def test_unknown_and_ambiguous_bases_are_counted_apart(self) -> None:
        """N means "not known"; R and Y state a real ambiguity. QC treats them differently."""
        _, counts = join([(DEFLINE, "ACGTRYNN")], [row()])
        assert (counts.ambiguous_bases, counts.unknown_bases) == (2, 2)

    def test_name_problems_are_counted(self) -> None:
        defline = DEFLINE.replace("A/EXAMPLETOWN/7/2021", "EXAMPLEUNQUALIFIED")
        _, counts = join([(defline, "ACGT")], [row()])
        assert counts.name_problems["name.shape"] == 1

    def test_one_isolate_may_carry_two_sequences(self) -> None:
        """EPI_ISL alone is not unique, which is why the key includes the accession."""
        second = DEFLINE.replace("o=EPI900001", "o=EPI900002")
        records, counts = join([(DEFLINE, "ACGT"), (second, "TTTT")], [row()])
        assert {record.accession for record in records} == {"EPI900001", "EPI900002"}
        assert counts.sequences == 2 and counts.isolates == 1


class TestRefusals:
    def test_a_sequence_without_a_metadata_row_raises(self) -> None:
        with pytest.raises(PullError, match="no metadata row"):
            join([(DEFLINE, "ACGT")], [row(Isolate_Id="EPI_ISL_OTHER")])

    def test_the_message_names_the_missing_records(self) -> None:
        with pytest.raises(PullError, match="EPI_ISL_1"):
            join([(DEFLINE, "ACGT")], [])

    def test_a_defline_cut_before_its_accession_raises(self) -> None:
        """A header broken across lines loses its later fields; the key must not default."""
        truncated = DEFLINE.split("_|_j=")[0]
        with pytest.raises(PullError, match="no segment accession.*EPI_ISL_1"):
            join([(truncated, "ACGT")], [row()])

    def test_a_repeated_key_raises(self) -> None:
        with pytest.raises(PullError, match="more than once"):
            join([(DEFLINE, "ACGT"), (DEFLINE, "TTTT")], [row()])

    def test_a_workbook_row_with_no_sequence_is_not_an_error(self) -> None:
        """The workbook is per-isolate and may legitimately list more than the FASTA."""
        records, counts = join([(DEFLINE, "ACGT")], [row(), row(Isolate_Id="EPI_ISL_2")])
        assert len(records) == 1
        assert counts.isolates == 2


def test_counts_serialise_for_provenance() -> None:
    _, counts = join([(DEFLINE, "ACGU")], [row(Collection_Date="2021-03")])
    as_json = counts.to_json()
    assert as_json["uracil_converted"] == 1
    assert as_json["date_precision"] == {"month": 1}


class Cell:
    """Stands in for an xlrd cell: text is ctype 1, a number 2, an Excel date 3."""

    def __init__(self, value: object, ctype: int = 1) -> None:
        self.value = value
        self.ctype = ctype


class TestWorkbook:
    def test_values_become_text_and_line_breaks_are_joined_and_counted(self) -> None:
        header = ["Isolate_Id", "Passage_History", "Host_Age"]
        cells = [[Cell("EPI_ISL_1"), Cell("details: P1;\ntype: example"), Cell(34.0, 2)]]
        book = gisaid.rows_from_cells(header, cells, Path("example.xls"))
        assert book.rows == [
            {"Isolate_Id": "EPI_ISL_1", "Passage_History": "details: P1; type: example",
             "Host_Age": "34"}
        ]  # fmt: skip
        assert book.line_breaks == {"Passage_History": 1}

    def test_an_excel_date_cell_is_refused(self) -> None:
        """A bare year typed into a date cell is how 2023 becomes 1905-07-15."""
        with pytest.raises(PullError, match="row 2, Collection_Date: an Excel date cell"):
            gisaid.rows_from_cells(["Collection_Date"], [[Cell(2023.0, 3)]], Path("x.xls"))

    def test_a_repeated_column_is_refused(self) -> None:
        with pytest.raises(PullError, match="repeated column"):
            gisaid.rows_from_cells(["Host", "Host"], [], Path("x.xls"))


class TestColumnsAndLabs:
    def test_a_missing_required_column_is_refused(self) -> None:
        bare = row()
        del bare["Host"]
        with pytest.raises(PullError, match=r"no column\(s\) \['Host'\]"):
            join([(DEFLINE, "ACGT")], [bare])

    def test_labs_come_from_the_defline(self) -> None:
        """The workbook has no lab columns; a workbook value must not be read."""
        defline = DEFLINE.replace("j=Example Lab", "j=Example Origin").replace(
            "k=Example Lab", "k=Example Submitter"
        )
        records, _ = join([(defline, "ACGT")], [row(Originating_Lab="WRONG")])
        assert (records[0].originating_lab, records[0].submitting_lab) == (
            "Example Origin",
            "Example Submitter",
        )

    def test_embargo_column_is_optional(self) -> None:
        records, _ = join([(DEFLINE, "ACGT")], [row()])
        assert records[0].embargoed_until == ""
        records, _ = join([(DEFLINE, "ACGT")], [row(Publishing_Embargo_Until="2027-01-01")])
        assert records[0].embargoed_until == "2027-01-01"


def test_a_full_date_that_disagrees_with_the_defline_is_flagged() -> None:
    """Not a precision difference: GISAID's two files name different days."""
    records, counts = join([(DEFLINE, "ACGT")], [row(Collection_Date="2020-05-05")])
    assert str(records[0].collection_date) == "2020-05-05"  # the workbook still wins
    assert "date.defline-conflict" in records[0].problems
    assert (counts.defline_date_differs, counts.defline_date_conflicts) == (1, 1)
