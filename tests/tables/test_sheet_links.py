"""A workbook whose link to another workbook is malformed still gives its own cells."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from af.tables import sheet

openpyxl = pytest.importorskip("openpyxl")

REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
LINK_PART = "xl/externalLinks/externalLink1.xml"


def _workbook(path: Path) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["EXAMPLE/1/2020", "<80", 160])
    ws.append(["EXAMPLE/2/2020", 320, None])
    wb.save(path)
    return path


def _with_bad_link(good: Path, path: Path) -> Path:
    """A copy of ``good`` that refers to another workbook, the cached copy of which is not
    well-formed XML (an unescaped '<' in a cached value, as a real lab file has)."""
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(path, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "xl/workbook.xml":
                ref = f'<externalReferences><externalReference xmlns:r="{REL}" r:id="rId99"/>'
                data = data.replace(
                    b"</sheets>", b"</sheets>" + ref.encode() + b"</externalReferences>"
                )
            elif item.filename == "xl/_rels/workbook.xml.rels":
                rel = (
                    f'<Relationship Id="rId99" Type="{REL}/externalLink" '
                    'Target="externalLinks/externalLink1.xml"/>'
                )
                data = data.replace(b"</Relationships>", rel.encode() + b"</Relationships>")
            dst.writestr(item, data)
        dst.writestr(LINK_PART, b"<externalLink><cell><v><80</v></cell></externalLink>")
    return path


def test_malformed_link_is_not_read(tmp_path: Path) -> None:
    good = _workbook(tmp_path / "good.xlsx")
    linked = _with_bad_link(good, tmp_path / "linked.xlsx")
    with pytest.raises(Exception):  # noqa: B017, PT011  openpyxl's own default cannot open it
        openpyxl.load_workbook(linked, data_only=True, read_only=True)
    assert [s.rows for s in sheet.load(linked)] == [s.rows for s in sheet.load(good)]


def test_cells_are_read_as_before(tmp_path: Path) -> None:
    (only,) = sheet.load(_workbook(tmp_path / "good.xlsx"))
    assert only.name == "Sheet1"
    assert only.rows == [["EXAMPLE/1/2020", "<80", "160"], ["EXAMPLE/2/2020", "320", ""]]
