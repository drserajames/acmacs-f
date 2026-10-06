"""Files with an .xlsx name that are not readable workbooks, for the readers' per-file guard."""

from __future__ import annotations

import zipfile
from pathlib import Path


def broken_workbooks(good: Path, *names: str) -> list[Path]:
    """Next to ``good``: an empty file, then a copy of ``good`` whose workbook part is not
    well-formed XML, under the given names (as many as given, cycling the two kinds)."""
    out = []
    for i, name in enumerate(names):
        path = good.parent / name
        if i % 2 == 0:
            path.write_bytes(b"")
        else:
            with zipfile.ZipFile(good) as src, zipfile.ZipFile(path, "w") as dst:
                for item in src.infolist():
                    data = src.read(item.filename)
                    if item.filename == "xl/workbook.xml":
                        data = b"<workbook><sheets><sheet name='x' &bad;/>"
                    dst.writestr(item, data)
        out.append(path)
    return out
