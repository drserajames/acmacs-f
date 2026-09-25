"""Placeholder figures: a one-page PDF saying what will go here, plus an I7 marked placeholder.

They let a report be built end to end before its real figures exist. The builder refuses them
unless the config sets ``allow_placeholders``, and then stamps the cover with the count, so a
placeholder cannot reach a delivered report unnoticed.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from af.report.i7 import I7_NAME, I7_VERSION
from af.util.artefacts import sha256_path


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, lines: list[str], width: int = 800, height: int = 800) -> None:
    """Write a minimal valid one-page PDF showing ``lines`` in Helvetica. No dependencies."""
    text = ["BT", "/F1 28 Tf"]
    y = height / 2 + 20 * len(lines)
    for line in lines:
        text.append(f"1 0 0 1 60 {y:.0f} Tm ({_pdf_escape(line)}) Tj")
        y -= 44
    text.append("ET")
    stream = (f"0.85 g 20 20 {width - 40} {height - 40} re f 0 g\n" + "\n".join(text)).encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
        f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def make(
    root: Path, slot: str, title: str, created: dt.datetime, version: str = "placeholder"
) -> Path:
    """Write a placeholder figure for ``slot`` under the figure root; return its I7 path."""
    if created.tzinfo is None:
        raise ValueError("created needs a time zone")
    directory = root / slot / version
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / "figure.pdf"
    tall = slot.startswith("tree/")  # tree figures are whole pages
    write_pdf(pdf, ["PLACEHOLDER", title, slot], 800, 1130 if tall else 800)
    doc = {
        "i7_version": I7_VERSION,
        "kind": "placeholder",
        "title": title,
        "placeholder": True,
        "figure": {"pdf": pdf.name, "sha256": sha256_path(pdf), "pages": 1},
        "provenance": {"producer": "af.report.placeholder", "created": created.isoformat()},
    }
    i7 = directory / I7_NAME
    i7.write_text(json.dumps(doc, indent=1))
    return i7
