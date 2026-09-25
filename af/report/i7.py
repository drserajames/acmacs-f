"""I7: the small JSON beside every report figure, saying what the figure shows.

Figure producers (maps, tree figure, geo) write ``figure.i7.json`` next to ``figure.pdf``.
The report builder checks that the JSON describes the PDF it sits beside (sha256), and the
same-science comparison reads only these JSONs, from both sides: af's figures and the shipped
report's figures converted by :mod:`af.report.compare.reference_ae`. The comparison therefore
never depends on how a figure was drawn.

Unknown top-level keys are errors, so a producer's typo cannot silently drop a field.
The full field list is in ``notes/reports/I7-DRAFT.md`` (draft v1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.util.artefacts import sha256_path

I7_VERSION = 1
I7_NAME = "figure.i7.json"
KINDS = ("map", "tree", "geo", "placeholder")

REQUIRED_TOP = {"i7_version", "kind", "figure", "title", "provenance", "placeholder"}
OPTIONAL_TOP = {"map", "tree", "geo", "notes"}
REQUIRED_FIGURE = {"pdf", "sha256", "pages"}
REQUIRED_MAP = {"chart", "window", "viewport", "clade_scheme", "antigens", "sera", "legend"}
REQUIRED_POINT = {"id", "name", "passage_class", "xy", "shown", "in_viewport", "clade", "colour"}
REQUIRED_TREE = {"subtype", "leaves", "sections", "time_series"}
REQUIRED_LEAF = {"id", "name", "date", "clade", "shown", "order"}
REQUIRED_GEO = {"subtype", "month", "locations"}


class I7Error(ValueError):
    """An I7 file is missing, malformed, or does not match its figure."""


def _need(d: dict[str, Any], keys: set[str], where: str) -> None:
    missing = keys - d.keys()
    if missing:
        raise I7Error(f"{where}: missing {sorted(missing)}")


def validate(doc: dict[str, Any], where: str = "I7") -> None:
    """Raise :class:`I7Error` unless ``doc`` is a well-formed I7 document."""
    _need(doc, REQUIRED_TOP, where)
    unknown = doc.keys() - REQUIRED_TOP - OPTIONAL_TOP
    if unknown:
        raise I7Error(f"{where}: unknown keys {sorted(unknown)}")
    if doc["i7_version"] != I7_VERSION:
        raise I7Error(f"{where}: i7_version {doc['i7_version']} != {I7_VERSION}")
    kind = doc["kind"]
    if kind not in KINDS:
        raise I7Error(f"{where}: kind {kind!r} not in {KINDS}")
    _need(doc["figure"], REQUIRED_FIGURE, f"{where}.figure")
    if kind == "placeholder":
        if doc["placeholder"] is not True:
            raise I7Error(f"{where}: kind placeholder but placeholder is not true")
        return
    body = doc.get(kind)
    if body is None:
        raise I7Error(f"{where}: kind {kind} but no '{kind}' block")
    if kind == "map":
        _need(body, REQUIRED_MAP, f"{where}.map")
        for group in ("antigens", "sera"):
            for i, point in enumerate(body[group]):
                _need(point, REQUIRED_POINT, f"{where}.map.{group}[{i}]")
                if point["shown"] and point["in_viewport"] and not point["colour"]:
                    # A drawn point always has a fill; a null one means the producer lost it.
                    raise I7Error(f"{where}.map.{group}[{i}]: drawn but no colour")
    elif kind == "tree":
        _need(body, REQUIRED_TREE, f"{where}.tree")
        for i, leaf in enumerate(body["leaves"]):
            _need(leaf, REQUIRED_LEAF, f"{where}.tree.leaves[{i}]")
    else:
        _need(body, REQUIRED_GEO, f"{where}.geo")


@dataclass(frozen=True)
class Figure:
    """A figure as the report builder sees it: the PDF, its I7 file, and the parsed I7."""

    pdf: Path
    i7_path: Path
    doc: dict[str, Any]

    @property
    def placeholder(self) -> bool:
        return bool(self.doc["placeholder"])

    @property
    def title(self) -> str:
        return str(self.doc["title"])


def load_figure(i7_path: Path, check_pdf: bool = True) -> Figure:
    """Load an I7 file and, by default, check its PDF exists with the recorded hash.

    A hash mismatch means the PDF was re-rendered without its JSON (or the reverse); a report
    must not embed a figure whose description is of a different drawing.
    """
    if not i7_path.is_file():
        raise I7Error(f"{i7_path}: no such I7 file")
    try:
        doc = json.loads(i7_path.read_text())
    except json.JSONDecodeError as err:
        raise I7Error(f"{i7_path}: not JSON: {err}") from err
    if not isinstance(doc, dict):
        raise I7Error(f"{i7_path}: top level is not an object")
    validate(doc, str(i7_path))
    pdf = (i7_path.parent / doc["figure"]["pdf"]).resolve()
    if check_pdf:
        if not pdf.is_file() or pdf.stat().st_size == 0:
            raise I7Error(f"{i7_path}: figure {pdf} missing or empty")
        actual = sha256_path(pdf)
        if actual != doc["figure"]["sha256"]:
            raise I7Error(
                f"{i7_path}: figure {pdf.name} sha256 {actual[:12]} != recorded "
                f"{str(doc['figure']['sha256'])[:12]} (PDF and I7 out of step)"
            )
    return Figure(pdf=pdf, i7_path=i7_path, doc=doc)
