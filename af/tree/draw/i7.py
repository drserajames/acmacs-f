"""The I7 JSON beside a tree figure: what it shows, for the report builder and the comparison.

Format: I7 draft v1 (workstream 11, notes/reports/I7-DRAFT.md). Unknown top-level keys are an
error there, so everything else the figure step knows (label placement measures, filter and
override counts) goes into a separate draw report, not into I7.

The PDF is written first; I7 carries its SHA-256 so the builder can refuse a mismatch.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import af

from .layout import Layout
from .model import DrawTree
from .sections import HzBand
from .timeseries import TimeSeries

I7_VERSION = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_block(
    tree: DrawTree,
    layout: Layout,
    hz: list[HzBand],
    ts: TimeSeries,
    marked_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """The I7 `tree` block. Every leaf is listed, drawn or not (`order` is null if not drawn).

    ``sections`` are the lettered bands (the reference extractor reads the same from the shipped
    trees), each named by its first and last leaf's strain name so the two sides compare like
    for like; the clade brackets are in the draw report.
    """
    order = {int(node): row for row, node in enumerate(layout.leaf_nodes)}
    leaves = []
    for i in tree.leaves():
        leaf_id = str(tree.leaf_id[i])
        leaves.append(
            {
                "id": leaf_id,
                "name": tree.name[i],
                "epi_isl": leaf_id.split("+")[0] if leaf_id.startswith("EPI_ISL") else None,
                "date": tree.date[i],
                "clade": tree.clade[i],
                "shown": i in order,
                "order": order.get(i),
                "marked": leaf_id in marked_ids,
            }
        )
    sections = [
        {
            "clade": band.clade,
            "prefix": band.letter,
            "first_leaf": str(tree.name[layout.leaf_nodes[band.first]]),
            "last_leaf": str(tree.name[layout.leaf_nodes[band.last]]),
            "n_leaves": band.last - band.first + 1,
            # drawn positions: a strain drawn twice (two passages) makes a name ambiguous
            "first_order": band.first,
            "last_order": band.last,
        }
        for band in hz
    ]
    first, last = ts.months[0], ts.months[-1]
    return {
        "subtype": tree.subtype,
        "time_series": {"first": f"{first[0]}-{first[1]:02d}", "last": f"{last[0]}-{last[1]:02d}"},
        "sections": sections,
        "leaves": leaves,
    }


def write_i7(
    pdf: Path,
    title: str,
    tree_info: Mapping[str, Any],
    inputs: Mapping[str, str],
    out: Path | None = None,
    notes: str = "",
) -> Path:
    """Write ``<pdf stem>.i7.json`` beside the PDF. ``inputs``: store item -> content hash."""
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise FileNotFoundError(f"I7 needs the rendered PDF first: {pdf}")
    doc = {
        "i7_version": I7_VERSION,
        "kind": "tree",
        "title": title,
        "placeholder": False,
        "figure": {"pdf": pdf.name, "sha256": _sha256(pdf), "pages": 1},
        "provenance": {
            "producer": f"af.tree.draw {af.__version__}",
            "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "inputs": dict(inputs),
        },
        "tree": dict(tree_info),
        "notes": notes,
    }
    path = out or pdf.with_suffix(".i7.json")
    path.write_text(json.dumps(doc, indent=1))
    return path
