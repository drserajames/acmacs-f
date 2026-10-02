"""The I7 JSON that describes a rendered map (interface I7, workstream 11).

Why beside the renderer and not in it: the report builder never looks inside a PDF. It embeds the
PDF the I7 names (by sha256), and the comparison reads only the I7. So the I7 is written from the
same :class:`Scene` the PDF is drawn from, and only after the PDF exists. It records what each
point is (passage class, vaccine, clade), which the renderer itself never reads.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from af.map.labels import Placed
from af.map.render import GREY
from af.map.style import Scene, ScenePoint
from af.map.viewport import Frame


def drawn_fill(p: ScenePoint) -> str:
    """The fill actually drawn (I7 never has a null colour): sera are outline-only."""
    if p.kind == "serum":
        return "transparent"
    return GREY if (p.greyed or p.colour is None) else p.colour


def check_provenance_inputs(inputs: Any) -> None:
    """Every input a figure was made from is named with its content hash (design rule 5)."""
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("provenance needs 'inputs': {name: {'sha256': ..., ...}}")
    for name, item in inputs.items():
        if not isinstance(item, dict) or len(str(item.get("sha256", ""))) != 64:
            raise ValueError(f"provenance input {name!r} has no sha256")


def i7_document(
    scene: Scene,
    frame: Frame,
    labels: Mapping[str, Placed],
    *,
    chart: str,
    pdf: Path,
    created: dt.datetime,
    provenance: dict[str, Any],
    orientation: dict[str, object] | None = None,
    flags: Sequence[str] = (),
) -> dict[str, Any]:
    """The I7 JSON for a rendered map (eu-23's I7 draft v1). ``created`` must carry a time zone."""
    if created.tzinfo is None:
        raise ValueError("I7 'created' needs a time zone")
    check_provenance_inputs(provenance.get("inputs"))
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    page = frame.page(scene.xy())
    inside = np.asarray((page >= 0).all(axis=1) & (page <= 1).all(axis=1), dtype=np.bool_)

    def point(i: int, p: ScenePoint) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": p.id,
            "name": p.name,
            "passage_class": p.passage_class,
            "date": p.date.isoformat() if p.date else None,
            "xy": [round(p.xy[0], 4), round(p.xy[1], 4)] if p.xy is not None else None,
            "shown": p.shown,
            "in_viewport": bool(inside[i]) if p.xy is not None else False,
            "clade": p.legend,
            "colour": drawn_fill(p),
            "greyed": p.greyed,
            "vaccine": p.vaccine is not None,
            "reference": p.reference,
            "sequenced": p.sequenced,
        }
        if p.hidden_reason:
            d["hidden_reason"] = p.hidden_reason
        if p.kind == "serum":
            d["serum_id"] = p.serum_id
        if p.id in labels:
            d["label"] = labels[p.id].text
        return d

    map_block: dict[str, Any] = {
        "chart": chart,
        "window": {
            "name": scene.window.name,
            "since": scene.window.since.isoformat() if scene.window.since else None,
        },
        # Displayed coordinates grow upwards (Q105); the viewport is [x, y, w, h] with (x, y) the
        # frame's top-left corner as drawn, i.e. its largest y.
        "y_axis": "up",
        "viewport": [frame.x, frame.y, frame.size, frame.size],
        "clade_scheme": scene.scheme,
        "legend": [{"clade": t, "count": n} for t, _, n in scene.legend],
        "antigens": [point(i, p) for i, p in enumerate(scene.points) if p.kind == "antigen"],
        "sera": [point(i, p) for i, p in enumerate(scene.points) if p.kind == "serum"],
    }
    shown_ag = [p for p in scene.points if p.kind == "antigen" and p.shown]
    map_block["colour_coverage"] = {
        "shown_antigens": len(shown_ag),
        "sequenced": sum(p.sequenced for p in shown_ag),
        "painted": sum(p.colour is not None for p in shown_ag),
        "sequenced_unpainted": scene.sequenced_unpainted,
        "unsequenced": sum(not p.sequenced for p in shown_ag),
        "vaccines_recoloured_from_cell": scene.vaccines_recoloured,
        "vaccines_without_cell_preparation": list(scene.vaccines_without_cell),
    }
    if flags:
        # Short notes about what happened while the map was made: a curation rule that refused,
        # a guard that fired. The comparison prints them, so a flagged map is never silently odd.
        map_block["flags"] = list(flags)
    if orientation is not None:
        map_block["orientation"] = orientation
    return {
        "i7_version": 1,
        "kind": "map",
        "title": scene.title,
        "placeholder": False,
        "figure": {"pdf": pdf.name, "sha256": digest, "pages": 1},
        "provenance": {"producer": "af.map.render", "created": created.isoformat(), **provenance},
        "map": map_block,
    }


def write_i7(document: dict[str, Any], out_pdf: Path) -> Path:
    """Write ``<pdf stem>.i7.json`` beside the PDF, after checking the PDF hash still matches."""
    digest = hashlib.sha256(out_pdf.read_bytes()).hexdigest()
    if document["figure"]["sha256"] != digest:
        raise RuntimeError(f"{out_pdf} changed after its I7 was built")
    target = out_pdf.with_name(out_pdf.stem + ".i7.json")
    target.write_text(json.dumps(document, indent=1, allow_nan=False) + "\n")
    return target
