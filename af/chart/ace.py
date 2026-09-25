"""`.ace` read and write.

The format is JSON (spec: ae `doc/ace-format.js`), stored plain or compressed. The
compression is detected from the magic bytes, never from the file name, because the
round's files use `.ace` for xz, brotli and plain JSON alike.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import json
import lzma
import math
from pathlib import Path
from typing import Any

import numpy as np

from af.chart.model import Antigen, Chart, Projection, Serum, Titres, empty_table
from af.chart.titre import Titre

XZ_MAGIC = b"\xfd7zXZ\x00"
GZIP_MAGIC = b"\x1f\x8b"
BZIP2_MAGIC = b"BZh"


class AceError(ValueError):
    pass


# ----------------------------------------------------------------------
# bytes


def decompress(data: bytes) -> bytes:
    if data.startswith(XZ_MAGIC):
        return lzma.decompress(data)
    if data.startswith(GZIP_MAGIC):
        return gzip.decompress(data)
    if data.startswith(BZIP2_MAGIC):
        return bz2.decompress(data)
    if data.lstrip()[:1] == b"{":
        return data
    try:
        import brotli  # brotli has no magic bytes; it is the remaining case
    except ImportError as err:  # pragma: no cover
        raise AceError(
            "data is not xz/gzip/bzip2/JSON and the brotli module is not installed"
        ) from err
    try:
        return brotli.decompress(data)
    except brotli.error as err:
        raise AceError("data is not xz, gzip, bzip2, brotli or JSON") from err


def read_json(path: Path) -> dict[str, Any]:
    data = decompress(Path(path).read_bytes())
    try:
        doc = json.loads(data)
    except json.JSONDecodeError as err:
        raise AceError(f"{path}: not valid JSON after decompression: {err}") from err
    if "c" not in doc:
        raise AceError(f'{path}: no chart ("c") in the document')
    return doc


def content_hash(path: Path) -> str:
    """sha256 of the decompressed JSON, so a chart hashes the same however it is compressed."""
    return hashlib.sha256(decompress(Path(path).read_bytes())).hexdigest()


# ----------------------------------------------------------------------
# JSON -> Chart

_AG_KEYS = {"N", "P", "R", "a", "D", "l"}
_SR_KEYS = {"N", "I", "P", "R", "a", "s"}


def _antigen(d: dict[str, Any]) -> Antigen:
    return Antigen(
        name=d.get("N", ""),
        passage=d.get("P", ""),
        reassortant=d.get("R", ""),
        annotations=tuple(d.get("a", ())),
        date=d.get("D", ""),
        lab_ids=tuple(d.get("l", ())),
        extra={k: v for k, v in d.items() if k not in _AG_KEYS},
    )


def _serum(d: dict[str, Any]) -> Serum:
    return Serum(
        name=d.get("N", ""),
        serum_id=d.get("I", ""),
        passage=d.get("P", ""),
        reassortant=d.get("R", ""),
        annotations=tuple(d.get("a", ())),
        species=d.get("s", ""),
        extra={k: v for k, v in d.items() if k not in _SR_KEYS},
    )


def _sparse_to_layer(rows: list[dict[str, str]]) -> dict[tuple[int, int], Titre]:
    layer: dict[tuple[int, int], Titre] = {}
    for ag, row in enumerate(rows):
        for sr, text in row.items():
            t = Titre.parse(text)
            if not t.is_missing:
                layer[(ag, int(sr))] = t
    return layer


def _dense_to_layer(rows: list[list[str]]) -> dict[tuple[int, int], Titre]:
    layer: dict[tuple[int, int], Titre] = {}
    for ag, row in enumerate(rows):
        for sr, text in enumerate(row):
            t = Titre.parse(text)
            if not t.is_missing:
                layer[(ag, sr)] = t
    return layer


def _titres(t: dict[str, Any], n_ag: int, n_sr: int) -> Titres:
    table = empty_table(n_ag, n_sr)
    if "l" in t:
        cells = _dense_to_layer(t["l"])
    elif "d" in t:
        cells = _sparse_to_layer(t["d"])
    else:
        raise AceError('titre table has neither "l" nor "d"')
    for (ag, sr), titre in cells.items():
        table[ag][sr] = titre
    layers = [
        _sparse_to_layer(layer)
        if isinstance(layer, list) and (not layer or isinstance(layer[0], dict))
        else _dense_to_layer(layer)
        for layer in t.get("L", [])
    ]
    return Titres(table=table, layers=layers)


def _layout(rows: list[list[float]]) -> np.ndarray:
    dim = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), dim), np.nan)
    for i, r in enumerate(rows):
        if len(r) == dim:
            out[i] = [math.nan if v is None else v for v in r]
    return out


_PROJ_KEYS = {"c", "l", "s", "m", "C", "t", "g", "f", "d", "D", "U"}


def _projection(p: dict[str, Any]) -> Projection:
    layout = _layout(p.get("l", []))
    dim = layout.shape[1]
    t = p.get("t")
    transformation = None
    if t is not None:
        if len(t) != dim * dim:
            raise AceError(f"transformation has {len(t)} values for a {dim}-D layout")
        transformation = np.array(t, dtype=float).reshape(dim, dim)
    return Projection(
        layout=layout,
        stress=p.get("s"),
        minimum_column_basis=str(p.get("m", "none")),
        forced_column_bases=np.array(p["C"], dtype=float) if p.get("C") else None,
        transformation=transformation,
        disconnected=tuple(p.get("D", ())),
        unmovable=tuple(p.get("U", ())),
        dodgy_is_regular=bool(p.get("d", False)),
        avidity_adjusts=np.array(p["f"], dtype=float) if p.get("f") else None,
        gradient_multipliers=np.array(p["g"], dtype=float) if p.get("g") else None,
        comment=p.get("c", ""),
        extra={k: v for k, v in p.items() if k not in _PROJ_KEYS},
    )


def chart_from_json(doc: dict[str, Any]) -> Chart:
    c = doc["c"]
    antigens = [_antigen(a) for a in c.get("a", [])]
    sera = [_serum(s) for s in c.get("s", [])]
    titres = _titres(c.get("t", {"d": []}), len(antigens), len(sera))
    chart = Chart(
        info=c.get("i", {}),
        antigens=antigens,
        sera=sera,
        titres=titres,
        forced_column_bases=np.array(c["C"], dtype=float) if c.get("C") else None,
        projections=[_projection(p) for p in c.get("P", [])],
        extra={k: v for k, v in c.items() if k not in {"i", "a", "s", "t", "C", "P"}},
    )
    for proj in chart.projections:
        if proj.layout.shape[0] != chart.n_points:
            raise AceError(
                f"projection layout has {proj.layout.shape[0]} points, chart has {chart.n_points}"
            )
    return chart


def read_chart(path: Path | str) -> Chart:
    return chart_from_json(read_json(Path(path)))


# ----------------------------------------------------------------------
# Chart -> JSON


def _ag_json(a: Antigen) -> dict[str, Any]:
    d: dict[str, Any] = {"N": a.name}
    if a.passage:
        d["P"] = a.passage
    if a.reassortant:
        d["R"] = a.reassortant
    if a.annotations:
        d["a"] = list(a.annotations)
    if a.date:
        d["D"] = a.date
    if a.lab_ids:
        d["l"] = list(a.lab_ids)
    d.update(a.extra)
    return d


def _sr_json(s: Serum) -> dict[str, Any]:
    d: dict[str, Any] = {"N": s.name}
    if s.serum_id:
        d["I"] = s.serum_id
    if s.passage:
        d["P"] = s.passage
    if s.reassortant:
        d["R"] = s.reassortant
    if s.annotations:
        d["a"] = list(s.annotations)
    if s.species:
        d["s"] = s.species
    d.update(s.extra)
    return d


def _layer_sparse(layer: dict[tuple[int, int], Titre], n_ag: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = [{} for _ in range(n_ag)]
    for (ag, sr), t in sorted(layer.items()):
        rows[ag][str(sr)] = str(t)
    return rows


def _titres_json(t: Titres) -> dict[str, Any]:
    n_ag, n_sr = t.shape
    filled = sum(1 for row in t.table for x in row if not x.is_missing)
    out: dict[str, Any] = {}
    if n_ag * n_sr and filled / (n_ag * n_sr) > 0.5:
        out["l"] = [[str(x) for x in row] for row in t.table]
    else:
        out["d"] = [
            {str(j): str(x) for j, x in enumerate(row) if not x.is_missing} for row in t.table
        ]
    if t.layers:
        out["L"] = [_layer_sparse(layer, n_ag) for layer in t.layers]
    return out


def _proj_json(p: Projection) -> dict[str, Any]:
    d: dict[str, Any] = {}
    if p.comment:
        d["c"] = p.comment
    d["l"] = [[] if np.isnan(row).any() else [float(v) for v in row] for row in p.layout]
    if p.stress is not None:
        d["s"] = float(p.stress)
    if p.minimum_column_basis not in ("", "none"):
        d["m"] = p.minimum_column_basis
    if p.forced_column_bases is not None:
        d["C"] = [float(v) for v in p.forced_column_bases]
    if p.transformation is not None:
        d["t"] = [float(v) for v in p.transformation.reshape(-1)]
    if p.gradient_multipliers is not None:
        d["g"] = [float(v) for v in p.gradient_multipliers]
    if p.avidity_adjusts is not None:
        d["f"] = [float(v) for v in p.avidity_adjusts]
    if p.dodgy_is_regular:
        d["d"] = True
    if p.unmovable:
        d["U"] = list(p.unmovable)
    if p.disconnected:
        d["D"] = list(p.disconnected)
    d.update(p.extra)
    return d


def chart_to_json(chart: Chart, created: str = "acmacs-f") -> dict[str, Any]:
    c: dict[str, Any] = {
        "i": chart.info,
        "a": [_ag_json(a) for a in chart.antigens],
        "s": [_sr_json(s) for s in chart.sera],
        "t": _titres_json(chart.titres),
    }
    if chart.forced_column_bases is not None:
        c["C"] = [float(v) for v in chart.forced_column_bases]
    if chart.projections:
        c["P"] = [_proj_json(p) for p in chart.projections]
    c.update(chart.extra)
    return {"  version": "acmacs-ace-v1", "?created": created, "c": c}


def write_chart(chart: Chart, path: Path | str, created: str = "acmacs-f") -> Path:
    """Write xz-compressed JSON, via a temporary file so a crash never leaves half a chart."""
    path = Path(path)
    data = json.dumps(
        chart_to_json(chart, created), separators=(",", ":"), ensure_ascii=False
    ).encode()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(lzma.compress(data, preset=6))
    tmp.replace(path)
    return path
