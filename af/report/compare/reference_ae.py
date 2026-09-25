"""Convert a shipped (ae-built) report's maps and trees into I7 documents: the reference side.

A tool, run by hand against a read-only round folder (or a frozen snapshot of one). It needs
ae's compiled ``ae_backend`` only to bake a named semantic style into per-point colours
(``Chart.semantic_style_to_legacy``, ae's native reproduction of kateri's setFrom); every other
number comes from the exported chart JSON. ``ae_backend`` is imported inside the function, so
importing :mod:`af` never needs ae.

ae semantics reproduced here, cited so they can be checked:

- a point is hidden when its baked plot-spec entry has ``"+": false`` (ae ``-reset`` ``"-": true``);
- a point with no coordinates (disconnected) is not drawn;
- drawn coordinates are the layout times the stored 2x2 ``t = [a, b, c, d]``:
  ``x' = a*x + c*y``, ``y' = b*x + d*y`` (ae cc/chart/v3/transformation.hh:189-193);
- "greyed" (outside the window) means the style's ``o12m``/``o6m`` rule applies to the antigen and
  it is drawn in that rule's colour: antigens with no clade are grey too, and ``-vaccines``
  recolours some old ones after the rule;
- the frame is the style's ``V`` in kateri's recentred frame, shifted back to absolute
  coordinates (ae cc/map-draw/styled-draw.cc:731-760).

Usage::

    python -m af.report.compare.reference_ae ROUND_DIR OUT_DIR h1-cdc:clades bvic-cdc:clades-v2

writes ``OUT_DIR/map.<folder>.<window>.i7.json`` for the windows all, 12m and 6m (ae's style
names ``<variant>``, ``<variant>-12m`` and ``<variant>-6m``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import lzma
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from af.map.vaccines import passage_class as af_passage_class
from af.report.i7 import I7_VERSION
from af.util.artefacts import sha256_path

WINDOW_SUFFIX = {"all": "", "12m": "-12m", "6m": "-6m"}


def window_rules(styles: dict[str, Any], name: str) -> dict[str, str]:
    """Attribute -> fill colour of the style's "outside the window" rules (e.g. ``o12m`` -> grey).

    ae greys old antigens with a modifier selecting ``{"o12m": true}`` (the ``-o12m-grey``
    style). A later modifier can recolour some of them (``-vaccines`` comes after it), and
    antigens with no clade are grey in the base colour anyway. So an antigen counts as greyed
    only when it has the attribute *and* is drawn in that rule's colour.
    """
    return {
        key: str(modifier.get("F", "")).lower()
        for modifier in style_chain(styles, name)
        for key, value in (modifier.get("T") or {}).items()
        if key in ("o6m", "o12m") and value is True
    }


def style_chain(
    styles: dict[str, Any], name: str, seen: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    """Flatten a named style into its modifiers, following ``{"R": "<parent>"}`` references."""
    if name in seen:
        raise ValueError(f"style cycle {(*seen, name)}")
    style = styles.get(name)
    if style is None:
        return []  # ae treats a missing referenced style (e.g. "-new-1": null) as empty
    out: list[dict[str, Any]] = []
    for modifier in style.get("A", []):
        if set(modifier) == {"R"}:
            out.extend(style_chain(styles, modifier["R"], (*seen, name)))
        else:
            out.append(modifier)
    return out


def style_viewport(styles: dict[str, Any], name: str) -> list[float] | None:
    """The last viewport set along the style's references (the -reset family carries it)."""
    found: list[float] | None = None

    def walk(n: str, seen: tuple[str, ...]) -> None:
        nonlocal found
        style = styles.get(n)
        if not style or n in seen:
            return
        for modifier in style.get("A", []):
            if set(modifier) == {"R"}:
                walk(modifier["R"], (*seen, n))
        if "V" in style:
            found = style["V"]

    walk(name, ())
    return found


def colour_labels(styles: dict[str, Any], name: str) -> dict[str, str]:
    """Fill colour -> legend label. A colour shared by two labels becomes "A | B", visibly."""
    labels: dict[str, set[str]] = {}
    for modifier in style_chain(styles, name):
        fill, legend = modifier.get("F"), modifier.get("L")
        if fill and isinstance(legend, dict) and legend.get("t"):
            labels.setdefault(fill.lower(), set()).add(legend["t"])
    return {colour: " | ".join(sorted(v)) for colour, v in labels.items()}


def transform(layout: list[list[float]], t: list[float] | None) -> list[list[float] | None]:
    a, b, c, d = t if t else (1.0, 0.0, 0.0, 1.0)
    out: list[list[float] | None] = []
    for p in layout:
        if not p or len(p) < 2 or any(v is None or math.isnan(v) for v in p[:2]):
            out.append(None)
        else:
            out.append([p[0] * a + p[1] * c, p[0] * b + p[1] * d])
    return out


def absolute_viewport(xy: list[list[float] | None], v: list[float] | None) -> list[float]:
    pts = [p for p in xy if p is not None]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    rx, ry = math.ceil(max(xs) - min(xs) + 1.0), math.ceil(max(ys) - min(ys) + 1.0)
    if v:
        return [-rx / 2 + v[0] + cx, -ry / 2 + v[1] + cy, v[2], v[3]]
    return [cx - rx / 2, cy - ry / 2, rx, ry]


def passage_class(entry: dict[str, Any]) -> str | None:
    """af's rule (:func:`af.map.vaccines.passage_class`), applied to the shipped chart's passage.

    Not ae's ``T.p`` attribute: af classes a passage as egg if ANY step was in eggs (DECISIONS
    25 Sep), ae by the last step only. With one rule on both sides, a preparation keys the same
    in the comparison. ``None`` means "not a passage" (specimen ids, blanks), as af records it.
    """
    return af_passage_class(entry.get("P", ""), entry.get("R", ""))


def designation(entry: dict[str, Any]) -> str:
    """Name + reassortant + annotations + passage (+ serum id): identity within one chart."""
    parts = [entry["N"]]
    if entry.get("R"):
        parts.append(entry["R"])
    parts.extend(entry.get("a", []))
    for key in ("P", "I"):
        if entry.get(key):
            parts.append(entry[key])
    return " ".join(parts)


def map_i7(
    chart_file: Path, style: str, pdf: Path | None, chart_label: str, window: str,
    projection: int = 0,
) -> dict[str, Any]:  # fmt: skip
    """The I7 document for one shipped map: ``chart_file`` rendered with named ``style``."""
    import ae_backend.chart_v3 as chart_v3  # the reference tool only; see module docstring

    chart = chart_v3.Chart(str(chart_file))
    styles = json.loads(chart.export())["c"].get("R", {})
    if style not in styles:
        raise KeyError(f"{chart_file}: no style {style!r}")
    chart.semantic_style_to_legacy(style)
    c = json.loads(chart.export())["c"]
    spec, proj = c["p"], c["P"][projection]
    xy = transform(proj["l"], proj.get("t"))
    viewport = absolute_viewport(xy, style_viewport(styles, style))
    labels = colour_labels(styles, style)
    grey_rules = window_rules(styles, style)
    n_antigens = len(c["a"])

    def point(i: int, entry: dict[str, Any], serum: bool) -> dict[str, Any]:
        plot = spec["P"][spec["p"][i]]
        p = xy[i] if i < len(xy) else None
        inside = p is not None and (
            viewport[0] <= p[0] <= viewport[0] + viewport[2]
            and viewport[1] <= p[1] <= viewport[1] + viewport[3]
        )
        colour = (plot.get("F") or "transparent").lower()  # no fill = outline only
        attrs = entry.get("T", {})
        rec: dict[str, Any] = {
            "id": designation(entry), "name": entry["N"], "passage_class": passage_class(entry),
            "date": entry.get("D"), "xy": p,
            "shown": plot.get("+", True) is not False and p is not None,
            "in_viewport": inside, "colour": colour, "clade": labels.get(colour),
            "greyed": any(attrs.get(key) and colour == fill for key, fill in grey_rules.items()),
            "vaccine": bool(attrs.get("V")),
            "reference": None if serum else bool(entry.get("T", {}).get("R")),
        }  # fmt: skip
        if serum:
            rec["serum_id"] = entry.get("I")
        return rec

    antigens = [point(i, a, False) for i, a in enumerate(c["a"])]
    sera = [point(n_antigens + j, s, True) for j, s in enumerate(c["s"])]
    legend: dict[str, int] = {}
    for p in antigens:
        if p["shown"] and p["in_viewport"] and p["clade"]:
            legend[p["clade"]] = legend.get(p["clade"], 0) + 1
    has_pdf = pdf is not None and pdf.is_file()
    return {
        "i7_version": I7_VERSION,
        "kind": "map",
        "title": styles[style].get("T", {}).get("T", {}).get("t", style),
        "placeholder": False,
        "figure": {
            "pdf": str(pdf) if has_pdf else None,
            "sha256": sha256_path(pdf) if has_pdf and pdf else None,
            "pages": 1,
        },
        "provenance": {
            "producer": "af.report.compare.reference_ae",
            "created": dt.datetime.fromtimestamp(chart_file.stat().st_mtime, dt.UTC).isoformat(),
            "chart": str(chart_file),
            "chart_sha256": sha256_path(chart_file),
            "style": style,
            "projection": projection,
            "stress": proj.get("s"),
        },  # fmt: skip
        "map": {
            "chart": chart_label,
            "window": {"name": window},
            "viewport": viewport,
            "clade_scheme": style,
            "antigens": antigens,
            "sera": sera,
            "legend": [{"clade": k, "count": v} for k, v in sorted(legend.items())],
        },  # fmt: skip
    }


def tjz_leaves(tjz: Path) -> list[dict[str, Any]]:
    """Every leaf of an ae tree file: its name (with ae's hash suffix), date and finest clade."""
    raw = tjz.read_bytes()
    doc = json.loads(lzma.decompress(raw) if raw[:6] == b"\xfd7zXZ\x00" else raw)
    out: list[dict[str, Any]] = []
    stack = [doc["tree"]]
    while stack:  # iterative: trees are deeper than Python's stack
        node = stack.pop()
        children = node.get("t")
        if children:
            stack.extend(reversed(children))
        else:
            tags = list(node.get("L") or [])
            # "clade" stays the last tag (ae's own display choice); "clade_tags" keeps every tag,
            # because the last is sometimes a legacy name coarser than an earlier canonical one.
            out.append({"id": node["n"], "date": node.get("d"),
                        "clade": tags[-1] if tags else None, "clade_tags": tags})  # fmt: skip
    return out


def tree_i7(
    tjz: Path, tal: Path, pdf: Path, subtype: str, tal_draw: Path, work: Path
) -> dict[str, Any]:
    """The I7 document for one shipped tree figure: which leaves it draws, in which order.

    The leaf list is tal-draw's own (``<tree> <out>.names``), through ae's ``make_tree``, which is
    the same code path the report's ``tree/0do`` runs, so the hides, the date filter and the
    ladderized order are what the figure shows. Verified on the Sep 2026 round: re-rendering
    with these settings gives the shipped PDFs pixel for pixel (H1, H3; B/Vic differs only by a
    title). Sections and the time-series window come from ae's ``.tal`` readers.
    """
    from ae.report.trees import make_tree  # the reference tool only; see module docstring
    from ae.tal.section_maps import parse_time_series, sections_for

    names_file = work / f"{tjz.stem}.names"
    make_tree(tjz, tal, names_file, tal_draw=str(tal_draw))
    drawn = names_file.read_text(encoding="utf-8", errors="replace").splitlines()
    order = {name: i for i, name in enumerate(drawn)}
    if len(order) != len(drawn):
        raise ValueError(f"{tjz}: tal-draw listed a leaf twice")
    leaves = []
    for leaf in tjz_leaves(tjz):
        position = order.get(leaf["id"])
        leaves.append({
            "id": leaf["id"], "name": strain_name(leaf["id"]), "date": leaf["date"],
            "clade": leaf["clade"], "clade_tags": leaf["clade_tags"],
            "shown": position is not None, "order": position,
        })  # fmt: skip
    unknown = set(order) - {leaf["id"] for leaf in leaves}
    if unknown:
        raise ValueError(f"{tjz}: tal-draw drew {len(unknown)} leaves the tree file does not have")
    sections = []
    for section in sections_for(tal, str(tjz)):
        first, last = order.get(section["first"]), order.get(section["last"])
        sections.append({
            "clade": section["label"], "prefix": section["prefix"],
            "first_leaf": strain_name(section["first"]), "last_leaf": strain_name(section["last"]),
            # Drawn positions: a strain name can be drawn twice, so names alone can be ambiguous.
            "first_order": first, "last_order": last,
            "n_leaves": (last - first + 1) if first is not None and last is not None else None,
        })  # fmt: skip
    series = parse_time_series(tal)
    return {
        "i7_version": I7_VERSION,
        "kind": "tree",
        "title": f"{subtype} phylogenetic tree (shipped)",
        "placeholder": False,
        "figure": {"pdf": str(pdf), "sha256": sha256_path(pdf), "pages": 1},
        "provenance": {
            "producer": "af.report.compare.reference_ae",
            "created": dt.datetime.fromtimestamp(pdf.stat().st_mtime, dt.UTC).isoformat(),
            "tree": str(tjz),
            "tree_sha256": sha256_path(tjz),
            "tal": str(tal),
            "tal_sha256": sha256_path(tal),
            "tal_draw": str(tal_draw),
            "tal_draw_sha256": sha256_path(tal_draw),
        },  # fmt: skip
        "tree": {
            "subtype": subtype,
            "leaves": leaves,
            "sections": sections,
            "time_series": _drawn_months(series) if series else None,
        },
    }


def _drawn_months(series: tuple[str, str]) -> dict[str, str]:
    """The ``.tal`` time-series end is exclusive: end "2026-10" draws through September 2026
    (ae cc/tal/time-series.cc, following AD). I7 records the first and LAST month drawn."""
    first, end = series
    year, month = int(end[:4]), int(end[5:7])
    year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    return {"first": first, "last": f"{year:04d}-{month:02d}"}


def strain_name(leaf_id: str) -> str:
    """ae leaf id without its ``_<passage>_<hash>`` suffix."""
    from af.report.compare.trees import strain_key

    return strain_key(leaf_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shipped ae maps and trees -> I7 references.")
    parser.add_argument("round_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("maps", nargs="*", metavar="FOLDER:VARIANT")
    parser.add_argument("--chart", default="styled.ace")
    parser.add_argument("--projection", type=int, default=0, help="calibration only")
    parser.add_argument(
        "--tree", action="append", default=[], metavar="SLOT=SUBTYPE,TREE,TAL,PDF",
        help="a tree figure, paths relative to round_dir; e.g. "
        "tree/h3/report=h3,tree/h3.asr.after-2021.tjz,tree/h3.after-2021.tal,"
        "tree/h3.asr.after-2021.pdf",
    )  # fmt: skip
    parser.add_argument("--tal-draw", type=Path, help="ae's tal-draw binary (required with --tree)")
    args = parser.parse_args(argv)
    if not args.maps and not args.tree:
        parser.error("nothing to extract: give FOLDER:VARIANT maps and/or --tree")
    if args.tree and (args.tal_draw is None or not args.tal_draw.is_file()):
        parser.error("--tree needs --tal-draw pointing at the tal-draw binary")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for item in args.maps:
        folder, sep, variant = item.partition(":")
        if not sep or not variant:
            parser.error(f"{item!r}: expected FOLDER:VARIANT")
        chart = args.round_dir / folder / args.chart
        for window, suffix in WINDOW_SUFFIX.items():
            style = variant + suffix
            pdf = args.round_dir / folder / f"out.1.{style}.pdf"
            doc = map_i7(chart, style, pdf, folder, window, args.projection)
            out = args.out_dir / f"map.{folder}.{window}.i7.json"
            out.write_text(json.dumps(doc, indent=1))
            drawn = sum(p["shown"] and p["in_viewport"] for p in doc["map"]["antigens"])
            print(f"{out.name}: {len(doc['map']['antigens'])} antigens, {drawn} drawn in frame",
                  file=sys.stderr)  # fmt: skip
    for item in args.tree:
        slot, sep, spec = item.partition("=")
        parts = spec.split(",")
        if not sep or len(parts) != 4:
            parser.error(f"{item!r}: expected SLOT=SUBTYPE,TREE,TAL,PDF")
        subtype, tree, tal, pdf = parts
        with tempfile.TemporaryDirectory() as work:
            doc = tree_i7(args.round_dir / tree, args.round_dir / tal, args.round_dir / pdf,
                          subtype, args.tal_draw, Path(work))  # fmt: skip
        out = args.out_dir / f"{slot.replace('/', '.')}.i7.json"
        out.write_text(json.dumps(doc))
        leaves = doc["tree"]["leaves"]
        print(f"{out.name}: {len(leaves)} leaves, {sum(x['shown'] for x in leaves)} drawn, "
              f"{len(doc['tree']['sections'])} sections", file=sys.stderr)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
