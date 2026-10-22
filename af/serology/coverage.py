"""The serum-coverage addenda (1-3): ``[[coverage]]`` blocks of af.map.build's addenda pass.

Each block is one addendum: the maps it covers, in page order, and its sections (a round's
section table, :mod:`af.chart.serum_selection`). :func:`run_coverage` gives every serum of those
maps its strain's clades and amino acids from the stores (:meth:`af.map.colouring.StoreColours
.for_sera`), chooses the sera each section shows across all the maps at once, computes their
circles and coverage on each map's drawn projection (:mod:`af.chart.serum_circles`), and draws
one figure per shown serum in 08's coverage look (af.map.addenda.render_coverage). The page plan
is the fixed description page, then six maps to a page: cell sera over the egg sera of the same
strain, a centre's maps kept together (:func:`af.chart.serum_selection.pages`), with each
section's caption.

Sarah, Q228 (21 Oct 2026): "keep both rules, use the best match for the figure but declare the
maps which disagree". Each block also writes a round declaration fragment: one
``[[declaration]]`` per shown serum whose two rules draw different coverage maps (af.report.
howmade reads them per chain version), with a Markdown page listing them and their counts, for
the round's ``declarations/`` (11 adds both). Hand declarations about the block's maps
(``declarations = [{ map, ruling, text }]``) are appended, on that map's chain version, and
counted.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.chart.serum_circles import SerumCircle

RULING = "Q228"
KINDS = ("visible", "titre-only", "no-homologous", "neither")
RADIUS = "empirical"  # the radius a coverage map draws, as the Sep 2026 addenda did
FOLD = 2.0
CAPTION = "Ferret serum coverage maps for {label} sera."
LEGEND = "Cell passage serum circles in blue, egg passage serum circles in red"
TEXT_MARGINS = {"top": "5em", "bottom": "5em", "left": "2.5em", "right": "10em"}

#: The description page before the maps (the Sep 2026 addenda's, as plain paragraphs).
DESCRIPTION = (
    "Ferret serum circle description",
    "Ferret serum circles shown in blue are for sera raised against cell passaged-viruses, in "
    "red for sera raised against egg-passaged viruses, in orange for sera raised against "
    "reassortant viruses.",
    "Strains outside a serum circle are >4-fold low-reactors to the homologous titer for the "
    "serum.",
    "Here we describe why serum circles have different radii. One might expect the serum circle "
    "delimiting >4-fold low-reactors to have radius 2. However, this would only be the case "
    "when the homologous titer is the same as the maximum titer for a serum - and this is not "
    "always the case.",
    'The theoretical (or "target") distance in an antigenic map from serum S to a antigen A is '
    "log2 (max titer for serum S against any antigen) - log2 (titer for serum S against antigen "
    "A). In other words, the number of 2-folds between the maximum titer for serum S, and the "
    "titer of serum S to antigen A.",
    "Thus the theoretical distance between a serum and antigen is dependent on both the maximum "
    "titer observed for the serum and its homologous titer.",
    "If low reactors were defined as >4-fold from the max titer for a serum then the "
    "theoretical radius for all serum circles would be 2 units, and this text would not be "
    "necessary. But low reactors are defined as >4-fold from the homologous titer, hence the "
    "radius is 2 units plus the number of 2-folds between max titer and the homologous titer for "
    "a serum. Saying the same thing mathematically the theoretical radius for a serum circle is "
    "2 + log2 (max titer for serum S against any antigen A) - log2 (homologous titer for serum "
    "S).",
    "In addition to the theoretical serum circle radius, we also calculate an empirical radius. "
    "The difference is that the theoretical radius is calculated from the target distance "
    "between sera and antigens as specified by the HI titers, whereas the empirical radius is "
    "determined from the antigenic map and thus the actual distances in the antigenic map "
    "between the sera and antigens. There are some extra details about the empirical "
    "calculation but they are not central, and are omitted here.",
    "The theoretical and empirical radii are similar to each other, and are on average only "
    "about 0.5 antigenic units different. The empirical radius is the one shown on the "
    "antigenic maps.",
    "The center of a serum circle for serum S is at the serum point in the map for serum S.",
)


class CoverageError(RuntimeError):
    """A block the round's maps cannot draw: a map, serum or declaration it names is missing."""


@dataclass(frozen=True)
class Shown:
    """A serum an addendum shows, with its circle record."""

    addendum: int
    section: str
    reason: str
    chart_name: str  # the map folder
    chain: Mapping[str, Any]  # the map's chain input: dataset and version (and the rest as given)
    record: SerumCircle


def run_coverage(block: Mapping[str, Any], pages: Mapping[str, Any], ctx: Any) -> Any:
    """One serum-coverage addendum (af.map.addenda's handler for ``[[coverage]]`` blocks)."""
    from af.chart.serum_circles import serum_circles
    from af.chart.serum_selection import Section, candidates, columns, select_sera
    from af.chart.serum_selection import pages as page_plan
    from af.map.addenda import AddendumResult, block_maps, directory_name, render_coverage

    name = directory_name(block)
    out_dir = ctx.out_root / "addenda" / name
    if str(block.get("radius", RADIUS)) != RADIUS or float(block.get("fold", FOLD)) != FOLD:
        raise CoverageError(f"{name}: coverage draws the {RADIUS} radius at fold {FOLD}")
    sections = [Section.from_config(s) for s in block.get("sections", ())]
    if not sections:
        raise CoverageError(f"{name}: no sections")
    residues = {int(r[:-1]) for s in sections for r in s.aa}
    folders = block_maps(block)
    if None in folders:
        raise CoverageError(f"{name}: coverage pages are laid out by centre; no empty map slots")
    found: dict[tuple[str, ...], tuple[str, SerumCircle, str]] = {}
    cands = []
    for f in folders:
        assert f is not None
        page = pages[f]
        strains = ctx.colours.for_sera(page.chart, residues)
        records = serum_circles(page.chart, page.layout(), FOLD, projection=page.projection)
        mine = candidates(page.chart, strains, chart_name=f, centre=page.centre)
        for cand, rec, strain in zip(mine, records, strains, strict=True):
            found[cand.key] = (f, rec, _clade_line(strain))
        cands += mine
    classes = block.get("passage_classes")
    selection = select_sera(
        cands, sections, since=ctx.since,
        passage_classes=frozenset(classes) if classes is not None else None,
    )  # fmt: skip
    window = str(block.get("window", "all"))
    figure_of: dict[tuple[str, ...], Path] = {}
    shown: list[Shown] = []
    for label, chosen in selection.sections.items():
        for c in chosen:
            f, rec, clade_line = found[c.serum.key]
            page = pages[f]
            lab = ctx.labs.get(str(page.chart.info.get("l", "")))
            if lab is None:
                raise CoverageError(f"{f}: lab {page.chart.info.get('l')!r} not in the labs table")
            shown.append(Shown(int(block["addendum"]), label, c.reason, f, _chain(page), rec))
            figure_of[c.serum.key] = render_coverage(
                page, rec, radius=RADIUS, clade_line=clade_line, created=ctx.created, lab=lab,
                window=window,
                out_pdf=out_dir / f / f"sr{rec.serum}" / "v1" / "figure.pdf",
                addendum={"name": name, "kind": "coverage", "section": label,
                          "reason": c.reason, "rules": [rec.circle.rule, rec.coverage.rule],
                          "radius": RADIUS, "record": rec.to_json()},
            )  # fmt: skip

    def cell(c: Any) -> str | None:
        return None if c is None else str(figure_of[c.key].relative_to(out_dir))

    plan: list[dict[str, Any]] = [{"text": list(DESCRIPTION), "margins": dict(TEXT_MARGINS)}]
    for label, chosen in selection.sections.items():
        for laid in page_plan(columns(chosen)):
            tops = [cell(t) for t, _ in laid] + [None] * (3 - len(laid))
            bottoms = [cell(b) for _, b in laid] + [None] * (3 - len(laid))
            plan.append({"caption": CAPTION.format(label=label), "legend": LEGEND,
                         "cells": tops + bottoms})  # fmt: skip
    hand = [_hand_declaration(name, d, pages, folders) for d in block.get("declarations", ())]
    md_file = f"declarations/{name}-homologous-rules.md"
    toml_path = out_dir / f"{name}-homologous-rules.toml"
    md_path = out_dir / f"{name}-homologous-rules.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {"addendum": name, "serology": str(getattr(ctx.colours, "serology", "?"))}
    counts = write_fragment(shown, hand, toml_path, md_path, md_file, sources)
    manifest = {
        "kind": "coverage",
        "addendum": int(block["addendum"]),
        "subtype": block["subtype"],
        "title": block.get("title", ""),
        "pages": plan,
        "figures": [str(p.relative_to(out_dir)) for p in figure_of.values()],
        "declarations": [toml_path.name, md_path.name],
        "selection": selection.report(),
        "disagreements": counts,
    }
    return AddendumResult(
        out_dir, manifest, tuple(figure_of.values()), selection.report(),
        [_shown_json(s, name) for s in shown],
    )  # fmt: skip


def _clade_line(strain: Any) -> str:
    """The figure's third title line: the serum's strain's clade (its lineage's last), or none
    when no sequence of the strain is known."""
    if strain.clade:
        return str(strain.clade)
    return str(strain.lineage[-1]) if strain.lineage else ""


def _chain(page: Any) -> Mapping[str, Any]:
    """The chain version a map was drawn from: a declaration is made on its dataset and
    version (af.report.howmade quotes it on that version only)."""
    ref = dict(page.chain_ref)
    if not ref.get("dataset") or not ref.get("version"):
        raise CoverageError(f"{page.folder}: its chain input names no dataset and version: {ref}")
    return ref


def _shown_json(s: Shown, name: str) -> dict[str, Any]:
    return {"addendum": name, "section": s.section, "reason": s.reason, "map": s.chart_name,
            "serum": s.record.designation, "chain": dict(s.chain),
            "disagreement": s.record.disagreement(drawn=RADIUS),
            "record": s.record.to_json()}  # fmt: skip


def _hand_declaration(
    name: str, entry: Mapping[str, Any], pages: Mapping[str, Any], folders: Sequence[Any]
) -> dict[str, str]:
    """A hand declaration about one of the block's maps, on the chain version it was drawn
    from: ``{ map, ruling, text }``."""
    if set(entry) != {"map", "ruling", "text"}:
        raise CoverageError(f"{name}: a declaration needs map, ruling and text: {dict(entry)}")
    if entry["map"] not in folders:
        raise CoverageError(f"{name}: declaration about {entry['map']!r}, not one of its maps")
    chain = _chain(pages[entry["map"]])
    return {
        "ruling": str(entry["ruling"]),
        "dataset": str(chain["dataset"]),
        "version": str(chain["version"]),
        "text": str(entry["text"]),
        "file": f"declarations/{name}-homologous-rules.md",
    }


def declaration(
    shown: Shown,
    md_file: str,
    *,
    radius: str = RADIUS,
    kind: str = "serum coverage",
    coverage: bool = True,
) -> dict[str, str]:
    """One shown serum's [[declaration]]: what the figure draws from the best match, and what
    ae's circle rule would have drawn. ``radius``, ``kind`` and ``coverage`` describe the
    figure: a coverage map draws the empirical radius and the colouring, a circle panel the
    theoretical radius alone."""
    r = shown.record
    found = r.disagreement(drawn=radius, coverage=coverage)
    assert found is not None
    return {
        "ruling": RULING,
        "dataset": str(shown.chain["dataset"]),
        "version": str(shown.chain["version"]),
        "text": (
            f"{kind} addendum {shown.addendum}, {shown.section}: {r.designation} ({found}): "
            f"drawn from the best match ({_rule(r.circle, r.coverage, radius, coverage)}); "
            f"ae's circle rule gives {_rule(r.other_circle, r.other_coverage, radius, coverage)}"
        ),
        "file": md_file,
    }


def _rule(circle: Any, cov: Any, radius: str, coverage: bool) -> str:
    """What one rule draws, as a declaration quotes it."""
    value = getattr(circle, radius)
    if value is None and (not coverage or cov.status != "good"):
        return "no circle and no colouring" if coverage else "no circle"
    titre = circle.titre if radius == "empirical" else circle.theoretical_titre
    if coverage:
        titre = cov.titre or titre
    drawn = "no circle" if value is None else f"{radius} radius {value:.2f}"
    parts = [f"titre {titre or 'no titre'}", drawn]
    if coverage:
        parts.append(
            f"{len(cov.within)} within, {len(cov.outside)} outside"
            if cov.status == "good"
            else f"no colouring ({cov.status})"
        )
    return ", ".join(parts)


def write_fragment(
    shown: Sequence[Shown],
    also: Sequence[Mapping[str, str]],
    toml_path: Path,
    md_path: Path,
    md_file: str,
    sources: Mapping[str, str],
    *,
    radius: str = RADIUS,
    kind: str = "serum coverage",
    coverage: bool = True,
) -> dict[str, Any]:
    """The declaration fragment and its page; returns the counts. The one writer for every
    addendum kind: ``radius``, ``kind`` and ``coverage`` describe what its figures draw (the
    serum-circle panels: radius="theoretical", kind="serum circle", coverage=False)."""

    def found(s: Shown) -> str | None:
        return s.record.disagreement(drawn=radius, coverage=coverage)

    disagree = [s for s in shown if found(s) is not None]
    counts: dict[str, Any] = {
        "shown": len(shown),
        "disagree": len(disagree),
        "by_kind": dict(Counter(found(s) for s in disagree)),
        "by_addendum": dict(Counter(f"addendum {s.addendum}" for s in disagree)),
        "hand": len(also),
    }
    entries = [
        declaration(s, md_file, radius=radius, kind=kind, coverage=coverage) for s in disagree
    ] + [dict(e) for e in also]
    made_from = ", ".join(f"{k} {v}" for k, v in sources.items())
    head = (
        f"# For declarations.toml: {kind} maps whose two homologous rules disagree "
        f"({RULING}), made by af.serology.coverage from {made_from}.\n"
    )
    toml_path.write_text(head + "".join("\n" + _toml(e) for e in entries))
    lines = [
        f"# {kind[0].upper()}{kind[1:]} maps whose two homologous rules disagree ({RULING})",
        "",
        'Sarah, 21 Oct 2026: "keep both rules, use the best match for the figure but declare the '
        'maps which disagree". Each map is drawn from the best match (the best-ranked antigen of '
        "the serum's name with a titre, passage matched); ae's circle rule (the smallest radius "
        "among ae's homologous antigens) drew the shipped circles.",
        "",
        f"{counts['disagree']} of {counts['shown']} shown sera disagree: "
        + ", ".join(f"{counts['by_kind'].get(k, 0)} {k}" for k in KINDS)
        + (f"; {len(also)} hand declaration(s) follow the table." if also else "."),
        "",
        "| addendum | section | map | serum | kind | best match (drawn) | ae's circle rule |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in disagree:
        r = s.record
        lines.append(
            f"| {s.addendum} | {s.section} | {s.chart_name} | {r.designation} | {found(s)} | "
            f"{_rule(r.circle, r.coverage, radius, coverage)} | "
            f"{_rule(r.other_circle, r.other_coverage, radius, coverage)} |"
        )
    if also:
        lines += ["", "## Hand declarations", ""]
        lines += [f"- {e['dataset']}@{e['version']} ({e['ruling']}): {e['text']}" for e in also]
    md_path.write_text("\n".join(lines) + "\n")
    return counts


def _toml(entry: Mapping[str, str]) -> str:
    def q(v: str) -> str:
        return json.dumps(v, ensure_ascii=False)  # a TOML basic string is JSON-compatible

    keys = ("ruling", "dataset", "version", "text", "file")
    if set(entry) != set(keys):
        raise CoverageError(f"declaration keys {sorted(entry)}: expected {list(keys)}")
    return "[[declaration]]\n" + "".join(f"{k} = {q(entry[k])}\n" for k in keys)
