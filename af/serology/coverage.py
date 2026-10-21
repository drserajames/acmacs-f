"""The serum-coverage addenda's sera and the maps whose two homologous rules disagree.

``python -m af.serology.coverage``: for each subtype in a round's section table (acmacs-f-data
``config/addenda/serum-coverage/<round>.toml``), take every map it lists as the round drew it
(the map figure's chain version, chart and finished layout), give each serum its strain's
clades and amino acids from the stores (:meth:`af.map.colouring.StoreColours.for_sera`), choose
the sera each section shows (:mod:`af.chart.serum_selection`), and compute their circles and
coverage (:mod:`af.chart.serum_circles`).

Sarah, Q228 (21 Oct 2026): "keep both rules, use the best match for the figure but declare the
maps which disagree". The output is a round declaration fragment: one ``[[declaration]]`` per
shown serum whose two rules disagree (af.report.howmade reads them per chain version), and a
Markdown page listing them with their counts, for the round's ``declarations/`` (11 adds both).
Hand declarations about the same pages (``--also``) are appended as given, and counted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from af.chart.ace import content_hash, read_chart
from af.chart.model import Chart
from af.chart.serum_circles import SerumCircle, serum_circles
from af.chart.serum_selection import Section, candidates, select_sera
from af.map.colouring import StoreColours
from af.map.config import ColouringConfig
from af.report.figures import resolve
from af.seq.matching_rules import matching_rules
from af.store import Store, StoreRef
from af.util.artefacts import sha256_path

RULING = "Q228"
KINDS = ("visible", "titre-only", "no-homologous", "neither")
WINDOW = "all"  # coverage maps are drawn on the lab's whole map
DRAWN = "empirical"  # the radius a coverage map draws (af.chart.serum_circles)


class CoverageError(RuntimeError):
    """A map, chart or serum the section table names that the round does not have."""


@dataclass(frozen=True)
class MapInput:
    """One map of the round as its figure records it."""

    chart_name: str
    centre: str
    chain: StoreRef
    chart: Chart
    layout: np.ndarray
    minimum_column_basis: str


@dataclass(frozen=True)
class Shown:
    """A serum an addendum shows, with its circle record."""

    addendum: int
    section: str
    reason: str
    map: MapInput
    record: SerumCircle


def map_input(store: Store, figures: Path, chart_name: str, centre: str) -> MapInput:
    """The chart and finished layout a round's map was drawn with: the chain version and chart
    its figure names (the step the figure's content hash picks out, as af.map.build recorded
    it), and the figure's point positions."""
    doc = resolve(figures, f"map/{chart_name}/{WINDOW}").figure.doc
    chain = dict(doc["provenance"]["inputs"]["chain"])
    chart_hash = chain.pop("sha256")
    chain.pop("until_table", None)
    ref = StoreRef.from_json(chain)
    root = store.resolve(ref)
    steps = json.loads((root / "chain.json").read_text())["steps"]
    found = [
        root / s["directory"] / s["chosen_file"]
        for s in steps
        if content_hash(root / s["directory"] / s["chosen_file"]) == chart_hash
    ]
    if len(found) != 1:
        raise CoverageError(f"{chart_name}: {len(found)} steps of {ref} match the figure's chart")
    chart = read_chart(found[0])
    layout = np.full((chart.n_points, 2), np.nan)
    for kind, offset in (("antigens", 0), ("sera", chart.n_antigens)):
        for point in doc["map"][kind]:
            if point["xy"] is not None:
                layout[offset + int(point["id"][2:])] = point["xy"]
    mcb = chart.projections[0].minimum_column_basis if chart.projections else "none"
    return MapInput(chart_name, centre, ref, chart, layout, mcb)


def shown_sera(
    colours: StoreColours, maps: Sequence[MapInput], config: Mapping[str, Any], since: dt.date
) -> list[Shown]:
    """The sera one addendum shows, section by section, each with its circle record."""
    sections = [Section.from_config(s) for s in config["sections"]]
    residues = {int(r[:-1]) for s in sections for r in s.aa}
    by_key: dict[tuple[str, ...], tuple[MapInput, SerumCircle]] = {}
    cands = []
    for m in maps:
        strains = colours.for_sera(m.chart, residues)
        records = serum_circles(m.chart, m.layout, 2.0, minimum_column_basis=m.minimum_column_basis)
        for c, rec in zip(
            candidates(m.chart, strains, chart_name=m.chart_name, centre=m.centre),
            records,
            strict=True,
        ):
            if c.key in by_key:
                raise CoverageError(f"{m.chart_name}: two sera are {c.key[1:]}")
            by_key[c.key] = (m, rec)
            cands.append(c)
    selection = select_sera(cands, sections, since=since)
    return [
        Shown(int(config["addendum"]), label, chosen.reason, *by_key[chosen.serum.key])
        for label, sera in selection.sections.items()
        for chosen in sera
    ]


def declaration(shown: Shown, md_file: str) -> dict[str, str]:
    r = shown.record
    kind = r.disagreement(drawn=DRAWN)
    assert kind is not None
    return {
        "ruling": RULING,
        "dataset": shown.map.chain.dataset,
        "version": shown.map.chain.version,
        "text": (
            f"serum coverage addendum {shown.addendum}, {shown.section}: {r.designation} "
            f"({kind}): drawn from the best match ({_rule(r.circle, r.coverage)}); "
            f"ae's circle rule gives {_rule(r.other_circle, r.other_coverage)}"
        ),
        "file": md_file,
    }


def _rule(circle: Any, coverage: Any) -> str:
    if circle.empirical is None and coverage.status != "good":
        return "no circle and no colouring"
    titre = coverage.titre or circle.titre or "no titre"
    radius = "no circle" if circle.empirical is None else f"radius {circle.empirical:.2f}"
    sets = (
        f"{len(coverage.within)} within, {len(coverage.outside)} outside"
        if coverage.status == "good"
        else f"no colouring ({coverage.status})"
    )
    return f"titre {titre}, {radius}, {sets}"


def write_fragment(
    shown: Sequence[Shown],
    also: Sequence[Mapping[str, str]],
    toml_path: Path,
    md_path: Path,
    md_file: str,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    """The declaration fragment and its page; returns the counts."""
    disagree = [s for s in shown if s.record.disagreement(drawn=DRAWN) is not None]
    counts: dict[str, Any] = {
        "shown": len(shown),
        "disagree": len(disagree),
        "by_kind": dict(Counter(s.record.disagreement(drawn=DRAWN) for s in disagree)),
        "by_addendum": dict(Counter(f"addendum {s.addendum}" for s in disagree)),
        "hand": len(also),
    }
    entries = [declaration(s, md_file) for s in disagree] + [dict(e) for e in also]
    made_from = ", ".join(f"{k} {v}" for k, v in sources.items())
    head = (
        f"# For declarations.toml: serum-coverage maps whose two homologous rules disagree "
        f"({RULING}), made by af.serology.coverage from {made_from}.\n"
    )
    toml_path.write_text(head + "".join("\n" + _toml(e) for e in entries))
    lines = [
        f"# Serum-coverage maps whose two homologous rules disagree ({RULING})",
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
            f"| {s.addendum} | {s.section} | {s.map.chart_name} | {r.designation} | "
            f"{r.disagreement(drawn=DRAWN)} | {_rule(r.circle, r.coverage)} | "
            f"{_rule(r.other_circle, r.other_coverage)} |"
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--af-data", type=Path, required=True, help="acmacs-f-data checkout")
    parser.add_argument("--clones", type=Path, required=True, help="clade nomenclature clones")
    parser.add_argument("--acmacs-data", type=Path, required=True, help="the user's clade tables")
    parser.add_argument("--figures", type=Path, required=True, help="the round's figure root")
    parser.add_argument("--sections", type=Path, required=True, help="the round's section table")
    parser.add_argument("--since", type=dt.date.fromisoformat, required=True,
                        help="YYYY-MM-DD: sera titrated on or after it")  # fmt: skip
    parser.add_argument("--also", type=Path, help="hand [[declaration]]s to append")
    parser.add_argument("--md-file", required=True, help="the page's path as declarations cite it")
    parser.add_argument("--out-toml", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)
    config = tomllib.loads(args.sections.read_text())["coverage"]
    store = Store.open(args.store)
    colours = StoreColours(
        store,
        ColouringConfig("store", args.acmacs_data, args.clones),
        matching_rules(args.af_data),
    )
    shown: list[Shown] = []
    for cfg in config.values():
        if not isinstance(cfg, dict) or "sections" not in cfg:
            continue  # recency_months and other round-wide keys
        maps = [map_input(store, args.figures, m["chart"], m["centre"]) for m in cfg["maps"]]
        shown += shown_sera(colours, maps, cfg, args.since)
    also = tomllib.loads(args.also.read_text()).get("declaration", []) if args.also else []
    sources = {"sections": f"{args.sections.name} ({sha256_path(args.sections)[:12]})",
               "serology": str(colours.serology)}  # fmt: skip
    counts = write_fragment(shown, also, args.out_toml, args.out_md, args.md_file, sources)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
