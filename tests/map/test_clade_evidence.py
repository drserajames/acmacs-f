"""Maps under tree-clade evidence (Sarah, 8 Oct 2026): unobserved antigens count as unsequenced,
and antigens the scheme cannot paint are reported by one warning per clade."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from af.chart.model import Antigen, Chart, Titres
from af.clades.colours import ColourEntry
from af.clades.colours import ColourScheme as CladeColourScheme
from af.geo.colours import STATE_NO_SCHEME_ROW, STATE_NO_SUPPORTED, STATE_UNOBSERVED, DotStyle
from af.map.build import unpainted_clades
from af.map.style import PointIn
from tests.map.test_colouring import caller_setup


def test_an_unobserved_antigen_counts_as_unsequenced(tmp_path: Path) -> None:
    colours, chart, clade_subtype = caller_setup(tmp_path)
    key = next(iter(colours._sequences))
    colours._sequences[key] = dataclasses.replace(
        colours._sequences[key], clade_evidence="unobserved"
    )
    own = CladeColourScheme(
        clade_subtype, "caller", (ColourEntry(1, "P", "Clade P", "#aa0000", False),)
    )
    got = colours.for_chart(chart, own)
    assert got.sequenced[0] is False  # its clade is not known: not a scheme gap
    assert got.styles[0].state == STATE_UNOBSERVED
    assert got.provenance["clade_evidence"] == {"unobserved": 1}
    assert got.provenance["uncoloured"][STATE_UNOBSERVED] == 1


def test_unpainted_antigens_are_reported_once_per_clade_over_drawn_points() -> None:
    names = ["/".join(("A", "PLACE", str(n), "2021")) for n in range(6)]
    chart = Chart({"V": "A(H1N1)"}, [Antigen(n) for n in names], [], Titres([[]] * 6))
    gap = DotStyle("", None, state=STATE_NO_SCHEME_ROW, clade="A", tree_clade="C.1.9")
    none = DotStyle("", None, state=STATE_NO_SUPPORTED, tree_clade="C.1.9")
    painted = DotStyle("Clade P", "#aa0000", "clade")
    styles = [gap, gap, gap, none, painted, gap]
    points = [
        PointIn("ag0", names[0], "antigen", (0.0, 0.0), reference=True),
        PointIn("ag1", names[1], "antigen", (1.0, 0.0)),
        PointIn("ag2", names[2], "antigen", None),  # not drawn: no coordinates
        PointIn("ag3", names[3], "antigen", (2.0, 0.0)),
        PointIn("ag4", names[4], "antigen", (3.0, 0.0)),
        PointIn("ag5", names[5], "antigen", (4.0, 0.0), hide="H-1"),  # hidden: not drawn
    ]
    got = unpainted_clades(chart, points, styles, "clades")
    assert [(r["clade"], r["points"], r["reference_antigens"]) for r in got] == [
        ("A", 2, 1),
        (None, 1, 0),
    ]
    assert got[0]["warning"] == (
        "WARNING: 2 antigens, 1 of them reference antigens, are not painted: their sequences "
        "support clade A (tree labels: C.1.9 2; own loci contradicted), and colour scheme "
        "clades has no row for A."
    )
    assert got[1]["warning"].startswith("WARNING: 1 antigen, 0 of them reference antigens, is ")
    assert unpainted_clades(chart, points, [painted] * 6, "clades") == []  # nothing to report
