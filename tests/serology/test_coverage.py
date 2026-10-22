"""The serum-coverage addenda's shown sera and their Q228 declarations (af.serology.coverage),
on an invented chart and stand-in serum strains: no store."""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

import numpy as np

from af.chart.model import Antigen, Chart, Serum, Titres
from af.chart.titre import Titre
from af.map.colouring import SerumStrain
from af.report.howmade import load_declarations
from af.serology.coverage import MapInput, shown_sera, write_fragment
from af.store import StoreRef

NAME = "A(H3N2)/EXAMPLETOWN/1/2025"
OTHER = "A(H3N2)/EXAMPLEVILLE/2/2025"
REF = StoreRef(
    "chains", "lab-a/h3-hi-lab-a/merged", "0123456789abcdef", "0123456789abcdef" + "0" * 48
)


class Strains:
    """Stands in for StoreColours.for_sera: every serum's strain is clade X.1."""

    def for_sera(self, chart: Chart, residues: object) -> list[SerumStrain]:
        return [
            SerumStrain(i, f"EPI_{i}", f"A{i}", "MDCK1", "X.1", ("X", "X.1"), {}, "test")
            for i in range(chart.n_sera)
        ]


def map_input() -> MapInput:
    """Serum 0 has two homologous antigens: the best match (exact passage, 320) and a smaller
    radius (SIAT1, 640), so its two rules draw different maps. Serum 1 has one: they agree."""
    antigens = [
        Antigen(NAME, "MDCK1"),
        Antigen(NAME, "SIAT1"),
        *(Antigen(OTHER, f"MDCK{i}") for i in (1, 2, 3)),
    ]
    sera = [Serum(NAME, "F1/25", "MDCK1"), Serum(OTHER, "F2/25", "MDCK1")]
    columns = [["320", "640", "80", "40", "20"], ["40", "40", "640", "320", "40"]]
    t = [[Titre.parse(columns[s][a]) for s in range(2)] for a in range(5)]
    chart = Chart({"D": "20260110"}, antigens, sera, Titres(t))
    layout = np.array([[1.5, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 0], [2, 0.5]])
    return MapInput("h3-hi-lab-a", "lab-a", REF, chart, layout, "none")


SECTIONS = {
    "addendum": 2,
    "sections": [
        {"label": "vaccine", "strains": ["EXAMPLETOWN/1/2025"], "reason": "the vaccine strain"},
        {"label": "X.1", "clade": "X.1", "reason": "the clade"},
    ],
}


def test_shown_sera_carry_both_rules_and_the_disagreements_are_declared(tmp_path: Path) -> None:
    shown = shown_sera(Strains(), [map_input()], SECTIONS, dt.date(2025, 9, 1))  # type: ignore[arg-type]
    assert [(s.section, s.record.serum_id, s.record.disagreement()) for s in shown] == [
        ("vaccine", "F1/25", "visible"),
        ("X.1", "F2/25", None),
    ]
    also = [{"ruling": "hand", "dataset": REF.dataset, "version": REF.version,
             "text": "a hand declaration", "file": "declarations/x.md"}]  # fmt: skip
    toml, md = tmp_path / "frag.toml", tmp_path / "frag.md"
    counts = write_fragment(shown, also, toml, md, "declarations/x.md", {"sections": "test"})
    assert counts == {"shown": 2, "disagree": 1, "by_kind": {"visible": 1},
                      "by_addendum": {"addendum 2": 1}, "hand": 1}  # fmt: skip
    entries = tomllib.loads(toml.read_text())["declaration"]
    assert [e["ruling"] for e in entries] == ["Q228", "hand"]
    assert entries[0]["dataset"] == REF.dataset and entries[0]["version"] == REF.version
    assert entries[0]["text"].startswith(
        f"serum coverage addendum 2, vaccine: {NAME} F1/25 (visible)"
    )
    assert "titre 320, radius 2.50" in entries[0]["text"]  # drawn: the best match
    assert "titre 640, radius 2.00" in entries[0]["text"]  # ae's circle rule
    assert len(load_declarations(toml)) == 2  # the round's loader takes the fragment
    page = md.read_text()
    assert "1 of 2 shown sera disagree: 1 visible, 0 titre-only, 0 no-homologous, 0 neither" in page
    assert "## Hand declarations" in page
