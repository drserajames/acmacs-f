"""Synthetic tables for tests: invented strains on a true map, titres from distances.

No real virus names, sera or titres (acmacs-f is public). Each table shares the reference
antigens and some sera with the others, so a chain has something to merge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from af.chart.ace import write_chart
from af.chart.model import Antigen, Chart, Serum, Titres, empty_table
from af.chart.titre import Titre

DILUTIONS = [10 * 2**k for k in range(12)]


def _titre(logged: float, rng: np.random.Generator, threshold: float = 0.0) -> Titre:
    noisy = logged + rng.normal(0, 0.5)
    step = int(np.floor(noisy))
    if step < 0 or noisy < threshold:
        return Titre.parse("<10")
    return Titre.parse(str(DILUTIONS[min(step, len(DILUTIONS) - 1)]))


def make_tables(
    directory: Path, group: str = "h9-hi-test-lab", n_tables: int = 5, seed: int = 1
) -> list[Path]:
    """Write `n_tables` synthetic tables `<group>-<date>.ace`; returns their paths in date order."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_ref = 6
    ref_ag = rng.uniform(-4, 4, size=(n_ref, 2))
    sera_pos = ref_ag + rng.normal(0, 0.3, size=ref_ag.shape)
    potency = rng.uniform(7, 9, size=n_ref)
    paths = []
    for k in range(n_tables):
        n_test = 8
        test = rng.uniform(-5, 5, size=(n_test, 2))
        ag_pos = np.vstack([ref_ag, test])
        antigens = [Antigen(name=f"TEST-{i + 1}", passage="E3") for i in range(n_ref)]
        antigens += [
            Antigen(
                name=f"TEST-{100 * (k + 1) + i}",
                passage="MDCK1",
                date=f"2021-0{1 + k % 9}-15",
            )
            for i in range(n_test)
        ]
        sera = [
            Serum(
                name=f"TEST-{i + 1}",
                serum_id=f"F{i + 1:03d}",
                passage="E3",
                species="FERRET",
            )
            for i in range(n_ref)
        ]
        table = empty_table(len(antigens), n_ref)
        for i, p in enumerate(ag_pos):
            for j in range(n_ref):
                d = np.linalg.norm(p - sera_pos[j])
                table[i][j] = _titre(potency[j] - d, rng)
        date = f"2021{1 + k:02d}15"
        chart = Chart(
            info={"V": "A(H9N2)", "A": "HI", "l": "TEST", "D": date, "r": "turkey"},
            antigens=antigens,
            sera=sera,
            titres=Titres(table),
        )
        path = Path(directory) / f"{group}-{date}.ace"
        write_chart(chart, path)
        paths.append(path)
    return paths
