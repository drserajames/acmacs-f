"""af tables as ae's per-table titre cells, for comparing with ae's committed charts.

ae merged a lab's repeat readings of one antigen/serum pair into one titre (whocc-cdc-tsv-ace,
lispmds rules, a lone '>' kept). The same merge is applied here to af's kept readings, and
cells are keyed by what identifies them in both systems, so a comparison can be done cell by
cell: ``<lab id>|<passage>|<harvest date>|<n>|<serum id>|<BOOSTED>``, where ``<n>`` numbers
antigens sharing the first three fields in table order (a workbook can list one preparation
twice, and both ae and af keep both rows).
"""

from __future__ import annotations

from collections.abc import Iterable

from af.chart.titre import MoreThanOnly, Titre, merge_titres

from .model import Table


def merged_cells(tables: Iterable[Table]) -> dict[str, str]:
    """The cells of one group+date (all its tables), merged the way ae merged repeats."""
    readings: dict[str, list[str]] = {}
    for table in tables:
        seen: dict[str, int] = {}
        for i, antigen in enumerate(table.antigens):
            lab_id = antigen.lab_ids[0] if antigen.lab_ids else ""
            base = f"{lab_id}|{antigen.passage}|{antigen.passage_date or ''}"
            key_ag = f"{base}|{seen.get(base, 0)}"
            seen[base] = seen.get(base, 0) + 1
            for j, serum in enumerate(table.sera):
                key = f"{key_ag}|{serum.serum_id}|{'BOOSTED' in serum.annotations}"
                readings.setdefault(key, []).extend(table.titres[i][j])
    out = {}
    for key, values in readings.items():
        if values:
            merged, _ = merge_titres([Titre.parse(v) for v in values], MoreThanOnly.ADJUST_TO_NEXT)
            out[key] = str(merged)
    return out


def differences(ae: dict[str, str], af: dict[str, str]) -> dict[str, dict[str, str]]:
    """Cells whose titre differs ('*' where a side has none)."""
    return {
        key: {"ae": ae.get(key, "*"), "af": af.get(key, "*")}
        for key in sorted(ae.keys() | af.keys())
        if ae.get(key, "*") != af.get(key, "*")
    }
