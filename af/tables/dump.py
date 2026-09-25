"""Readable text dump of an af table: for review and for diffing two reads. Output only.

The layout is fixed and line-oriented so that ``diff`` between two dumps shows exactly the
antigens, sera or cells that changed.
"""

from __future__ import annotations

from .model import Table


def dump(table: Table) -> str:
    out = [
        f"table    {table.table_id}",
        f"hash     {table.content_hash()}",
        f"source   {table.source_key}  {table.meta}",
        f"what     {table.lab} {table.subtype} {table.lineage or '-'} {table.assay}"
        f" rbc:{table.rbc or '-'} date:{table.date}",
        f"size     {len(table.antigens)} antigens x {len(table.sera)} sera",
    ]
    if table.dropped:
        out.append("dropped  " + ", ".join(f"{k}: {v}" for k, v in table.dropped.items()))
    out.extend(f"warning  {w}" for w in table.warnings)
    out.append("")
    for no, sr in enumerate(table.sera, start=1):
        extra = " ".join(filter(None, [sr.species, *sr.annotations]))
        date = f" ({sr.passage_date})" if sr.passage_date else ""
        out.append(f"SR {no:3d}  {sr.name} | {sr.serum_id} | {sr.passage}{date} {extra}".rstrip())
    out.append("")
    for no, ag in enumerate(table.antigens, start=1):
        date = f" ({ag.passage_date})" if ag.passage_date else ""
        ref = " REF" if ag.reference else ""
        out.append(
            f"AG {no:3d}  {ag.name} | {ag.passage}{date} | {' '.join(ag.lab_ids)}"
            f" | {ag.date or '-'}{ref}"
        )
    out.append("")
    width = max([len(_cell(c)) for row in table.titres for c in row] + [5])
    out.append("         " + " ".join(f"{n:>{width}d}" for n in range(1, len(table.sera) + 1)))
    for no, row in enumerate(table.titres, start=1):
        out.append(f"AG {no:3d}  " + " ".join(f"{_cell(c):>{width}s}" for c in row))
    return "\n".join(out) + "\n"


def _cell(readings: list[str]) -> str:
    return ",".join(readings) if readings else "*"
