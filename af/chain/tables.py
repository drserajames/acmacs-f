"""Chain input from the tables store: af tables (interface I2) turned into one-layer charts.

A chain step's input is a chart file named by the table's **map hash**, written once per
map hash. The pipeline hashes that file, so a table whose map-relevant content changes
(titres, antigen or serum identity) restarts the chain at its step, while a table whose
lab metadata changes (CDC adds EPI_ISL or a sequenced passage long after a test) does not.

The chart copies ae's representation of a table (whocc-cdc-tsv-ace), so cross-table
identity works as in today's chains:
* a passage carries its harvest date: "MDCK2/SIAT1 (2016-05-12)" (`ae_passage`);
* repeated readings of one cell in one test are merged with the lispmds rules, which is
  how ae's CDC reader collapsed them (eu-ae: whocc-cdc-tsv-ace -> titer_merge).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from af.chain.config import ChainConfigError, TableRef
from af.chart.ace import write_chart
from af.chart.model import Antigen, Chart, Serum, Titres, empty_table
from af.chart.titre import Titre, merge_titres
from af.store.ref import StoreRef
from af.store.store import Store
from af.tables.model import Table

KIND = "tables"


def cell_titre(readings: list[str]) -> Titre:
    """One cell: no reading is missing, several readings are merged (lispmds)."""
    titres = [Titre.parse(r) for r in readings]
    if len(titres) <= 1:
        return titres[0] if titres else Titre.parse("*")
    merged, _ = merge_titres(titres)
    return merged


def table_chart(table: Table) -> Chart:
    """An af table as a one-layer chart, in ae's representation (see the module docstring)."""
    antigens = [
        Antigen(
            name=a.name,
            passage=a.ae_passage(),
            reassortant=a.reassortant,
            annotations=tuple(a.annotations),
            date=a.date or "",
            lab_ids=tuple(a.lab_ids),
            extra={"L": a.lineage[0]} if a.lineage else {},
        )
        for a in table.antigens
    ]
    sera = [
        Serum(
            name=s.name,
            serum_id=s.serum_id,
            passage=s.ae_passage(),
            reassortant=s.reassortant,
            annotations=tuple(s.annotations),
            species=s.species,
            extra={"L": s.lineage[0]} if s.lineage else {},
        )
        for s in table.sera
    ]
    cells = empty_table(len(antigens), len(sera))
    for i, row in enumerate(table.titres):
        for j, readings in enumerate(row):
            cells[i][j] = cell_titre(readings)
    date = table.test_date.strftime("%Y%m%d") + (
        f".{table.date_suffix}" if table.date_suffix > 1 else ""
    )
    info = {"V": table.subtype, "A": table.assay, "l": table.lab, "D": date}
    if table.rbc:
        info["r"] = table.rbc
    return Chart(info=info, antigens=antigens, sera=sera, titres=Titres(cells))


def tables_from_store(
    store_root: Path,
    dataset: str,
    inputs_dir: Path,
    exclude: set[str] = frozenset(),  # type: ignore[assignment]
) -> tuple[StoreRef, list[TableRef]]:
    """The dataset's CURRENT tables as chain inputs, in (date, suffix) order.

    Each table becomes `<inputs_dir>/<table_id>.<map_hash[:16]>.ace`, written only if absent.
    The stored table is checked against its content hash (on read) and its map hash (here).
    Every id in `exclude` must name a table of the dataset (design rule 1).
    """
    store = Store.open(store_root)
    ref = store.current(KIND, dataset)
    version = store.resolve(ref)
    index = json.loads((version / "index.json").read_text())["tables"]
    unmatched = set(exclude) - set(index)
    if unmatched:
        raise ChainConfigError(f"exclusions match no table in {dataset}: {sorted(unmatched)}")
    inputs_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for table_id, entry in index.items():
        if table_id in exclude:
            continue
        path = inputs_dir / f"{table_id}.{entry['map_hash'][:16]}.ace"
        if not path.exists():
            text = (version / "tables" / f"{table_id}.json").read_text(encoding="utf-8")
            table = Table.from_json(json.loads(text))  # verifies the content hash
            if table.map_hash() != entry["map_hash"]:
                raise ChainConfigError(f"{table_id}: map hash differs from the store's index")
            write_chart(table_chart(table), path)
        date = datetime.date.fromisoformat(entry["date"])
        refs.append(TableRef(table_id, path, date, int(entry["date_suffix"])))
    if not refs:
        raise ChainConfigError(f"no tables in {dataset}")
    return ref, sorted(refs, key=lambda t: (t.date, t.suffix))
