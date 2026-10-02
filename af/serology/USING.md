# Using the serology join: from an antigen to its sequence and clade

The serology store keeps titres; the join (`af.serology.joins`) links each antigen row to a
sequence in the sequence store and to that sequence's clade. Geo, stat and the antigenic maps
all go through this one path, so a virus has the same sequence, clade and colour everywhere.

## The antigen path

```python
from pathlib import Path

from af.clades.nomenclature import lineage
from af.clades.store import clade_set_for, dataset_for
from af.serology import query
from af.serology.joins import (
    link_from_store,
    preparation_clade,
    preparation_key,
    preparation_sequences,
)
from af.seq.matching_rules import matching_rules
from af.store import Store

store = Store.open(Path("/path/to/store"))
rules = matching_rules(Path("/path/to/acmacs-f-data"))  # every matching table, required
clones = Path("/path/to/influenza-clade-nomenclature")
con = query.connect(store.resolve(store.current("serology", "all")))

links = link_from_store(con, store, rules, with_clades=True)
links.refs  # the sequences and clades versions read: keep them to reproduce this
sequences = preparation_sequences(con, rules.passages)  # PreparationKey -> PreparationSequence

key = ("A(H3N2)", "A(H3N2)/EXAMPLETOWN/1/2021", "", (), "SIAT1 (2021-02-01)")
# ... or from a chart point: key = preparation_key(chart, "antigen", index)
found = sequences.get(key)  # None: no row of this preparation found a usable sequence

clade_set = clade_set_for(store, store.current("clades", dataset_for("A(H3N2)")), clones)
clade = preparation_clade(found)  # None: unknown; "": no clade named; or a name
path = lineage(clade_set, clade)  # root -> leaf
# path is None: unknown (no sequence, no clade row, or candidates that disagree)
# path == (): the nomenclature names no clade here: nothing to know
```

Keep the three outcomes apart: `None` is "unknown", `()` is "known to have no clade". A
consumer that collapses them would read a missing sequence as a virus outside every clade.
`preparation_clade` and `af.clades.nomenclature.lineage` both keep them apart; write neither
by hand.

## What "the sequence of a preparation" means

A preparation is serology's grouping of antigen rows: (table subtype, name, reassortant,
annotations, identity passage), the identity passage being the lab's passage with its harvest
date. **Passage is part of identity**: an egg and a cell preparation of one virus are two
preparations, never merged. `preparation_sequences` decides each preparation's sequence, in
this order:

1. its rows that matched cleanly;
2. else its rows matched with a doubt ae also uses (egg antigen with only a cell sequence,
   reassortant, lab EPI_ISL whose name differs), kept in `doubts`;
3. else its rows that refused a name tie: no sequence, the `tied` candidates, and ae's
   `ranked` pick.

When its rows name two GISAID records of one virus (`conflict`), the record whose passage
matches the preparation's is taken (`resolution == "rows.passage-matched"`); otherwise none is,
and `alternatives` lists them.

A preparation with no sequence (`epi_isl is None`) may still have one clade: when every tied
candidate or alternative carries the same clade, or, for a tie that splits, the clade of ae's
`ranked` pick (`preparation_clade`). `af.geo.colours.dot_styles` applies the same order of rules
for colour, comparing the candidates' colours rather than their clades (two clades can share a
colour row); anything that needs a clade per antigen should call `preparation_clade` rather than
re-derive it.

## Reproducing a result

`link_from_store` reads CURRENT unless told otherwise. To repeat a result after CURRENT moves,
pass the versions it read back in:

```python
from af.store import StoreRef

pins = {d: StoreRef.from_json(r) for d, r in links.refs["sequences"].items()}
clades = [StoreRef.from_json(r) for r in links.refs["clades"]]
again = link_from_store(con, store, rules, with_clades=True, sequences=pins, clades=clades)
```

The serology version itself is pinned by connecting to it (`store.resolve(ref)`), and a
serology version for chosen tables can be built privately with `af.serology.store.build`.
