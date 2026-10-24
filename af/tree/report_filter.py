"""The report tree: older leaves kept only if they were titrated (task 5.5).

Today's pipeline (W8 ``pdf`` step (c), INVENTORY C §2A) lists every strain in any WHO CC table
(``hidb5-find --list-names``) and runs ``tree-to-json --remove-leaves-isolated-before 2021
--important <list>``: leaves collected before 2021 are dropped unless they have titres, keeping the
topology. af does the same from a *populated* tree, so nothing is rebuilt:

* the pruned tree keeps every surviving node's ancestral state (carried by node, not by id,
  because pruning changes the ids of the ancestors of removed leaves);
* branch lengths and substitutions are recomputed by :func:`af.tree.populate.populate`, so a branch
  that absorbed a spliced-out node carries the *net* change along it, not a concatenated list;
* clade calls are carried over node by node unless an engine is given to re-run.

Titrated means matched to an antigen in the tables store: by EPI_ISL where the lab supplies it
(field names come from config, since only some labs have them), else by normalised strain name.
The name normaliser is passed in, one editable copy (design rule 6), never a local re-spelling.
Until workstream 10's serology store exists this reads the tables store (prompt 05, task 5).

With a trim (af.tree.trim, found on the source tree as tree_trim.R found it), every leaf outside
the cut goes first, the outgroup with them, and the date rule then runs inside the cut. The tree is
rooted at the cut node. What the trim removed is counted apart from what the date rule removed, and
``removed_by_trim_otherwise_kept`` is what the trim alone cost: the leaves the date rule would
have kept.
"""

from __future__ import annotations

import datetime
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from af.tables.model import Table
from af.tree.asr.base import AncestralStates, Backend
from af.tree.populate import (
    CladeAssigner,
    CladeCall,
    CladeResult,
    LeafRecord,
    PopulatedTree,
    populate,
)
from af.tree.trim import TrimMatch


@dataclass(frozen=True)
class TitratedIndex:
    """Which sequences have titres, and from which centres (``Table.lab``, never a fixed list).

    Keyed by EPI_ISL and by normalised name; each value is the set of labs whose tables hold
    that antigen. The per-centre sets are what the centre-marked report figure needs.
    """

    epi_isl: Mapping[str, frozenset[str]]
    names: Mapping[str, frozenset[str]]
    name_key: Callable[[str], str]
    tables: int = 0
    antigens: int = 0

    @classmethod
    def from_tables(
        cls,
        tables: Iterable[Table],
        name_key: Callable[[str], str],
        epi_fields: Sequence[str] = (),
    ) -> TitratedIndex:
        """Index every antigen of every table. ``epi_fields`` name ``Antigen.source`` keys."""
        epi: dict[str, set[str]] = {}
        names: dict[str, set[str]] = {}
        n_tables = n_antigens = 0
        for table in tables:
            n_tables += 1
            for antigen in table.antigens:
                n_antigens += 1
                names.setdefault(name_key(antigen.name), set()).add(table.lab)
                if antigen.epi_isl:  # the lab's own pairing, where the table format carries one
                    epi.setdefault(antigen.epi_isl, set()).add(table.lab)
                for field_name in epi_fields:
                    value = str(antigen.source.get(field_name) or "").strip()
                    if value:
                        epi.setdefault(value, set()).add(table.lab)
        return cls(
            {key: frozenset(labs) for key, labs in epi.items()},
            {key: frozenset(labs) for key, labs in names.items()},
            name_key,
            n_tables,
            n_antigens,
        )

    def match(self, record: LeafRecord) -> str | None:
        """How the leaf matched ("epi_isl" or "name"), or None if it has no titres."""
        if record.epi_isl in self.epi_isl:
            return "epi_isl"
        if self.name_key(record.name) in self.names:
            return "name"
        return None

    def labs(self, record: LeafRecord) -> list[str]:
        """Every centre whose tables hold this sequence's antigen, by either key, sorted."""
        found = self.epi_isl.get(record.epi_isl, frozenset()) | self.names.get(
            self.name_key(record.name), frozenset()
        )
        return sorted(found)


def collected_before(record: LeafRecord, cutoff: datetime.date) -> bool | None:
    """True if the whole collection interval is before ``cutoff``; None if there is no date.

    The interval, not the point date: a year-only 2020 record lies wholly before 2021-01-01, and a
    year-only 2021 record does not, whatever day the source padded it to.
    """
    last = record.collection_date_last or record.collection_date
    if last is None:
        return None
    return last < cutoff


def report_tree(
    populated: PopulatedTree,
    cutoff: datetime.date,
    titrated: TitratedIndex,
    *,
    backend: Backend | None = None,
    assign_clades: CladeAssigner | None = None,
    continent_of: Callable[[LeafRecord], str | None] | None = None,
    trim: TrimMatch | None = None,
) -> PopulatedTree:
    """A new populated tree without the untitrated leaves collected wholly before ``cutoff``.

    The input is not modified. Undated leaves are kept and counted: dropping them would be a
    silent rule. ``titrated`` is recorded on every leaf of the result. With ``trim``, only the
    leaves under the cut are considered, and the result has no outgroup.
    """
    if populated.states is None:
        raise ValueError("the report tree is derived from a tree with ancestral states")
    tree, twins = populated.tree.copy()
    old_state = {
        id(twins[id(node)]): populated.states.nucleotides[node.node_id]
        for node in populated.tree.internal()
    }
    old_call: dict[int, CladeCall] = {
        id(twins[id(node)]): populated.clades[node.id_hex]
        for node in populated.tree.preorder()
        if node.id_hex in populated.clades
    }
    for node in populated.tree.preorder():  # prune on CMAPLE lengths, rescale afterwards
        twins[id(node)].branch_length = populated.ml_lengths[node.node_id]

    keep: list[str] = []
    how: dict[str, str | None] = {}
    counts = {"removed_before_cutoff": 0, "kept_titrated_before_cutoff": 0, "kept_undated": 0}
    outgroup = populated.counts.get("outgroup")
    if outgroup is None:
        raise ValueError("the populated tree records no outgroup; pass outgroup= to populate()")
    trimmed = dict.fromkeys(
        ("removed_by_trim", "removed_by_trim_titrated", "removed_by_trim_otherwise_kept"), 0
    )
    trimmed_by_year = Counter[str]()
    for key, record in populated.leaves.items():
        how[key] = titrated.match(record)
        before = collected_before(record, cutoff)
        if trim is not None and key not in trim.keys:
            if key != outgroup:  # the outgroup goes with any trim; it is recorded, not counted
                trimmed["removed_by_trim"] += 1
                trimmed["removed_by_trim_titrated"] += how[key] is not None
                trimmed["removed_by_trim_otherwise_kept"] += not before or how[key] is not None
                date = record.collection_date
                trimmed_by_year["undated" if date is None else str(date.year)] += 1
            continue
        if key == outgroup:  # the root is defined by it; old and untitrated is expected
            pass
        elif before is None:
            counts["kept_undated"] += 1
        elif before and how[key] is None:
            counts["removed_before_cutoff"] += 1
            continue
        elif before:
            counts["kept_titrated_before_cutoff"] += 1
        keep.append(key)
    if trim is not None and len(keep) < 2:
        raise ValueError(
            f"the trim at {trim.node_id} and the date rule leave {len(keep)} leaf left; not a tree"
        )
    tree.prune(keep)
    tree.assign_ids()

    states = AncestralStates(
        nucleotides={node.node_id: old_state[id(node)] for node in tree.internal()},
        backend=populated.states.backend,
        backend_version=populated.states.backend_version,
        seconds=populated.states.seconds,
        parameters=populated.states.parameters,
    )

    def carried(_nodes: Sequence[object]) -> CladeResult:
        calls = {
            node.id_hex: old_call[id(node)] for node in tree.preorder() if id(node) in old_call
        }
        return CladeResult(populated.clade_set_version or "", calls, populated.clade_parents)

    engine = assign_clades or (carried if populated.clades else None)
    result = populate(
        tree,
        populated.subtype,
        {key: populated.leaves[key] for key in keep},
        states,
        branch_scale=populated.branch_scale,
        backend=backend,
        assign_clades=engine,
        continent_of=continent_of,
        outgroup=None if trim is not None else outgroup,
    )
    result.build = populated.build  # pruning does not change how the topology was built
    if continent_of is None:
        result.continents = {key: populated.continents.get(key) for key in keep}
    result.titrated = {key: how[key] is not None for key in keep}
    result.titrated_by = {key: titrated.labs(populated.leaves[key]) for key in keep}
    for lab in sorted({lab for labs in result.titrated_by.values() for lab in labs}):
        result.counts[f"titrated_leaves_{lab}"] = sum(lab in v for v in result.titrated_by.values())
    result.counts.update(counts)
    result.counts["report_cutoff"] = cutoff.isoformat()
    result.counts["titrated_leaves"] = sum(result.titrated.values())
    result.counts["titrated_by_epi_isl"] = sum(how[key] == "epi_isl" for key in keep)
    result.counts["titrated_by_name"] = sum(how[key] == "name" for key in keep)
    result.counts["titre_index_tables"] = titrated.tables
    result.counts["titre_index_antigens"] = titrated.antigens
    if trim is not None:
        result.counts.update(trimmed)
        result.counts["removed_by_trim_by_year"] = dict(sorted(trimmed_by_year.items()))
        result.counts["trim"] = {**trim.to_json(), "outgroup_removed": outgroup}
    return result


__all__ = ["TitratedIndex", "collected_before", "report_tree"]
