"""The report tree: the source tree, cut at a node when the round asks for one. No tables.

Sarah, 25 Nov 2026: "A report tree shouldn't drop sequences that weren't titrated - that is biasing
the tree. If the tree needs to be smaller, then it should be cut at a node." And: "the tables shoud
not affect the trees". So there is **no date rule** here and nothing reads a titre table:

* The W8 pipeline this replaced dropped leaves collected before 2021 unless a WHO CC table held
  them (``tree-to-json --remove-leaves-isolated-before 2021 --important <list>``). That rule is
  gone. It made the old part of the tree a sample of what had been assayed rather than of what
  circulated, and it let every tables publish change which leaves a report tree had.
* The only way a report tree is smaller than its source is a **node cut** (``trim``,
  af.tree.trim): every leaf outside the cut goes, the outgroup with them, and the tree is rooted at
  the cut node. A cut removes a whole clade, so it does not bias what remains inside it. Which node
  is cut is config (acmacs-f-data ``trees/trim.tsv``), never code; with no trim the report tree
  holds every leaf of its source.
* The ``titrated`` / ``titrated_by`` leaf fields are left empty, as on a weekly tree, so I6 writes
  them as null ("not evaluated"). Marking leaves by centre is a figure's business, from a figure
  input, not the tree's.

The tree is derived from a *populated* tree, so nothing is rebuilt:

* the pruned tree keeps every surviving node's ancestral state (carried by node, not by id,
  because pruning changes the ids of the ancestors of removed leaves);
* branch lengths and substitutions are recomputed by :func:`af.tree.populate.populate`, so a branch
  that absorbed a spliced-out node carries the *net* change along it, not a concatenated list;
* clade calls are carried over node by node unless an engine is given to re-run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

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


def report_tree(
    populated: PopulatedTree,
    *,
    backend: Backend | None = None,
    assign_clades: CladeAssigner | None = None,
    continent_of: Callable[[LeafRecord], str | None] | None = None,
    trim: TrimMatch | None = None,
) -> PopulatedTree:
    """A new populated tree: every leaf of ``populated``, or with ``trim`` only those under the cut.

    The input is not modified. With ``trim`` the result has no outgroup. No leaf is dropped for
    its date or for having no titres (module docstring).
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
    outgroup = populated.counts.get("outgroup")
    if outgroup is None:
        raise ValueError("the populated tree records no outgroup; pass outgroup= to populate()")
    removed_by_trim = 0
    trimmed_by_year = Counter[str]()
    for key, record in populated.leaves.items():
        if trim is not None and key not in trim.keys:
            if key != outgroup:  # the outgroup goes with any trim; it is recorded, not counted
                removed_by_trim += 1
                date = record.collection_date
                trimmed_by_year["undated" if date is None else str(date.year)] += 1
            continue
        keep.append(key)
    if trim is not None and len(keep) < 2:
        raise ValueError(f"the trim at {trim.node_id} leaves {len(keep)} leaf; not a tree")
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
    if trim is not None:
        result.counts["removed_by_trim"] = removed_by_trim
        result.counts["removed_by_trim_by_year"] = dict(sorted(trimmed_by_year.items()))
        result.counts["trim"] = {**trim.to_json(), "outgroup_removed": outgroup}
    return result


__all__ = ["report_tree"]
