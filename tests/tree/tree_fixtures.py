"""Shared synthetic tree for the populate / I6 / report-filter tests. Invented names only.

The tree (outgroup ``o``)::

    root ─┬─ o
          └─ x ─┬─ y ─┬─ a
                │     └─ b
                └─ z ─┬─ c
                      └─ d

Internal sequences are chosen so that branch x→z carries no nucleotide change (it must be
collapsed on the mutation scale) and x→y carries one codon change.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator, Mapping

from af.tree.asr.base import AncestralStates
from af.tree.build import finish_tree
from af.tree.io import newick
from af.tree.model import Node, Tree
from af.tree.populate import LeafRecord, leaf_key

KEYS = {name: leaf_key(f"EPI_ISL_90000{i}", f"EPI90000{i}") for i, name in enumerate("oabcd")}
LEAF_SEQ = {
    "o": "ATGAAACCC",
    "a": "ATGGAACCC",  # AAA->GAA on y->a: K2E
    "b": "ATGGAACCN",  # an ambiguity: no change counted, no aa change
    "c": "ATGAAACCG",  # synonymous CCC->CCG on z->c
    "d": "ATGAAA---",  # a deletion: gaps never count as nucleotide changes
}
INTERNAL_SEQ = {"root": "ATGAAACCT", "x": "ATGAAACCC", "y": "ATGGAACCC", "z": "ATGAAACCC"}


def records() -> dict[str, LeafRecord]:
    out = {}
    for index, name in enumerate("oabcd"):
        precision = "year" if name == "b" else "day"
        date = datetime.date(2020 + index, 1, 1 if name == "b" else 15)
        out[KEYS[name]] = LeafRecord(
            epi_isl=f"EPI_ISL_90000{index}",
            accession=f"EPI90000{index}",
            name=f"A/EXAMPLETOWN/{index}/2020",
            nucleotides=LEAF_SEQ[name],
            collection_date=date,
            date_precision=precision,  # type: ignore[arg-type]
            collection_date_first=date if name != "b" else datetime.date(2021, 1, 1),
            collection_date_last=date if name != "b" else datetime.date(2021, 12, 31),
            country="EXAMPLELAND",
            region="Europe",
        )
    return out


def built() -> tuple[Tree, dict[str, int]]:
    """The finished tree, and each internal node's id by its letter."""
    k = KEYS
    text = (
        f"({k['o']}:0.1,(({k['a']}:0.2,{k['b']}:0.3):0.05,({k['c']}:0.1,{k['d']}:0.1):0.01):0.02);"
    )
    tree = finish_tree(newick.loads(text), outgroup=k["o"]).tree
    ids = {}
    for node in tree.internal():
        leaves = {leaf.name for leaf in _leaves_under(node)}
        if node.parent is None:
            ids["root"] = node.node_id
        elif leaves == {k["a"], k["b"]}:
            ids["y"] = node.node_id
        elif leaves == {k["c"], k["d"]}:
            ids["z"] = node.node_id
        else:
            ids["x"] = node.node_id
    return tree, ids


def _leaves_under(node: Node) -> Iterator[Node]:
    stack = [node]
    while stack:
        item = stack.pop()
        if item.is_leaf:
            yield item
        stack.extend(item.children)


def states_for(ids: Mapping[str, int]) -> AncestralStates:
    return AncestralStates(
        nucleotides={ids[name]: INTERNAL_SEQ[name] for name in ids},
        backend="stub",
        backend_version="0",
        seconds=0.0,
    )


class GapBlind:
    name = "gapblind"
    reconstructs_gaps = False
    optimises_branch_lengths = False

    def version(self) -> str:
        return "0"

    def reconstruct(self, tree, alignment, work_dir, threads=1):
        raise NotImplementedError
