"""Interface I6: what one tree-store version holds, and how to read and write it.

A version of ``trees/<subtype>/<purpose>`` (I9) is a directory of four files::

    tree.nwk            rooted, collapsed, ladderized; leaves named by leaf key (EPI_ISL|accession),
                        internal nodes by their stable 16-hex id; lengths in the configured scale
    nodes.parquet       one row per node in drawing (pre-)order; the columns are in NODE_COLUMNS
    ancestral.parquet   node_id, nucleotides for internal nodes (leaves are in the sequence store)
    tree.json           what the rows mean: format, subtype, scale, alignment length, ASR backend,
                        clade-set version, and every step's counts

Nothing time-stamped is written, so an identical rebuild is byte-identical and the store keeps the
same version id (I9). Provenance lives beside, in the store's PROVENANCE.json.

``nodes.parquet`` is a superset of the columns workstream 9's ``DrawTree`` reads, with the same
names (``parent``, ``edge``, ``leaf_id``, ``name``, ``date``, ``date_precision``, ``continent``,
``clade``, ``aa``, ``aa_subs``), so its adapter is a column selection. The full spec is
``notes/trees/I6-DRAFT.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from af.store import Provenance, Store, StoreRef
from af.tree.io import newick
from af.tree.populate import PopulatedTree

FORMAT = "af-tree-i6/1"
TREE_FILE = "tree.nwk"
NODES_FILE = "nodes.parquet"
ANCESTRAL_FILE = "ancestral.parquet"
META_FILE = "tree.json"

_list_of_str = pa.list_(pa.string())

NODE_SCHEMA = pa.schema(
    [
        ("node_id", pa.string()),  # stable 16-hex id (XOR of leaf ids); the key to use
        ("parent", pa.int32()),  # row index of the parent, -1 for the root; this version only
        ("parent_id", pa.string()),
        ("is_leaf", pa.bool_()),
        ("n_leaves", pa.int32()),
        ("edge", pa.float64()),  # branch length into the node, in tree.json's branch_scale
        ("edge_ml", pa.float64()),  # CMAPLE's length, whatever the scale
        ("nuc_changes", pa.int32()),  # definite-base changes on the branch (null without states)
        ("leaf_id", pa.string()),  # leaf key EPI_ISL|accession; null on internal nodes
        ("name", pa.string()),
        ("epi_isl", pa.string()),
        ("accession", pa.string()),
        ("date", pa.string()),  # ISO collection date as recorded; read date_precision with it
        ("date_precision", pa.string()),  # day | month | year
        ("collection_date_first", pa.date32()),
        ("collection_date_last", pa.date32()),
        ("country", pa.string()),
        ("region", pa.string()),
        ("continent", pa.string()),
        ("clade", pa.string()),  # every node, from workstream 4's engine
        ("clade_support", pa.int32()),
        ("clade_unobservable", pa.int32()),
        ("clade_inherited", pa.bool_()),
        ("aa", pa.string()),  # observed (leaf) or reconstructed (internal) protein
        ("aa_subs", _list_of_str),  # on the branch into the node, e.g. "N145S"
        ("nuc_subs", _list_of_str),
        ("flags", _list_of_str),  # reported, not excluded (DECISIONS 25 Sep): e.g. clock_outlier
        ("titrated", pa.bool_()),  # leaf has titres (null: not evaluated)
        ("titrated_by", _list_of_str),  # centres (Table.lab) whose tables hold the antigen
    ]
)
NODE_COLUMNS = tuple(NODE_SCHEMA.names)

ANCESTRAL_SCHEMA = pa.schema([("node_id", pa.string()), ("nucleotides", pa.string())])


class I6Error(ValueError):
    """A tree-store version is incomplete or inconsistent."""


def node_table(populated: PopulatedTree) -> pa.Table:
    """The nodes.parquet table: one row per node, pre-order."""
    tree = populated.tree
    rows: dict[str, list[Any]] = {name: [] for name in NODE_COLUMNS}
    index_of: dict[int, int] = {}
    sizes: dict[int, int] = {}
    for node in tree.postorder():
        sizes[id(node)] = 1 if node.is_leaf else sum(sizes[id(c)] for c in node.children)
    for row, node in enumerate(tree.preorder()):
        index_of[id(node)] = row
        key = node.name if node.is_leaf else None
        record = populated.leaves.get(key) if key else None
        call = populated.clades.get(node.id_hex)
        rows["node_id"].append(node.id_hex)
        rows["parent"].append(-1 if node.parent is None else index_of[id(node.parent)])
        rows["parent_id"].append(None if node.parent is None else node.parent.id_hex)
        rows["is_leaf"].append(node.is_leaf)
        rows["n_leaves"].append(sizes[id(node)])
        rows["edge"].append(node.branch_length)
        rows["edge_ml"].append(populated.ml_lengths.get(node.node_id, node.branch_length))
        rows["nuc_changes"].append(populated.nuc_changes.get(node.node_id))
        rows["leaf_id"].append(key)
        rows["name"].append(record.name if record else None)
        rows["epi_isl"].append(record.epi_isl if record else None)
        rows["accession"].append(record.accession if record else None)
        date = record.collection_date if record else None
        rows["date"].append(date.isoformat() if date else None)
        rows["date_precision"].append(record.date_precision if record else None)
        rows["collection_date_first"].append(record.collection_date_first if record else None)
        rows["collection_date_last"].append(record.collection_date_last if record else None)
        rows["country"].append(record.country if record else None)
        rows["region"].append(record.region if record else None)
        rows["continent"].append(populated.continents.get(key) if key else None)
        rows["clade"].append(call.clade if call else None)
        rows["clade_support"].append(call.support if call else None)
        rows["clade_unobservable"].append(call.unobservable if call else None)
        rows["clade_inherited"].append(call.inherited if call else None)
        rows["aa"].append(populated.aa.get(node.node_id))
        rows["aa_subs"].append(populated.aa_subs.get(node.node_id, []))
        rows["nuc_subs"].append(populated.nuc_subs.get(node.node_id, []))
        rows["flags"].append(populated.flags.get(node.node_id, []))
        rows["titrated"].append(populated.titrated.get(key) if key else None)
        rows["titrated_by"].append(populated.titrated_by.get(key, []) if key else [])
    return pa.table(rows, schema=NODE_SCHEMA)


def metadata(populated: PopulatedTree, purpose: str) -> dict[str, Any]:
    leaves, internal = populated.tree.count()
    states = populated.states
    return {
        "format": FORMAT,
        "subtype": populated.subtype,
        "purpose": purpose,
        "leaf_key": "EPI_ISL|accession",
        "leaves": leaves,
        "internal_nodes": internal,
        "alignment_length": populated.alignment_length,
        "numbering": "mature HA",
        "branch_scale": populated.branch_scale,
        "outgroup": populated.counts.get("outgroup"),
        "asr": None
        if states is None
        else {
            "backend": states.backend,
            "version": states.backend_version,
            "parameters": dict(states.parameters),
            "reconstructs_gaps": populated.gaps_reconstructed,
        },
        "clade_set_version": populated.clade_set_version,
        "clade_parents": dict(sorted(populated.clade_parents.items())),
        "counts": dict(sorted(populated.counts.items())),
    }


def write(populated: PopulatedTree, directory: Path, purpose: str) -> list[Path]:
    """Write one version's files into ``directory`` (which must exist). Returns the paths."""
    directory = Path(directory)
    tree_path = directory / TREE_FILE
    newick.dump(populated.tree, tree_path, with_internal_labels=True, precision=12)
    nodes_path = directory / NODES_FILE
    pq.write_table(node_table(populated), nodes_path, compression="zstd")
    written = [tree_path, nodes_path]
    if populated.states is not None:
        internal = [node for node in populated.tree.preorder() if not node.is_leaf]
        ancestral = pa.table(
            {
                "node_id": [node.id_hex for node in internal],
                "nucleotides": [populated.states.nucleotides[node.node_id] for node in internal],
            },
            schema=ANCESTRAL_SCHEMA,
        )
        ancestral_path = directory / ANCESTRAL_FILE
        pq.write_table(ancestral, ancestral_path, compression="zstd")
        written.append(ancestral_path)
    meta_path = directory / META_FILE
    meta_path.write_text(json.dumps(metadata(populated, purpose), indent=1, default=str) + "\n")
    written.append(meta_path)
    return written


def read_metadata(directory: Path) -> dict[str, Any]:
    path = Path(directory) / META_FILE
    if not path.exists():
        raise I6Error(f"{directory}: no {META_FILE}; not a tree-store version")
    meta: dict[str, Any] = json.loads(path.read_text())
    if meta.get("format") != FORMAT:
        raise I6Error(f"{directory}: format {meta.get('format')!r}, this reader expects {FORMAT}")
    return meta


def read_nodes(directory: Path, columns: list[str] | None = None) -> pa.Table:
    """nodes.parquet, checked against the tree it belongs to."""
    directory = Path(directory)
    meta = read_metadata(directory)
    table = pq.read_table(directory / NODES_FILE, columns=columns)
    expected = meta["leaves"] + meta["internal_nodes"]
    if table.num_rows != expected:
        raise I6Error(f"{directory}: {table.num_rows} node rows, tree.json says {expected}")
    return table


def read_ancestral(directory: Path) -> dict[str, str]:
    table = pq.read_table(Path(directory) / ANCESTRAL_FILE)
    return dict(zip(table["node_id"].to_pylist(), table["nucleotides"].to_pylist(), strict=True))


def draw_columns(directory: Path) -> Mapping[str, list[Any]]:
    """The columns workstream 9's ``DrawTree`` takes, as Python lists, plus ``subtype``."""
    names = ["parent", "edge", "leaf_id", "name", "date", "date_precision"]
    names += ["continent", "clade", "aa", "aa_subs"]
    table = read_nodes(directory, columns=names)
    out: dict[str, Any] = {name: table[name].to_pylist() for name in names}
    out["subtype"] = read_metadata(directory)["subtype"]
    return out


def publish(
    store: Store, populated: PopulatedTree, purpose: str, provenance: Provenance
) -> StoreRef:
    """Write the version into ``trees/<subtype>/<purpose>`` and make it CURRENT (I9)."""
    dataset = f"{populated.subtype}/{purpose}"
    with store.build("trees", dataset) as builder:
        write(populated, builder.path, purpose)
        leaves, internal = populated.tree.count()
        summary = {
            "leaves": leaves,
            "internal_nodes": internal,
            "branch_scale": populated.branch_scale,
        }
        return builder.publish(provenance, summary=summary)
