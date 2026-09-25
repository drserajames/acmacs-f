"""Read a tree-store version (I6) into what the figure needs.

The figure takes a :class:`DrawTree`, the clade hierarchy, per-centre leaf sets for the
centre-marked variant, and the input hashes for provenance. All of it comes from one version
directory, so the hierarchy always matches the clade set that labelled the nodes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .model import DrawTree


class StoreInputError(ValueError):
    """The version is missing something the figure needs, or a request matches nothing."""


@dataclass
class StoreTree:
    tree: DrawTree
    parents: dict[str, str | None]
    clade_set_version: str
    titrated_by: dict[str, list[str]]  # leaf id -> centre ids
    inputs: dict[str, str]  # for the I7 provenance: item -> content hash

    def centre_leaves(self, centre: str) -> frozenset[str]:
        """Leaf ids titrated by ``centre``; a centre with no leaf is an error (design rule 1)."""
        ids = frozenset(leaf for leaf, labs in self.titrated_by.items() if centre in labs)
        if not ids:
            known = sorted({lab for labs in self.titrated_by.values() for lab in labs})
            raise StoreInputError(f"no leaf titrated by {centre!r}; centres present: {known}")
        return ids


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_version(directory: Path) -> StoreTree:
    from af.tree.io import i6  # the tree store's own reader (workstream 5)

    meta = i6.read_metadata(directory)
    for key in ("clade_parents", "clade_set_version"):
        if key not in meta:
            raise StoreInputError(f"{directory}: tree.json has no {key!r}")
    tree = DrawTree(**i6.draw_columns(directory))
    parents = {str(k): (str(v) if v else None) for k, v in meta["clade_parents"].items()}
    table = i6.read_nodes(directory, columns=["leaf_id", "titrated_by"])
    titrated = {
        leaf: list(labs or [])
        for leaf, labs in zip(
            table["leaf_id"].to_pylist(), table["titrated_by"].to_pylist(), strict=True
        )
        if leaf is not None
    }
    inputs = {
        "tree_nodes": _sha256(directory / "nodes.parquet"),
        "tree_metadata": _sha256(directory / "tree.json"),
    }
    return StoreTree(tree, parents, str(meta["clade_set_version"]), titrated, inputs)
