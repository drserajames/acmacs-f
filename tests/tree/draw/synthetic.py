"""Small invented trees for the draw tests (public repo: no real names, sequences or clades).

A tree is written as nested lists; a leaf is a dict. :func:`build` turns it into a
:class:`DrawTree` in pre-order, which is the order the figure expects.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")

from af.tree.draw.model import DrawTree  # noqa: E402

# Invented nomenclature: ROOTCL > X > X.1 > X.1.1, and X > X.2.
PARENTS: dict[str, str | None] = {
    "ROOTCL": None,
    "X": "ROOTCL",
    "X.1": "X",
    "X.1.1": "X.1",
    "X.2": "X",
}
BASE = "MKTIIALSYILCLVFA"  # invented 16-residue sequence


def mutate(seq: str, *subs: str) -> str:
    """Apply substitutions like 'K2N' (1-based) to ``seq``, checking the from-residue."""
    s = list(seq)
    for sub in subs:
        before, pos, after = sub[0], int(sub[1:-1]), sub[-1]
        assert s[pos - 1] == before, (sub, seq)
        s[pos - 1] = after
    return "".join(s)


def leaf(n: int, clade: str, date: str = "2025-03-15", precision: str = "day", **kw: Any) -> dict:
    return {
        "id": f"EPI_ISL_{n:07d}",
        "name": f"leaf-{n:03d}",
        "clade": clade,
        "date": date,
        "precision": precision,
        "continent": kw.pop("continent", "EUROPE"),
        "edge": 0.001,
        "aa": kw.pop("aa", BASE),
        **kw,
    }


def inner(
    *children: Any, edge: float = 0.001, subs: tuple[str, ...] = (), aa: str | None = None
) -> dict:
    return {"children": list(children), "edge": edge, "subs": list(subs), "aa": aa}


def build(root: dict, subtype: str = "TEST") -> DrawTree:
    cols: dict[str, list[Any]] = {
        k: []
        for k in (
            "parent",
            "edge",
            "leaf_id",
            "name",
            "date",
            "date_precision",
            "continent",
            "clade",
            "aa",
            "aa_subs",
        )
    }

    def visit(node: dict, parent: int) -> None:
        cols["parent"].append(parent)
        cols["edge"].append(node.get("edge", 0.001))
        me = len(cols["parent"]) - 1
        is_leaf = "children" not in node
        cols["leaf_id"].append(node["id"] if is_leaf else None)
        cols["name"].append(node["name"] if is_leaf else None)
        cols["date"].append(node["date"] if is_leaf else None)
        cols["date_precision"].append(node["precision"] if is_leaf else None)
        cols["continent"].append(node["continent"] if is_leaf else None)
        cols["clade"].append(node["clade"] if is_leaf else None)
        cols["aa"].append(node.get("aa"))
        cols["aa_subs"].append(list(node.get("subs", [])))
        for child in node.get("children", []):
            visit(child, me)

    visit(root, -1)
    return DrawTree(**cols, subtype=subtype)


def clade_block(first_n: int, count: int, clade: str, **kw: Any) -> list[dict]:
    return [leaf(first_n + k, clade, **kw) for k in range(count)]


def standard_tree() -> DrawTree:
    """Three sub-clades under X, with ASR substitutions on their stems.

    Rows: 0-19 X.1 (of which 10-19 X.1.1), 20-39 X.2, 40-44 X only (residual), 45-49 ROOTCL.
    Stem changes: X.1 K2N, X.1.1 L7F (in the window), X.2 A6S (old, outside the window).
    """
    x1_seq = mutate(BASE, "K2N")
    x11_seq = mutate(x1_seq, "L7F")
    x2_seq = mutate(BASE, "A6S")
    x11 = inner(*clade_block(10, 10, "X.1.1", aa=x11_seq, date="2026-02-10"), subs=("L7F",))
    x1 = inner(inner(*clade_block(0, 10, "X.1", aa=x1_seq)), x11, subs=("K2N",))
    x2 = inner(*clade_block(20, 20, "X.2", aa=x2_seq, date="2023-06-01"), subs=("A6S",))
    x = inner(x1, x2, inner(*clade_block(40, 5, "X")))
    return build(inner(x, inner(*clade_block(45, 5, "ROOTCL"))))
