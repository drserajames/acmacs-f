"""Match an external tool's nodes back to af's, by clade rather than by name.

Every ASR tool names internal nodes its own way, and several re-root the tree before writing it
out. Matching by name or by position is therefore wrong in a way that does not announce itself: the
states line up with the wrong nodes and the reconstruction looks plausible. af matches by **clade**
— the set of leaves below a node, as the XOR of their stable ids — which is invariant under
renaming, re-ordering and re-rooting anywhere outside the clade.

The same mechanism catches a tool that quietly drops nodes: matching reports how many of af's
internal nodes got no state, and the caller decides whether that is tolerable. (Measured: IQ-TREE
returned 13,476 of 13,477 H3 internal nodes; name matching would have hidden the missing one.)
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from af.tree.io import newick
from af.tree.model import Tree, TreeError


class MatchError(ValueError):
    """The tool's tree does not describe the same leaves as af's."""


def _read_tool_tree(path: Path) -> Tree:
    """Read a tool's output tree, whether it is Newick or NEXUS-wrapped."""
    text = Path(path).read_text()
    if text.lstrip().upper().startswith("#NEXUS"):
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if lowered.startswith("tree ") and "=" in stripped:
                text = stripped.split("=", 1)[1].strip()
                break
        else:
            raise MatchError(f"{path}: NEXUS file with no tree line")
    return newick.loads(text)


def clade_ids(tree: Tree, leaf_ids: Mapping[str, int]) -> dict[int, str]:
    """Map clade id -> the tree's own label, for every internal node that has one."""
    below: dict[int, int] = {}
    out: dict[int, str] = {}
    for node in tree.postorder():
        if node.is_leaf:
            name = node.name or ""
            if name not in leaf_ids:
                raise MatchError(f"the tool's tree has a leaf af does not know: {name!r}")
            below[id(node)] = leaf_ids[name]
            continue
        value = 0
        for child in node.children:
            value ^= below[id(child)]
        below[id(node)] = value
        if node.name:
            out[value] = node.name
    return out


def match_states_by_clade(
    tree: Tree, tool_tree: Path, states_by_name: Mapping[str, str]
) -> dict[int, str]:
    """Return af node id -> sequence, matching the tool's nodes by their clades.

    Matching is done on the *unrooted* clade (a clade and its complement are the same split), so a
    tool that re-roots its output still lines up.
    """
    if not tree.ids_assigned():
        tree.assign_ids()
    leaf_ids = {leaf.name or "": leaf.node_id for leaf in tree.leaves()}
    total = 0
    for value in leaf_ids.values():
        total ^= value

    tool = _read_tool_tree(tool_tree)
    tool_leaves = {leaf.name or "" for leaf in tool.leaves()}
    missing = set(leaf_ids) - tool_leaves
    extra = tool_leaves - set(leaf_ids)
    if missing or extra:
        raise MatchError(
            f"{tool_tree}: leaf sets differ — {len(missing)} missing, {len(extra)} unexpected"
        )

    by_clade = clade_ids(tool, leaf_ids)

    # Match on the clade as written first, and only then on its complement — a tool that re-roots
    # writes the same split the other way round. Each of the tool's nodes is used at most once:
    # without that, a clade and its complement both match the same tool node and one af node
    # silently receives another's sequence.
    out: dict[int, str] = {}
    used: set[int] = set()
    for direct in (True, False):
        for node in tree.internal():
            if node.node_id in out:
                continue
            clade = node.node_id if direct else total ^ node.node_id
            label = by_clade.get(clade)
            if label is None or clade in used:
                continue
            sequence = states_by_name.get(label)
            if sequence is not None:
                out[node.node_id] = sequence
                used.add(clade)
    if not out:
        raise MatchError(
            f"{tool_tree}: no node matched. The tool's tree describes the same leaves but no "
            "internal clade lined up, which usually means its labels are not in the states file."
        )
    return out


def unmatched_ids(tree: Tree, states: Mapping[int, str]) -> list[int]:
    """Which of af's internal nodes got no state.

    Usually empty. The one routine exception is the **root**: raxml-ng and IQ-TREE write unrooted
    trees, and in an unrooted tree the root is not a node — it is a point on the branch between its
    two children — so no state is reconstructed for it. TreeTime keeps the rooting and returns one.

    One further case is expected rather than wrong: if the root has exactly **two** children, the
    split above each child is the *same* edge, so an unrooting backend has one node where af has
    three, and a child of the root goes unmatched too. Real trees are rooted on the outgroup and
    have a multifurcating root, where only the root itself is affected.

    Anything beyond that is a real gap and the caller should treat it as an error.
    """
    if not tree.ids_assigned():
        raise TreeError("ids have not been assigned")
    return [node.node_id for node in tree.internal() if node.node_id not in states]


def unmatched_count(tree: Tree, states: Mapping[int, str]) -> int:
    """How many of af's internal nodes got no state. Report it, never ignore it."""
    return len(unmatched_ids(tree, states))
