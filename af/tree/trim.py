"""The shipped tree trim, kept as a fallback for a tree that already exists (task 5.2).

Today's W5 ``tree_trim.R`` cuts the H3 and B/Vic trees to the subtree under the one branch whose
amino-acid substitutions include 223V (H3) or 127T (B/Vic) with at least 15,000 tips below it; H1
is never cut. Sarah's ruling (task 5.2, COMPARISON.md §2): af's primary mechanism is to restart the
tree without retired clades, with this trim kept as an option; the M2 report trees use it (Q270 a).

The rule is data, one row per subtype in the data repo's ``trees/trim.tsv``, not code. As in
tree_trim.R, the branch is found on the *source* tree (the full tree, before the report's date
rule), any from-state counts (the branch's substitution need only end in 223V), and the tip count
is the node's leaves in that tree. tree_trim.R stops with no context unless exactly one branch
matches; here 0 or 2+ is a :class:`TrimError` naming every branch that carries the substitution,
with its tips, so the reader can see whether the threshold or the tree moved.

The trimmed tree is rooted at the cut node and has no outgroup, as tree_trim.R's had. Keeping the
outgroup would keep the long branch from the old root down to the cut, and that branch is the
width the trim exists to remove.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from af.tree.model import Node
from af.tree.populate import PopulatedTree

COLUMNS = ("subtype", "substitution", "min_tips", "reason")
_SUBSTITUTION = re.compile(r"(?P<position>[1-9][0-9]*)(?P<to>[A-Z*-])")


class TrimError(ValueError):
    """The trim rules are malformed, or the tree has no single branch the rule picks."""


@dataclass(frozen=True)
class TrimRule:
    """Keep the subtree under the one branch that gains ``to`` at ``position``."""

    subtype: str
    position: int
    to: str
    min_tips: int
    reason: str

    @property
    def substitution(self) -> str:
        return f"{self.position}{self.to}"

    def carried_by(self, aa_subs: list[str]) -> bool:
        """True if a branch's substitutions include this one, from any state."""
        return any(sub[1:] == self.substitution for sub in aa_subs)

    def to_json(self) -> dict[str, object]:
        return {
            "subtype": self.subtype,
            "substitution": self.substitution,
            "min_tips": self.min_tips,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TrimMatch:
    """The branch the rule picked on the source tree, and the leaves under it."""

    rule: TrimRule
    node_id: str
    tips: int
    clade: str | None
    keys: frozenset[str]

    def to_json(self) -> dict[str, object]:
        return {
            **self.rule.to_json(),
            "node": self.node_id,
            "tips": self.tips,
            "clade": self.clade,
        }


def parse_substitution(text: str) -> tuple[int, str]:
    """``223V`` -> (223, "V"). A from-state, a zero position or a lower-case residue is refused."""
    match = _SUBSTITUTION.fullmatch(text.strip())
    if match is None:
        raise TrimError(
            f"substitution {text!r}: expected <position><residue>, e.g. 223V (any from-state)"
        )
    return int(match["position"]), match["to"]


def load_trim_rules(path: Path) -> dict[str, TrimRule]:
    """Read ``trees/trim.tsv``: one row per subtype, every column required.

    Comment lines are dropped before the header is read, as in ``trees/placement.tsv``, so the
    file can say where its rows came from.
    """
    path = Path(path)
    body = [line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")]
    rows = list(csv.DictReader(body, delimiter="\t"))
    if not rows:
        raise TrimError(f"{path}: no rows")
    missing = [c for c in COLUMNS if c not in rows[0]]
    if missing:
        raise TrimError(f"{path}: missing column(s) {missing}; expected {list(COLUMNS)}")
    rules: dict[str, TrimRule] = {}
    for row in rows:
        subtype = (row["subtype"] or "").strip()
        reason = (row["reason"] or "").strip()
        if not subtype:
            raise TrimError(f"{path}: a row has no subtype")
        if subtype in rules:
            raise TrimError(f"{path}: {subtype} appears twice")
        if not reason:
            raise TrimError(f"{path}: {subtype} has no reason (design rule 11)")
        try:
            position, to = parse_substitution(row["substitution"] or "")
        except TrimError as error:
            raise TrimError(f"{path}: {subtype}: {error}") from None
        try:
            min_tips = int(row["min_tips"])
        except (TypeError, ValueError) as error:
            raise TrimError(f"{path}: {subtype}: min_tips: {error}") from error
        if min_tips < 2:
            raise TrimError(f"{path}: {subtype}: min_tips {min_tips}; a subtree has at least 2")
        rules[subtype] = TrimRule(subtype, position, to, min_tips, reason)
    return rules


def rule_for(rules: Mapping[str, TrimRule], subtype: str) -> TrimRule:
    """The subtype's rule. A trim asked for with no row is an error, never a silent no-trim."""
    try:
        return rules[subtype]
    except KeyError:
        known = ", ".join(sorted(rules)) or "none"
        raise TrimError(
            f"no trim rule for {subtype!r}; the file has: {known}. Leave --trim off for a "
            "subtype that is not trimmed"
        ) from None


def _tips(populated: PopulatedTree) -> dict[int, int]:
    """Leaves under every node, keyed by ``id(node)``."""
    tips: dict[int, int] = {}
    for node in populated.tree.postorder():
        tips[id(node)] = 1 if node.is_leaf else sum(tips[id(c)] for c in node.children)
    return tips


def _keys_under(node: Node) -> frozenset[str]:
    out: set[str] = set()
    stack = [node]
    while stack:
        item = stack.pop()
        if item.is_leaf:
            out.add(item.name or "")
        stack.extend(item.children)
    return frozenset(out)


def find_trim_node(populated: PopulatedTree, rule: TrimRule) -> TrimMatch:
    """The one branch carrying the rule's substitution with at least ``min_tips`` leaves."""
    if populated.subtype != rule.subtype:
        raise TrimError(f"rule for {rule.subtype} applied to a {populated.subtype} tree")
    tips = _tips(populated)
    carrying = [
        node
        for node in populated.tree.preorder()
        if node.parent is not None and rule.carried_by(populated.aa_subs.get(node.node_id, []))
    ]
    carrying.sort(key=lambda node: -tips[id(node)])
    chosen = [node for node in carrying if tips[id(node)] >= rule.min_tips]

    def clade(node: Node) -> str | None:
        call = populated.clades.get(node.id_hex)
        return call.clade if call is not None else None

    if len(chosen) != 1:
        listed = ", ".join(
            f"{node.id_hex} ({tips[id(node)]} tips, clade {clade(node) or '-'})"
            for node in carrying[:10]
        )
        more = f" and {len(carrying) - 10} more" if len(carrying) > 10 else ""
        raise TrimError(
            f"{rule.subtype}: {len(chosen)} branches carry {rule.substitution} with at least "
            f"{rule.min_tips} tips; the trim needs exactly one. Branches carrying it, largest "
            f"first: {listed or 'none'}{more}"
        )
    node = chosen[0]
    keys = _keys_under(node)
    outgroup = populated.counts.get("outgroup")
    if outgroup is not None and outgroup in keys:
        raise TrimError(f"{rule.subtype}: the {rule.substitution} branch contains the outgroup")
    return TrimMatch(rule, node.id_hex, tips[id(node)], clade(node), keys)


__all__ = [
    "TrimError",
    "TrimMatch",
    "TrimRule",
    "find_trim_node",
    "load_trim_rules",
    "parse_substitution",
    "rule_for",
]
