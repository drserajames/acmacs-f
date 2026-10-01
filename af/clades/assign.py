"""Assign clades by walking a tree whose nodes carry ancestral sequences.

Sarah chose this engine on 25 September 2026 over Nextclade and over matching signatures
sequence by sequence (``notes/clades/ENGINE-COMPARISON.md``): af applies the pinned
upstream nomenclature to af's own tree. Measured against Nextclade on the WHO CC trees it
agrees on 99.6% of H3 leaves, 99.2% of H1 and 99.8% of B/Vic, where the old
``clades.json`` matcher agreed on 85-94%.

The rule, at every node from the root down::

    label(node) = the deepest clade that is the parent's label or a descendant of it,
                  whose cumulative signature the node's sequence does not contradict;
                  otherwise the parent's label.

Two consequences are the point of using a tree:

* **A tip cannot fall out of its clade.** A virus that reverts a defining position keeps
  the clade its ancestors established. Signature matching drops it instead, and moves the
  virus to its parent clade: of 738 H3 viruses carrying a substitution at one of their
  clade's defining positions, this engine keeps 628 in the clade, against 375 under ae.
* **Descent is never lost.** A clade can only be assigned within its parent's clade, so
  the result is always consistent with the hierarchy.

Sequences with no place on a tree — map antigens before the tree is built, viruses that
never made it into one — are assigned by the fallback engine instead
(:mod:`af.clades.fallback`), which Sarah chose to be Nextclade.

**Known limitation: a sibling clade cannot be recovered once the walk has gone down the
wrong branch, and this happens silently.** Candidates at a node are the inherited clade and
its descendants only, so a node whose sequence fully matches a *sibling* of its inherited
clade is never tested against that sibling. Example (October 2026, an H3 tree whose ancestral
states put a sister clade's marker too high): a node of 3,700 leaves gained the sibling
clade's own defining substitution on its edge and matched that sibling's whole signature with
the same support as its inherited clade, yet kept the inherited label, and every leaf below
it followed; the sibling clade was assigned nowhere. Labelling is deliberately left as it is
(changing candidate selection would move every tree's labels); instead
:func:`sibling_conflicts` reports every such node, and :attr:`TreeAssignment.ties` every
exact tie between candidates that only the name decided.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace

from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence, Evidence, GapSupport


class AssignmentError(ValueError):
    """The tree or its sequences cannot support clade assignment."""


@dataclass(frozen=True)
class NameTie:
    """A node where candidates tied on everything but the name; ``chosen`` won by name."""

    node: str
    chosen: str
    others: tuple[str, ...]


@dataclass(frozen=True)
class SiblingConflict:
    """A node labelled ``label`` whose sequence also fully matches ``other``, a clade outside
    ``label``'s lineage, with at least the same support: the walk may have gone down the
    wrong branch above it (see the module docstring)."""

    node: str
    label: str
    other: str
    leaves: int
    label_support: int
    """Matching loci of ``label`` at this node; -1 when the node contradicts it."""
    other_support: int
    gained_on_edge: bool
    """``other``'s own defining loci match here but not all at the parent."""
    topmost: bool
    """No ancestor of this node has the same (label, other) conflict."""


@dataclass(frozen=True)
class Node:
    """One node of the tree the engine walks.

    ``name`` identifies the node within its tree. ``sequence`` is the observed sequence
    for a leaf, or the reconstructed one for an internal node.
    """

    name: str
    parent: str | None
    sequence: AlignedSequence


@dataclass(frozen=True)
class Assignment:
    """The clade of one node, and how much of the clade's signature its sequence showed.

    ``support`` and ``unobservable`` are kept because they are what a reviewer needs when
    an assignment looks wrong: a clade assigned on two matching positions out of five,
    with three unobservable, is a different claim from one assigned on all five.
    """

    node: str
    clade: str | None
    support: int = 0
    unobservable: int = 0
    inherited: bool = False

    @property
    def assigned(self) -> bool:
        return self.clade is not None


@dataclass(frozen=True)
class TreeAssignment:
    """Every node's clade, with the counts a review page reports."""

    clade_set_version: str
    subtype: str
    assignments: Mapping[str, Assignment]
    founders: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    ties: tuple[NameTie, ...] = ()
    """Nodes where two candidates tied on standing, depth and support, so the name chose."""

    def clade(self, node: str) -> str | None:
        try:
            return self.assignments[node].clade
        except KeyError:
            raise KeyError(f"no assignment for node {node!r}") from None

    def counts(self) -> dict[str, int]:
        """How many nodes carry each clade; unassigned nodes count under ``""``."""
        counts: dict[str, int] = {}
        for assignment in self.assignments.values():
            counts[assignment.clade or ""] = counts.get(assignment.clade or "", 0) + 1
        return counts

    def convergent(self) -> dict[str, tuple[str, ...]]:
        """Clades established at more than one place in the tree.

        A clade with several founders is either genuine convergence or a sign that its
        definition is too loose to identify one lineage; either way a reviewer should see
        it rather than have it averaged away.
        """
        return {clade: nodes for clade, nodes in self.founders.items() if len(nodes) > 1}


def assign_tree(
    nodes: Iterable[Node],
    clade_set: CladeSet,
    *,
    require_gap_support: bool = True,
) -> TreeAssignment:
    """Assign a clade to every node of one tree.

    ``require_gap_support`` refuses a gap-blind reconstruction when the nomenclature
    defines clades by deletions, because the affected clades would then simply never be
    assigned: B/Vic has six such mutations, and fed raxml-ng states the engine lost 9% of
    its agreement with Nextclade before the loci were excluded. Set it to False to
    proceed knowingly, with the loci treated as unobservable.
    """
    by_name = {node.name: node for node in nodes}
    if not by_name:
        raise AssignmentError("no nodes to assign")
    _check_gap_support(by_name.values(), clade_set, require_gap_support)
    children = _children(by_name)
    roots = [name for name, node in by_name.items() if node.parent not in by_name]
    if not roots:
        raise AssignmentError("every node has a parent inside the tree: the tree has a cycle")

    assignments: dict[str, Assignment] = {}
    founders: dict[str, list[str]] = {}
    ties: list[NameTie] = []
    stack: list[tuple[str, str | None]] = [(root, None) for root in roots]
    while stack:
        name, inherited = stack.pop()
        assignment = _assign_node(by_name[name], inherited, clade_set, ties)
        assignments[name] = assignment
        if assignment.clade is not None and assignment.clade != inherited:
            founders.setdefault(assignment.clade, []).append(name)
        stack.extend((child, assignment.clade) for child in children.get(name, ()))
    if len(assignments) != len(by_name):
        unreached = sorted(set(by_name) - set(assignments))
        raise AssignmentError(
            f"{len(unreached)} nodes are not reachable from a root: {unreached[:5]}"
        )
    return TreeAssignment(
        clade_set_version=clade_set.version,
        subtype=clade_set.subtype,
        assignments=assignments,
        founders={clade: tuple(nodes) for clade, nodes in founders.items()},
        ties=tuple(ties),
    )


def _assign_node(
    node: Node,
    inherited: str | None,
    clade_set: CladeSet,
    ties: list[NameTie] | None = None,
) -> Assignment:
    """The deepest candidate the sequence does not contradict, else the inherited clade.

    A tie on standing, depth and support is decided by the name, deterministically; each
    such tie is appended to ``ties`` so it is reported rather than silent.
    """
    # Restrict to the inherited clade's subtree — but by its deepest *published* ancestor.
    # A label with no published ancestry (a locally defined pre-nomenclature lineage) makes
    # no claim about where the virus sits in the nomenclature, so it must not shut the
    # published clades out: when it did, two local clades captured 107,292 H1 leaves that
    # upstream names.
    anchor = _published_anchor(inherited, clade_set)
    candidates = clade_set.descendants(anchor) if anchor else clade_set.names
    best: Assignment | None = None
    best_key: tuple[int, int, int, str] | None = None
    tied: list[str] = []
    for candidate in candidates:
        if clade_set[candidate].revoked and candidate != inherited:
            continue
        verdict = _test(node, clade_set, candidate)
        if verdict is None:
            continue
        support, unobservable, own_support = verdict
        if own_support == 0 and candidate != inherited:
            # Nothing on this clade's own branch is observable here, so the sequence
            # cannot be told apart from the parent's. Assigning it anyway would hand
            # every virus to whichever clade happens to be defined only by loci the
            # sequence cannot carry (a signal peptide, or a deletion under a gap-blind
            # reconstruction), purely because it is deeper.
            continue
        # Published standing first, then depth, then supporting evidence, then the name so
        # that sister clades never break ties by dictionary order (design rule 8).
        # Without the first term a locally defined clade attached at the root outranks the
        # nomenclature itself: measured on the real H1 tree, adding two local clades moved
        # 107,292 leaves that upstream does name.
        key = (
            clade_set.upstream_depth(candidate),
            clade_set.depth(candidate),
            support,
            candidate,
        )
        if best_key is not None and key[:3] == best_key[:3]:
            tied.append(candidate if key < best_key else best_key[3])
        elif best_key is None or key[:3] > best_key[:3]:
            tied = []
        if best_key is None or key > best_key:
            best, best_key = Assignment(node.name, candidate, support, unobservable), key
    if best is None:
        return Assignment(node.name, inherited, inherited=True)
    if tied and ties is not None:
        ties.append(NameTie(node.name, str(best.clade), tuple(sorted(set(tied) - {best.clade}))))
    return replace(best, inherited=best.clade == inherited)


def _published_anchor(inherited: str | None, clade_set: CladeSet) -> str | None:
    """The deepest published clade at or above ``inherited``, or None if there is none."""
    if inherited is None:
        return None
    for candidate in (inherited, *clade_set.ancestors(inherited)):
        if not clade_set.is_local(candidate):
            return candidate
    return None


def _test(node: Node, clade_set: CladeSet, candidate: str) -> tuple[int, int, int] | None:
    """Test a clade's whole signature against a sequence.

    Returns matching loci, unobservable loci, and how many of the matches are on the
    clade's **own** branch, or None if the sequence contradicts any locus.
    """
    own = {(position.alphabet, position.position) for position in clade_set[candidate].mutations}
    support = unobservable = own_support = 0
    for (alphabet, position), state in clade_set.cumulative(candidate).items():
        evidence = node.sequence.evidence(alphabet, position, state)
        if evidence is Evidence.CONTRADICTS:
            return None
        if evidence is Evidence.MATCHES:
            support += 1
            if (alphabet, position) in own:
                own_support += 1
        else:
            unobservable += 1
    return support, unobservable, own_support


def _children(by_name: Mapping[str, Node]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for name, node in by_name.items():
        parent = node.parent
        if parent is not None and parent in by_name:
            children.setdefault(parent, []).append(name)
    for names in children.values():
        names.sort()
    return children


def _check_gap_support(nodes: Iterable[Node], clade_set: CladeSet, required: bool) -> None:
    """Refuse a gap-blind reconstruction when clades are defined by deletions."""
    deletion_clades = sorted(
        clade.name
        for clade in clade_set
        if any(position.state == "-" for position in clade.mutations)
    )
    if not deletion_clades:
        return
    blind = [node.name for node in nodes if node.sequence.gaps is GapSupport.GAP_BLIND]
    if not blind:
        return
    message = (
        f"{clade_set.subtype} defines {len(deletion_clades)} clades by deletions "
        f"({', '.join(deletion_clades[:4])}...), but {len(blind)} nodes come from a "
        "reconstruction that cannot represent a gap. Use an indel-aware backend "
        "(TreeTime), or pass require_gap_support=False to treat those loci as unobservable."
    )
    if required:
        raise AssignmentError(message)


def assign_sequences(
    sequences: Mapping[str, AlignedSequence],
    clade_set: CladeSet,
    *,
    fallback: Callable[[Mapping[str, AlignedSequence], CladeSet], Mapping[str, str | None]],
) -> dict[str, Assignment]:
    """Assign clades to sequences that are not on a tree, through ``fallback``.

    Kept explicit rather than quietly reusing the tree engine on a single sequence:
    without ancestry there is nothing to stop a reversion dropping a virus out of its
    clade, which is the failure the tree engine exists to avoid. The caller must choose
    the fallback (Sarah chose Nextclade), and the result records which engine was used.
    """
    labels = fallback(sequences, clade_set)
    unknown = sorted(set(labels) - set(sequences))
    if unknown:
        raise AssignmentError(f"fallback returned labels for unknown sequences: {unknown[:5]}")
    return {
        name: Assignment(node=name, clade=labels.get(name), inherited=False) for name in sequences
    }


def published_labels_changed(
    before: TreeAssignment, after: TreeAssignment, clade_set: CladeSet
) -> dict[str, tuple[str | None, str | None]]:
    """Nodes whose **published** clade differs between two assignments of one tree.

    The check to run whenever the local layer changes: adding local definitions may give
    a name to viruses the nomenclature leaves unnamed, and may refine a published clade
    into a local child of it, but it must never change what published clade a virus is
    in. A local clade attached at the root once did exactly that to 107,292 leaves of the
    real H1 tree, and the symptom — every leaf's clade changing at once — looks from a
    consumer's side like a new nomenclature rather than a bug.

    A node is reported when the deepest published clade at or above its label changes.
    Moving from no label to a local one, or from a published clade to a local child of
    it, is not a change; moving from one published clade to another is.
    """
    changed: dict[str, tuple[str | None, str | None]] = {}
    for node, assignment in after.assignments.items():
        previous = before.assignments.get(node)
        if previous is None:
            continue
        was = _published_anchor(previous.clade, clade_set)
        now = _published_anchor(assignment.clade, clade_set)
        if was != now:
            changed[node] = (was, now)
    return changed


def sibling_conflicts(
    nodes: Iterable[Node], result: TreeAssignment, clade_set: CladeSet
) -> tuple[SiblingConflict, ...]:
    """Every node labelled X whose sequence also fully matches a clade Y outside X's lineage
    (neither X, nor an ancestor, nor a descendant of X), with support at least X's.

    "Fully matches" is the engine's own test: Y's whole signature uncontradicted and at least
    one of Y's own loci observed. Reported, never acted on (the module docstring says why).
    Y's own loci are checked first, so the full test runs only where Y could match at all.
    """
    by_name = {node.name: node for node in nodes}
    children = _children(by_name)
    leaves = _leaf_counts(by_name, children)
    own = {
        name: tuple(clade_set[name].mutations)
        for name in clade_set.names
        if not clade_set[name].revoked and clade_set[name].mutations
    }
    lineage = {
        name: {name, *clade_set.ancestors(name), *clade_set.descendants(name)}
        for name in clade_set.names
    }
    found: list[SiblingConflict] = []
    flagged: dict[str, set[tuple[str, str]]] = {}
    order = _preorder(by_name, children)
    for name in order:
        node = by_name[name]
        label = result.assignments[name].clade
        parent_flags = flagged.get(node.parent or "", set())
        here: set[tuple[str, str]] = set()
        if label is not None:
            mine = _test(node, clade_set, label)
            label_support = mine[0] if mine is not None else -1
            for other, mutations in own.items():
                if other in lineage[label] or not _own_observed(node, mutations):
                    continue
                verdict = _test(node, clade_set, other)
                if verdict is None or verdict[0] < label_support:
                    continue
                parent = by_name.get(node.parent) if node.parent else None
                gained = parent is not None and not _own_observed(parent, mutations, every=True)
                here.add((label, other))
                found.append(
                    SiblingConflict(
                        name,
                        label,
                        other,
                        leaves[name],
                        label_support,
                        verdict[0],
                        gained,
                        (label, other) not in parent_flags,
                    )
                )
        flagged[name] = here
    return tuple(found)


def _own_observed(node: Node, mutations: tuple, *, every: bool = False) -> bool:
    """Whether the node matches its own loci: none contradicted and one observed (or, with
    ``every``, all of them matched)."""
    matched = 0
    for position in mutations:
        evidence = node.sequence.evidence(position.alphabet, position.position, position.state)
        if evidence is Evidence.CONTRADICTS:
            return False
        if evidence is Evidence.MATCHES:
            matched += 1
        elif every:
            return False
    return matched == len(mutations) if every else matched > 0


def _preorder(by_name: Mapping[str, Node], children: Mapping[str, list[str]]) -> list[str]:
    roots = [name for name, node in by_name.items() if node.parent not in by_name]
    order: list[str] = []
    stack = list(roots)
    while stack:
        name = stack.pop()
        order.append(name)
        stack.extend(children.get(name, ()))
    return order


def _leaf_counts(by_name: Mapping[str, Node], children: Mapping[str, list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in reversed(_preorder(by_name, children)):
        below = children.get(name, ())
        counts[name] = sum(counts[child] for child in below) if below else 1
    return counts


#: How many of the largest topmost conflicts a review keeps; the summary counts cover all.
REVIEW_KEEP = 20


def conflict_review(
    conflicts: Iterable[SiblingConflict], ties: Iterable[NameTie], *, keep: int = REVIEW_KEEP
) -> dict[str, object]:
    """A tree's sibling-conflict review, as it is written into the tree's metadata.

    Counts always cover every conflict; the list keeps the ``keep`` largest *topmost* ones
    (a conflict's descendants repeat it) so the metadata stays readable and diff-able.
    """
    found = list(conflicts)
    top = sorted((c for c in found if c.topmost), key=lambda c: (-c.leaves, c.node))
    tied = list(ties)
    return {
        "counts": {
            "all": len(found),
            "multi_leaf": sum(1 for c in found if c.leaves > 1),
            "topmost": len(top),
            "topmost_multi_leaf": sum(1 for c in top if c.leaves > 1),
        },
        "topmost": [
            {
                "node": c.node,
                "label": c.label,
                "other": c.other,
                "leaves": c.leaves,
                "label_support": c.label_support,
                "other_support": c.other_support,
                "gained_on_edge": c.gained_on_edge,
            }
            for c in top[:keep]
        ],
        "topmost_truncated": max(0, len(top) - keep),
        "ties": {
            "count": len(tied),
            "examples": [
                {"node": t.node, "chosen": t.chosen, "others": list(t.others)} for t in tied[:keep]
            ],
        },
    }
