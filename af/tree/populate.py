"""Populate a built tree: leaf metadata, ancestral states, substitutions, clades, branch scale.

The build step (:mod:`af.tree.build`) produces topology and CMAPLE lengths. Everything a consumer
reads *about* the tree is attached here, in one pass, so the tree store holds one fact in one place
(interface I6, ``notes/trees/I6-DRAFT.md``):

* **leaf metadata** from the sequence store (I3): names, the date *with its precision and
  interval* (a year-only date is not a 1 January measurement; see the sequences workstream's
  finding on GISAID's 1-January expansion), country, region, continent;
* **ancestral states** from the configured backend (:mod:`af.tree.asr`), and the amino-acid and
  nucleotide substitutions on every branch;
* **clades** on every node, from workstream 4's engine, passed in as a callable so the tree store
  never reaches for an environment variable or a global;
* **the branch-length scale**, chosen per subtype in config (Sarah, 24 Sep 2026): ``"ml"`` keeps
  CMAPLE's lengths; ``"mutations"`` gives the delivered h3/B-Vic scale, nucleotide changes / L
  (measured: 101,217 of 107,152 h3 branches, ``notes/trees/COMPARISON.md`` §2). Both are always
  stored, so a consumer can switch without a rebuild.

Every step is counted, and a leaf with no metadata is an error, never a blank row.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from af.tree.asr.base import AncestralStates, Backend, translate
from af.tree.model import Node, Tree, TreeError

BranchScale = Literal["ml", "mutations"]
BRANCH_SCALES: tuple[BranchScale, ...] = ("ml", "mutations")
DatePrecision = Literal["day", "month", "year"]

_BASES = frozenset("ACGT")


class PopulateError(ValueError):
    """The tree and its inputs do not fit together (missing leaf metadata, missing states)."""


def leaf_key(epi_isl: str, accession: str) -> str:
    """A leaf's key: EPI_ISL plus the segment accession (Sarah: ids, not hashes).

    The pair, not EPI_ISL alone: an isolate id repeats per linked segment accession, and a few of
    those carry genuinely different sequences.
    """
    if not epi_isl or not accession:
        raise PopulateError(f"a leaf key needs both ids, got {epi_isl!r} and {accession!r}")
    return f"{epi_isl}|{accession}"


@dataclass(frozen=True)
class LeafRecord:
    """What the tree carries about one sequence, read from the sequence store (I3)."""

    epi_isl: str
    accession: str
    name: str
    nucleotides: str
    collection_date: datetime.date | None = None
    date_precision: DatePrecision | None = None
    collection_date_first: datetime.date | None = None
    collection_date_last: datetime.date | None = None
    country: str | None = None
    region: str | None = None

    @property
    def key(self) -> str:
        return leaf_key(self.epi_isl, self.accession)


@dataclass(frozen=True)
class CladeInput:
    """One node as the clade engine needs it. Ids are the tree's stable hex ids."""

    node_id: str
    parent_id: str | None
    nucleotides: str
    is_leaf: bool
    gaps_reconstructed: bool


@dataclass(frozen=True)
class CladeCall:
    """One node's clade and the evidence behind it (workstream 4's ``Assignment``)."""

    clade: str | None
    support: int = 0
    unobservable: int = 0
    inherited: bool = False


@dataclass(frozen=True)
class CladeResult:
    """What the clade engine returns: calls by node id, and the clade set that made them.

    ``parents`` (clade -> parent clade) travels with the calls so a consumer resolves ancestry
    against the same nomenclature pin that assigned the labels, never a different one.
    """

    version: str
    calls: Mapping[str, CladeCall]
    parents: Mapping[str, str | None] = field(default_factory=dict)


CladeAssigner = Callable[[Sequence[CladeInput]], CladeResult]
"""Nodes in, calls keyed by node id out. See :func:`af_clades_assigner`."""


@dataclass
class PopulatedTree:
    """A finished tree and everything I6 writes about it. ``tree`` is ladderized with ids."""

    tree: Tree
    subtype: str
    alignment_length: int
    branch_scale: BranchScale
    leaves: dict[str, LeafRecord]
    states: AncestralStates | None
    ml_lengths: dict[int, float]
    nuc_changes: dict[int, int]
    aa: dict[int, str]
    aa_subs: dict[int, list[str]]
    nuc_subs: dict[int, list[str]]
    clades: dict[str, CladeCall] = field(default_factory=dict)
    clade_set_version: str | None = None
    clade_parents: dict[str, str | None] = field(default_factory=dict)
    continents: dict[str, str | None] = field(default_factory=dict)
    flags: dict[int, list[str]] = field(default_factory=dict)
    titrated: dict[str, bool] = field(default_factory=dict)
    titrated_by: dict[str, list[str]] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    gaps_reconstructed: bool = True


def nucleotide_changes(above: str, below: str) -> list[tuple[int, str, str]]:
    """Positions (1-based) where both ends are a definite base and they differ.

    Gaps and ambiguity codes never count. That matches what the delivered trees' mutation lengths
    counted: B/Vic's gap-coded deletion region (nt 484-492) was excluded there, and counting it
    would add a spurious 9 to nearly every B/Vic branch (``COMPARISON.md`` §2b).
    """
    if above == below:  # most branches on a dense tree: skip the per-site walk
        return []
    return [
        (index + 1, first, second)
        for index, (first, second) in enumerate(zip(above.upper(), below.upper(), strict=False))
        if first != second and first in _BASES and second in _BASES
    ]


def populate(
    tree: Tree,
    subtype: str,
    leaves: Mapping[str, LeafRecord],
    states: AncestralStates | None,
    *,
    branch_scale: BranchScale = "ml",
    backend: Backend | None = None,
    collapse_unchanged: bool | None = None,
    assign_clades: CladeAssigner | None = None,
    continent_of: Callable[[LeafRecord], str | None] | None = None,
    outgroup: str | None = None,
) -> PopulatedTree:
    """Attach everything I6 carries to a finished tree (see the module docstring).

    ``leaves`` is keyed by :func:`leaf_key`, which is also the leaf's name in the tree. Records for
    sequences not in the tree are ignored and counted. ``states`` may be None only with the
    ``"ml"`` scale and no clade assignment: without ancestral states there are no substitutions.

    ``collapse_unchanged`` (default: on for the ``"mutations"`` scale) removes internal branches
    that carry no nucleotide change. The delivered h3/B-Vic trees have none (``COMPARISON.md`` §1);
    with the ML scale such branches are rare and are kept unless asked.
    """
    if branch_scale not in BRANCH_SCALES:
        raise PopulateError(
            f"unknown branch scale {branch_scale!r}; expected one of {BRANCH_SCALES}"
        )
    if not tree.ids_assigned():
        raise TreeError("assign ids (af.tree.build.finish_tree) before populating")
    if states is None and (branch_scale == "mutations" or assign_clades is not None):
        raise PopulateError(
            f"branch scale {branch_scale!r} and clade assignment need ancestral states; none given"
        )
    if collapse_unchanged is None:
        collapse_unchanged = branch_scale == "mutations"
    counts: dict[str, Any] = {}
    if outgroup is not None:
        if not any(leaf.name == outgroup for leaf in tree.root.children):
            raise PopulateError(f"outgroup {outgroup!r} is not a child of the root")
        counts["outgroup"] = outgroup

    tree_keys = [leaf.name or "" for leaf in tree.leaves()]
    missing = [key for key in tree_keys if key not in leaves]
    if missing:
        raise PopulateError(
            f"{len(missing)} of {len(tree_keys)} tree leaves have no sequence-store record, "
            f"e.g. {missing[:3]}; every leaf must be described, none left blank"
        )
    counts["leaves"] = len(tree_keys)
    counts["records_not_in_tree"] = len(leaves) - len(tree_keys)
    lengths = {len(leaves[key].nucleotides) for key in tree_keys}
    if len(lengths) != 1:
        raise PopulateError(f"leaf sequences are not aligned: lengths {sorted(lengths)[:5]}")
    alignment_length = lengths.pop()

    def sequence_of(node: Node) -> str | None:
        if node.is_leaf:
            return leaves[node.name or ""].nucleotides
        assert states is not None
        return states.nucleotides.get(node.node_id)

    if states is not None:
        absent = [node.id_hex for node in tree.internal() if node.node_id not in states.nucleotides]
        if absent:
            raise PopulateError(
                f"{len(absent)} internal nodes have no ancestral state (backend "
                f"{states.backend}), e.g. {absent[:3]}; a backend that drops nodes must fail here"
            )

    def branch_changes() -> dict[int, list[tuple[int, str, str]]]:
        out: dict[int, list[tuple[int, str, str]]] = {}
        for node in tree.preorder():
            if node.parent is not None:
                above, below = sequence_of(node.parent), sequence_of(node)
                assert above is not None and below is not None
                out[id(node)] = nucleotide_changes(above, below)
        return out

    changes = branch_changes() if states is not None else {}
    collapsed = 0
    if collapse_unchanged:
        # As ape::di2multi does, the collapsed branch's own ML length is dropped, not pushed down.
        for node in list(tree.postorder()):
            if not node.is_leaf and node.parent is not None and not changes[id(node)]:
                collapsed += _collapse_one(node)
        if collapsed:
            tree.ladderize()
            tree.assign_ids()
            # A collapsed node can differ from its parent at gaps or ambiguities, which do not
            # count as changes, so its children's changes are recomputed against the new parent.
            changes = branch_changes()
    counts["unchanged_branches_collapsed"] = collapsed

    # A gap-blind backend (raxml-ng, IQ-TREE) puts a residue where the leaves have a deletion, so
    # residue<->gap differences are unobservable there, not substitutions: counted on the round's
    # B/Vic raxml states they add ~116,000 spurious 162-164 changes, one per deleted leaf codon.
    gaps_reconstructed = backend.reconstructs_gaps if backend is not None else True
    ml_lengths: dict[int, float] = {}
    nuc_changes: dict[int, int] = {}
    nuc_subs: dict[int, list[str]] = {}
    aa: dict[int, str] = {}
    aa_subs: dict[int, list[str]] = {}
    proteins: dict[int, str] = {}
    translations: dict[str, str] = {}
    for node in tree.preorder():
        ml_lengths[node.node_id] = node.branch_length
        sequence = sequence_of(node)
        if sequence is not None:
            protein = translations.get(sequence)
            if protein is None:  # identical sequences are common and translate once
                protein = translations[sequence] = translate(sequence)
            proteins[node.node_id] = aa[node.node_id] = protein
        if node.parent is None or states is None:
            continue
        found = changes[id(node)]
        nuc_changes[node.node_id] = len(found)
        nuc_subs[node.node_id] = [f"{a}{pos}{b}" for pos, a, b in found]
        above, below = proteins[node.parent.node_id], proteins[node.node_id]
        aa_subs[node.node_id] = (
            []
            if above == below
            else [
                f"{a}{pos + 1}{b}"
                for pos, (a, b) in enumerate(zip(above, below, strict=False))
                if a != b and a != "X" and b != "X" and (gaps_reconstructed or "-" not in (a, b))
            ]
        )
    if states is not None:
        counts["nucleotide_changes"] = sum(nuc_changes.values())
        counts["aa_substitutions"] = sum(len(subs) for subs in aa_subs.values())

    if branch_scale == "mutations":
        for node in tree.preorder():
            if node.parent is not None:
                node.branch_length = nuc_changes[node.node_id] / alignment_length
    counts["branch_scale"] = branch_scale

    populated = PopulatedTree(
        tree=tree,
        subtype=subtype,
        alignment_length=alignment_length,
        branch_scale=branch_scale,
        leaves={key: leaves[key] for key in tree_keys},
        states=states,
        ml_lengths=ml_lengths,
        nuc_changes=nuc_changes,
        aa=aa,
        aa_subs=aa_subs,
        nuc_subs=nuc_subs,
        counts=counts,
        gaps_reconstructed=gaps_reconstructed,
    )

    if continent_of is not None:
        populated.continents = {
            key: continent_of(record) for key, record in populated.leaves.items()
        }
        counts["leaves_without_continent"] = sum(v is None for v in populated.continents.values())
        # Every value counted, so one outside the figure's legend vocabulary is visible.
        tally: dict[str, int] = {}
        for value in populated.continents.values():
            tally[value or ""] = tally.get(value or "", 0) + 1
        counts["continents"] = dict(sorted(tally.items()))

    if assign_clades is not None:
        inputs = [
            CladeInput(
                node_id=node.id_hex,
                parent_id=node.parent.id_hex if node.parent is not None else None,
                nucleotides=sequence_of(node) or "",
                is_leaf=node.is_leaf,
                gaps_reconstructed=True if node.is_leaf else gaps_reconstructed,
            )
            for node in tree.preorder()
        ]
        clade_result = assign_clades(inputs)
        version, calls = clade_result.version, clade_result.calls
        unassigned = [item.node_id for item in inputs if item.node_id not in calls]
        if unassigned:
            raise PopulateError(
                f"the clade engine returned no call for {len(unassigned)} nodes, "
                f"e.g. {unassigned[:3]}"
            )
        populated.clades = {item.node_id: calls[item.node_id] for item in inputs}
        populated.clade_set_version = version
        populated.clade_parents = dict(clade_result.parents)
        named = {call.clade for call in populated.clades.values() if call.clade}
        if populated.clade_parents and named - set(populated.clade_parents):
            unknown = sorted(named - set(populated.clade_parents))
            raise PopulateError(f"clades not in the clade set's hierarchy: {unknown[:5]}")
        leaf_calls = [populated.clades[node.id_hex] for node in tree.leaves()]
        counts["leaves_without_clade"] = sum(call.clade is None for call in leaf_calls)
        counts["clade_set_version"] = version

    return populated


def _collapse_one(node: Node) -> int:
    """Splice one internal node's children into its parent."""
    parent = node.parent
    assert parent is not None
    index = parent.children.index(node)
    for child in node.children:
        child.parent = parent
    parent.children[index : index + 1] = node.children
    return 1


def af_clades_assigner(clade_set: Any, *, require_gap_support: bool = True) -> CladeAssigner:
    """Adapt workstream 4's tree engine (``af.clades.assign.assign_tree``) to :data:`CladeAssigner`.

    Imported when called, so the tree package does not depend on ``af.clades`` at import time.
    Internal nodes from a gap-blind backend are marked ``GAP_BLIND``, which the engine refuses
    where deletions define clades unless ``require_gap_support`` is False.
    """
    import importlib

    assign = importlib.import_module("af.clades.assign")
    sequence = importlib.import_module("af.clades.sequence")

    def run(inputs: Sequence[CladeInput]) -> CladeResult:
        nodes = [
            assign.Node(
                name=item.node_id,
                parent=item.parent_id,
                sequence=sequence.AlignedSequence.from_nucleotides(
                    item.nucleotides,
                    gaps=sequence.GapSupport.OBSERVED
                    if item.gaps_reconstructed
                    else sequence.GapSupport.GAP_BLIND,
                ),
            )
            for item in inputs
        ]
        result = assign.assign_tree(nodes, clade_set, require_gap_support=require_gap_support)
        calls = {
            name: CladeCall(a.clade, a.support, a.unobservable, a.inherited)
            for name, a in result.assignments.items()
        }
        parents = {clade.name: clade.parent for clade in clade_set}
        return CladeResult(str(result.clade_set_version), calls, parents)

    return run


__all__ = [
    "BRANCH_SCALES",
    "BranchScale",
    "CladeAssigner",
    "CladeCall",
    "CladeInput",
    "CladeResult",
    "LeafRecord",
    "PopulateError",
    "PopulatedTree",
    "af_clades_assigner",
    "leaf_key",
    "nucleotide_changes",
    "populate",
]
