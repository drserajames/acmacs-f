"""Cutting the tree at a node: proposing it, pinning it, and finding it again after a rebuild.

Sarah, 25 September 2026: retirement should not be age-based, because that changes the tree's
structure. Instead the tree is cut at a node above which nothing has circulated for a while, the
node is chosen at the end of a VCM and held until the end of the next one, and the cut is applied to
the **selection** a tree is built from, not to a tree that already exists.

This module is the tree side of that: propose a node from a published tree, pin it so a later
rebuild can find the same place, and report whether circulation above it has resumed. Which node is
cut is Sarah's choice, from the table :func:`propose` prints; nothing here decides it and nothing
here cuts anything. Selection (workstream 2) consumes the pin through :func:`load_cut_pin` and never
touches a tree. The design, with the measurements behind every default, is
``notes/trees/CUT-NODE.md``.

Two findings from that note are load-bearing here:

- **A signature of reconstructed states is not a membership test.** Pinning the node by the states
  at the positions that changed along the path to it identified 26% of the H1 leaves actually below
  it, and at a tolerance loose enough to find 96% of them it admitted 3,881 sequences from above.
  The failure is structural: the descendants the cut is meant to keep are the ones that have drifted
  furthest at exactly those positions. The signature is carried in the pin for a person to read, and
  nothing here decides anything from it.
- **The MRCA of spread anchor leaves is exact**, and survives losing half of them, where randomly
  chosen anchors drift. So anchors are taken round-robin over the node's children, largest first.

The honest limit, unchanged from the design note: the MRCA pin is exact on the tree it was taken
from, and whether it survives a *rebuilt* topology is not established — that needs a second tree
(the incremental pair, task 5.7). The checks here (quorum, child span, leaf drift) exist to refuse a
recovery that has silently moved, not to promise that it will not move.
"""

from __future__ import annotations

import collections
import datetime
import tomllib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from af.seq.select import Key
from af.store import StoreRef
from af.tree.io import i6

WINDOW_MONTHS = 24
"""How far back "recently collected" reaches, counted from the newest collection date in the tree
and never from today (design rule 7). Measured at 24 months in ``CUT-NODE.md``."""

TOLERANCES = (0.0, 0.0001, 0.001, 0.005, 0.01, 0.05)
"""The sweep :func:`propose` reports. A tolerance is the fraction of recent leaves allowed to stay
above the cut. Tolerance 0 is kept in the table because it is instructive, not because it is
usable: it returns the root on H3 and H1, since a handful of recent leaves sit in old parts of
every tree, and a rule that must keep every recent leaf can never descend."""

ANCHORS = 20
"""Anchor leaves recorded in a pin. Measured: 20 spread anchors recover the node with half of them
gone, in all three subtypes."""

QUORUM = 15
"""Anchors that must still be found, of 20. Below this the run stops rather than recovering a node
from a thinned set: with one anchor left the MRCA is that leaf, which is always wrong."""

MAX_LEAF_DRIFT = 0.10
"""How far the recovered node's leaf count may differ from the recorded one before the recovery is
refused. A cut that has genuinely moved is a finding for the next VCM, not something af absorbs."""

MAX_BELOW_OUTSIDE = 0.10
"""How many of the pinned below-list leaves still in the tree may sit outside the recovered node.
This is what catches an MRCA that has descended into one child: the leaf count can drift for
legitimate reasons (the cycle's new viruses), but the viruses the cut kept last time should still be
under it."""

MAX_RESUMED = 0.05
"""Leaves kept because the cut never saw them, which then attached above the cut, as a fraction of
the tree. Above this the run stops: it means the retired part of the tree is growing back, which is
also what a backfill pull of old sequences looks like (``CUT-NODE.md`` §3c)."""

NODE_COLUMNS = [
    "node_id",
    "parent",
    "is_leaf",
    "n_leaves",
    "leaf_id",
    "epi_isl",
    "accession",
    "clade",
    "collection_date_last",
]


class CutError(ValueError):
    """A cut cannot be proposed, a pin is malformed, or a recovered cut has moved."""


# -------------------------------------------------------------------------------------------------
# The tree, as this module needs it


@dataclass(frozen=True)
class _Nodes:
    """The columns of ``nodes.parquet`` this module uses, with each subtree's extent and depth.

    Rows are pre-order, so a subtree is the contiguous block ``[row, row + extent[row])`` and a
    node's parent always has a lower row number. Both facts are relied on below, so :meth:`read`
    checks them rather than trusting them.
    """

    node_id: list[str]
    parent: list[int]
    is_leaf: list[bool]
    n_leaves: list[int]
    leaf_id: list[str | None]
    epi_isl: list[str | None]
    accession: list[str | None]
    clade: list[str | None]
    last: list[datetime.date | None]
    children: list[list[int]]
    extent: list[int]
    depth: list[int]

    @classmethod
    def read(cls, directory: Path) -> _Nodes:
        table = pq.read_table(Path(directory) / i6.NODES_FILE, columns=NODE_COLUMNS)
        rows = table.to_pydict()
        count = len(rows["node_id"])
        if not count:
            raise CutError(f"{directory}: the tree has no nodes")
        parent = list(rows["parent"])
        if parent[0] != -1:
            raise CutError(f"{directory}: row 0 is not the root; this reader needs pre-order rows")
        children: list[list[int]] = [[] for _ in range(count)]
        extent = [1] * count
        depth = [0] * count
        for row in range(1, count):
            if not 0 <= parent[row] < row:
                raise CutError(f"{directory}: node {row} has parent {parent[row]}; not pre-order")
            children[parent[row]].append(row)
            depth[row] = depth[parent[row]] + 1
        for row in range(count - 1, 0, -1):
            extent[parent[row]] += extent[row]
        return cls(
            node_id=list(rows["node_id"]),
            parent=parent,
            is_leaf=list(rows["is_leaf"]),
            n_leaves=list(rows["n_leaves"]),
            leaf_id=list(rows["leaf_id"]),
            epi_isl=list(rows["epi_isl"]),
            accession=list(rows["accession"]),
            clade=list(rows["clade"]),
            last=list(rows["collection_date_last"]),
            children=children,
            extent=extent,
            depth=depth,
        )

    def row_of(self, node_id: str) -> int:
        try:
            return self.node_id.index(node_id)
        except ValueError:
            raise CutError(f"no node {node_id!r} in this tree") from None

    def leaf_rows(self, row: int = 0) -> Iterator[int]:
        """The leaf rows of the subtree at ``row``."""
        return (other for other in range(row, row + self.extent[row]) if self.is_leaf[other])

    def key_of(self, row: int) -> Key:
        epi, accession = self.epi_isl[row], self.accession[row]
        if not epi or not accession:
            raise CutError(f"leaf row {row} has no EPI_ISL|accession")
        return (epi, accession)

    def sorted_children(self, row: int) -> list[int]:
        """Children largest first, ties by row, so anchor choice is reproducible (design rule 8)."""
        return sorted(self.children[row], key=lambda child: (-self.n_leaves[child], child))


def _months_before(day: datetime.date, months: int) -> datetime.date:
    """``day`` less ``months`` whole months, clamped into the target month (no dateutil)."""
    year, month = divmod(day.year * 12 + (day.month - 1) - months, 12)
    month += 1  # divmod gives 0-11
    first_of_next = datetime.date(year + month // 12, month % 12 + 1, 1)
    in_month = (first_of_next - datetime.timedelta(days=1)).day
    return datetime.date(year, month, min(day.day, in_month))


# -------------------------------------------------------------------------------------------------
# Proposing the node


@dataclass(frozen=True)
class CutProposal:
    """One tolerance's candidate cut node, and what cutting there would cost."""

    tolerance: float
    node_id: str
    clade: str | None
    leaves_below: int
    leaves_total: int
    recent_below: int
    recent_total: int
    children: int
    window_from: datetime.date
    window_to: datetime.date
    newest_above: datetime.date | None
    is_root: bool

    @property
    def kept(self) -> float:
        """Fraction of the tree's leaves below the cut."""
        return self.leaves_below / self.leaves_total if self.leaves_total else 0.0

    @property
    def recent_kept(self) -> float:
        return self.recent_below / self.recent_total if self.recent_total else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "node_id": self.node_id,
            "clade": self.clade,
            "leaves_below": self.leaves_below,
            "leaves_total": self.leaves_total,
            "leaves_above": self.leaves_total - self.leaves_below,
            "kept": round(self.kept, 6),
            "recent_below": self.recent_below,
            "recent_total": self.recent_total,
            "children": self.children,
            "recent_window": [str(self.window_from), str(self.window_to)],
            "newest_above": None if self.newest_above is None else str(self.newest_above),
            "is_root": self.is_root,
        }

    def describe(self) -> str:
        where = "the root (nothing would be cut)" if self.is_root else f"node {self.node_id}"
        if self.is_root:
            return (
                f"tolerance {self.tolerance:g}: {where}, so all {self.leaves_total} leaves and all "
                f"{self.recent_total} recent ones stay, and nothing is above the cut"
            )
        clade = f" [{self.clade}]" if self.clade else ""
        above = self.newest_above.isoformat() if self.newest_above else "none dated"
        return (
            f"tolerance {self.tolerance:g}: {where}{clade} keeps {self.leaves_below} of "
            f"{self.leaves_total} leaves ({100 * self.kept:.1f}%), {self.recent_below} of "
            f"{self.recent_total} recent ({100 * self.recent_kept:.2f}%), {self.children} "
            f"children; newest collection date above the cut {above}"
        )


def propose(
    directory: Path,
    *,
    window_months: int = WINDOW_MONTHS,
    tolerances: Sequence[float] = TOLERANCES,
) -> list[CutProposal]:
    """The candidate cut node at each tolerance, for a person to choose from.

    The rule: descend from the root to the deepest internal node that still has all but
    ``tolerance`` of the recently collected leaves below it.

    "Recent" is a window of ``window_months`` ending at the newest collection date *in the tree*,
    never today (design rule 7). Both date choices are deliberately in the fail-safe direction: the
    window ends at the newest date af is sure of (the newest ``collection_date_first``), and a leaf
    counts as recent when its ``collection_date_last`` falls inside it, so a year-precision record
    that *could* be recent is treated as recent. Being generous about what counts as recent makes
    the rule keep more, which is the safe way to be wrong about a virus that may still circulate.
    """
    nodes = _Nodes.read(directory)
    if any(tolerance < 0 or tolerance > 1 for tolerance in tolerances):
        raise CutError(f"tolerances must be fractions of the recent leaves: {list(tolerances)}")
    dated = [nodes.last[row] for row in nodes.leaf_rows() if nodes.last[row] is not None]
    if not dated:
        raise CutError(f"{directory}: no leaf has a collection date, so 'recent' has no meaning")
    window_to = max(day for day in dated if day is not None)
    window_from = _months_before(window_to, window_months)
    # A leaf with no date at all is not recent: there is nothing to put it in the window.
    recent = [
        1 if nodes.is_leaf[row] and (day := nodes.last[row]) and day >= window_from else 0
        for row in range(len(nodes.node_id))
    ]
    prefix = [0] * (len(recent) + 1)
    for row, value in enumerate(recent):
        prefix[row + 1] = prefix[row] + value

    def recent_under(row: int) -> int:
        return prefix[row + nodes.extent[row]] - prefix[row]

    total_recent = recent_under(0)
    if not total_recent:
        raise CutError(
            f"{directory}: no leaf was collected in the {window_months} months to {window_to}"
        )
    out: list[CutProposal] = []
    for tolerance in tolerances:
        need = total_recent - tolerance * total_recent
        row = 0
        while True:
            deeper = next(
                (
                    child
                    for child in nodes.sorted_children(row)
                    if not nodes.is_leaf[child] and recent_under(child) >= need
                ),
                None,
            )
            if deeper is None:
                break
            row = deeper
        out.append(
            CutProposal(
                tolerance=tolerance,
                node_id=nodes.node_id[row],
                clade=nodes.clade[row],
                leaves_below=nodes.n_leaves[row],
                leaves_total=nodes.n_leaves[0],
                recent_below=recent_under(row),
                recent_total=total_recent,
                children=len(nodes.children[row]),
                window_from=window_from,
                window_to=window_to,
                newest_above=_newest_above(nodes, row),
                is_root=row == 0,
            )
        )
    return out


def _newest_above(nodes: _Nodes, row: int) -> datetime.date | None:
    """The newest collection date among leaves the cut would remove: the resumption alarm."""
    block = range(row, row + nodes.extent[row])
    days = [
        nodes.last[other]
        for other in nodes.leaf_rows()
        if other not in block and nodes.last[other] is not None
    ]
    return max((day for day in days if day is not None), default=None)


# -------------------------------------------------------------------------------------------------
# Pinning it: the anchors and the below-list


def pick_anchors(directory: Path, node_id: str, *, count: int = ANCHORS) -> tuple[Key, ...]:
    """``count`` anchor leaves spread over the node's children, largest first.

    The MRCA of a set of leaves is the cut node only if the set spans at least two of the node's
    children; if every surviving anchor sits in one child, the MRCA descends into that child. So the
    anchors are taken branch by branch rather than at random, which measurably survives losing half
    of them where random anchors do not. Each branch contributes a typical leaf (descending into its
    largest child), and a branch already used is replaced by its own children, so with more anchors
    than children the spread goes one level deeper rather than clustering.
    """
    nodes = _Nodes.read(directory)
    row = nodes.row_of(node_id)
    if nodes.is_leaf[row]:
        raise CutError(f"{node_id} is a leaf; a cut node is internal")
    if count < 2:
        raise CutError(f"{count} anchors cannot span two children; a cut needs at least 2")
    if nodes.n_leaves[row] < count:
        raise CutError(f"{node_id} has {nodes.n_leaves[row]} leaves; {count} anchors asked for")
    taken: list[int] = []
    # One queue per child of the cut node, taken in turn: a single shared queue would be dominated
    # by whichever child was pushed first, which is exactly the clustering this is meant to avoid
    # (with 6 anchors over children of 6, 4 and 4 leaves a shared queue gave 4 / 1 / 1).
    queues = [collections.deque([child]) for child in nodes.sorted_children(row)]
    while len(taken) < count and any(queues):
        progressed = False
        for queue in queues:
            if len(taken) >= count:
                break
            leaf = _next_anchor(nodes, queue, taken)
            if leaf is not None:
                taken.append(leaf)
                progressed = True
        if not progressed:  # every branch is used up; the count check below reports it
            break
    if len(taken) < count:
        raise CutError(f"{node_id}: only {len(taken)} anchors could be chosen, {count} asked for")
    return tuple(nodes.key_of(leaf) for leaf in taken)


def _next_anchor(nodes: _Nodes, queue: collections.deque[int], taken: Sequence[int]) -> int | None:
    """One child's next unused anchor, widening the branch it came from for the round after."""
    while queue:
        branch = queue.popleft()
        if not nodes.is_leaf[branch]:
            queue.extend(nodes.sorted_children(branch))
        leaf = _representative(nodes, branch, taken)
        if leaf is not None:
            return leaf
    return None


def _representative(nodes: _Nodes, row: int, taken: Iterable[int]) -> int | None:
    """A typical unused leaf of the branch: descend into the largest child that still has one."""
    used = list(taken)
    while not nodes.is_leaf[row]:
        deeper = next(
            (child for child in nodes.sorted_children(row) if _has_free_leaf(nodes, child, used)),
            None,
        )
        if deeper is None:
            return None
        row = deeper
    return row if row not in used else None


def _has_free_leaf(nodes: _Nodes, row: int, used: Sequence[int]) -> bool:
    block = range(row, row + nodes.extent[row])
    return nodes.n_leaves[row] > sum(1 for leaf in used if leaf in block)


def below_keys(directory: Path, node_id: str) -> frozenset[Key]:
    """Every leaf below the node, as ``(EPI_ISL, accession)``: the pin's exact below-list.

    At the moment the node is chosen this set is known exactly, so the pin needs no inference and no
    threshold, and it survives anything that later happens to the topology.
    """
    nodes = _Nodes.read(directory)
    row = nodes.row_of(node_id)
    return frozenset(nodes.key_of(leaf) for leaf in nodes.leaf_rows(row))


@dataclass(frozen=True)
class CutPin:
    """One subtype's cut, as chosen at the end of one VCM and held for the next.

    ``below`` and ``chosen_from`` are what selection needs: keep a key when it is in ``below``, or
    when it was not in that sequences-store version at all (so the cut never saw it). ``anchors``
    and the two ``expected_`` counts are what :func:`recover` needs to find the node again and to
    refuse a node that has moved. ``clade``, ``signature`` and ``chosen_on`` are for a person
    reading the pin; nothing here decides anything from them.
    """

    subtype: str
    cycle: str
    node_id: str
    below: frozenset[Key]
    chosen_from: StoreRef
    anchors: tuple[Key, ...]
    quorum: int
    expected_leaves: int
    expected_children: int
    reason: str
    chosen_on: datetime.date | None = None
    clade: str | None = None
    signature: Mapping[int, str] = field(default_factory=dict)

    def keeps(self, key: Key, known_at_the_cut: bool) -> bool:
        """Whether selection keeps this key: below the cut, or unknown when the cut was chosen.

        ``known_at_the_cut`` is the caller's answer to "was this key in ``chosen_from``?" — it is
        the sequence store's question, not the tree's, so this module does not try to answer it.
        No date arithmetic: a virus collected before the cut but submitted after it is kept by the
        second test, which is what a collection-date window got wrong.
        """
        return key in self.below or not known_at_the_cut


def load_cut_pin(path: Path, cycle: str, subtype: str) -> CutPin:
    """One row of a cuts file (acmacs-f-data ``trees/cuts.toml``), with its below-list.

    A cycle and subtype with no row is an error, never a default: once config names a cycle the pin
    is not optional (design rules 1 and 4). Superseded rows stay in the file with the cycle they
    served, so a historical rebuild cuts where that round cut.
    """
    path = Path(path)
    data = tomllib.loads(path.read_text())
    rows = data.get("cut") or []
    if not isinstance(rows, list):
        raise CutError(f"{path}: [[cut]] must be a list of tables")
    found = [row for row in rows if row.get("cycle") == cycle and row.get("subtype") == subtype]
    if not found:
        have = ", ".join(sorted(f"{r.get('cycle')}/{r.get('subtype')}" for r in rows)) or "none"
        raise CutError(f"{path}: no cut for cycle {cycle!r} subtype {subtype!r}; it has: {have}")
    if len(found) > 1:
        raise CutError(f"{path}: {len(found)} cuts for cycle {cycle!r} subtype {subtype!r}")
    row = found[0]
    for required in ("node_id", "chosen_from", "anchors", "expected_leaves", "expected_children"):
        if required not in row:
            raise CutError(f"{path}: {cycle}/{subtype} has no {required}")
    if not str(row.get("reason", "")).strip():
        raise CutError(f"{path}: {cycle}/{subtype} has no reason (design rule 11)")
    below_file = row.get("below_file")
    if not below_file:
        raise CutError(f"{path}: {cycle}/{subtype} has no below_file")
    anchors = tuple(_parse_key(f"{path}: {cycle}/{subtype} anchor", a) for a in row["anchors"])
    if len(set(anchors)) != len(anchors):
        raise CutError(f"{path}: {cycle}/{subtype} lists the same anchor twice")
    quorum = int(row.get("quorum", QUORUM))
    if not 2 <= quorum <= len(anchors):
        raise CutError(
            f"{path}: {cycle}/{subtype} quorum {quorum} with {len(anchors)} anchors; it must be "
            f"at least 2 (one anchor's MRCA is that leaf) and at most the number of anchors"
        )
    below = read_below(path.parent / below_file)
    if row["expected_leaves"] != len(below):
        raise CutError(
            f"{path}: {cycle}/{subtype} records {row['expected_leaves']} leaves below the cut, "
            f"but {below_file} lists {len(below)}"
        )
    missing = [anchor for anchor in anchors if anchor not in below]
    if missing:
        raise CutError(
            f"{path}: {cycle}/{subtype}: {len(missing)} anchor(s) are not in the below-list, so "
            f"the pin contradicts itself"
        )
    chosen_on = row.get("chosen_on")
    return CutPin(
        subtype=subtype,
        cycle=cycle,
        node_id=str(row["node_id"]),
        below=below,
        chosen_from=StoreRef.from_json(row["chosen_from"]),
        anchors=anchors,
        quorum=quorum,
        expected_leaves=int(row["expected_leaves"]),
        expected_children=int(row["expected_children"]),
        reason=str(row["reason"]).strip(),
        chosen_on=chosen_on if isinstance(chosen_on, datetime.date) else None,
        clade=row.get("clade"),
        signature={int(k): str(v) for k, v in (row.get("signature") or {}).items()},
    )


def read_below(path: Path) -> frozenset[Key]:
    """A one-column file of ``EPI_ISL|accession`` leaf keys: the pinned below-list."""
    path = Path(path)
    if not path.exists():
        raise CutError(f"{path}: no below-list")
    keys = [
        _parse_key(f"{path}:{number}", line.split("\t")[0].strip())
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.strip() and not line.startswith("#") and line.split("\t")[0].strip() != "leaf_id"
    ]
    if not keys:
        raise CutError(f"{path}: no keys")
    return frozenset(keys)


def _parse_key(where: str, value: Any) -> Key:
    epi, _, accession = str(value).partition("|")
    if not epi or not accession:
        raise CutError(f"{where}: {value!r} is not EPI_ISL|accession")
    return (epi, accession)


# -------------------------------------------------------------------------------------------------
# Finding it again


@dataclass(frozen=True)
class RecoveredCut:
    """Where the pin's anchors put the cut in this tree, and how far it has moved."""

    node_id: str
    leaves_below: int
    children: int
    anchors_found: int
    anchors_missing: tuple[Key, ...]
    children_spanned: int
    expected_leaves: int
    expected_children: int
    below_present: int
    below_under: int
    node_id_changed: bool = False

    @property
    def leaf_drift(self) -> float:
        return abs(self.leaves_below - self.expected_leaves) / max(self.expected_leaves, 1)

    @property
    def below_outside(self) -> int:
        """Pinned below-list leaves still in the tree, but no longer under the cut."""
        return self.below_present - self.below_under

    @property
    def below_shortfall(self) -> float:
        return self.below_outside / self.below_present if self.below_present else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_id_changed": self.node_id_changed,
            "leaves_below": self.leaves_below,
            "expected_leaves": self.expected_leaves,
            "leaf_drift": round(self.leaf_drift, 6),
            "children": self.children,
            "expected_children": self.expected_children,
            "anchors_found": self.anchors_found,
            "anchors_missing": [f"{epi}|{acc}" for epi, acc in self.anchors_missing],
            "children_spanned": self.children_spanned,
            "below_present": self.below_present,
            "below_under": self.below_under,
            "below_outside": self.below_outside,
            "below_shortfall": round(self.below_shortfall, 6),
        }

    def describe(self) -> str:
        gone = (
            "all anchors found"
            if not self.anchors_missing
            else f"{len(self.anchors_missing)} anchor(s) missing: "
            + ", ".join(f"{epi}|{acc}" for epi, acc in self.anchors_missing)
        )
        which = "NOT the recorded node" if self.node_id_changed else "the recorded node"
        return (
            f"cut recovered at {self.node_id} ({which}): "
            f"{self.leaves_below} leaves below against {self.expected_leaves} recorded "
            f"(drift {100 * self.leaf_drift:.2f}%), {self.children} children against "
            f"{self.expected_children}, {self.anchors_found} anchors over "
            f"{self.children_spanned} children, {self.below_under} of {self.below_present} "
            f"surviving below-list leaves still under it; {gone}"
        )


def recover(
    directory: Path,
    pin: CutPin,
    *,
    max_leaf_drift: float = MAX_LEAF_DRIFT,
    max_below_outside: float = MAX_BELOW_OUTSIDE,
) -> RecoveredCut:
    """Find the pinned cut in this tree, as the MRCA of the anchors still present.

    Refuses, with the numbers in the message, when fewer anchors than the quorum are found, when the
    leaf count below the recovered node has drifted too far from the recorded one, or when too many
    of the pinned below-list leaves still in the tree now sit outside it. Missing anchors inside the
    quorum are carried in the result for the caller to report: GISAID records are withdrawn and
    renamed, so some loss is normal, but it is never silent. Anchors are never re-picked to make up
    the number — a replacement anchor is a new choice, and belongs to the next VCM.

    **Two checks the design note proposed are not here, because they cannot fail.** That every
    surviving anchor is below the recovered node is true by construction: the MRCA has all of them
    below it. That the survivors span at least two of the recovered node's children is also
    automatic, because the MRCA of two or more distinct leaves always has anchors in two of its
    children — the children spanned are *reported*, never refused on. The failure that note was
    describing (all survivors inside one child, so the MRCA descends into it) is real, and what
    catches it is the pair of checks on what is below the recovered node: the leaf count, and how
    much of the pinned below-list is still under it.
    """
    nodes = _Nodes.read(directory)
    rows_by_key = {nodes.key_of(leaf): leaf for leaf in nodes.leaf_rows()}
    found = [rows_by_key[anchor] for anchor in pin.anchors if anchor in rows_by_key]
    missing = tuple(anchor for anchor in pin.anchors if anchor not in rows_by_key)
    if len(found) < pin.quorum:
        raise CutError(
            f"{pin.cycle}/{pin.subtype}: {len(found)} of {len(pin.anchors)} anchors are in this "
            f"tree, under the quorum of {pin.quorum}. Missing: "
            f"{', '.join(f'{e}|{a}' for e, a in missing)}. The cut cannot be recovered from a "
            f"thinned anchor set; choosing new anchors is a decision for the next VCM"
        )
    row = _mrca(nodes, found)
    spanned = len({_child_towards(nodes, row, leaf) for leaf in found})
    in_tree = {nodes.key_of(leaf) for leaf in nodes.leaf_rows()}
    under = {nodes.key_of(leaf) for leaf in nodes.leaf_rows(row)}
    present = pin.below & in_tree
    result = RecoveredCut(
        node_id=nodes.node_id[row],
        leaves_below=nodes.n_leaves[row],
        children=len(nodes.children[row]),
        anchors_found=len(found),
        anchors_missing=missing,
        children_spanned=spanned,
        expected_leaves=pin.expected_leaves,
        expected_children=pin.expected_children,
        below_present=len(present),
        below_under=len(present & under),
        node_id_changed=nodes.node_id[row] != pin.node_id,
    )
    if result.leaf_drift > max_leaf_drift:
        raise CutError(
            f"{pin.cycle}/{pin.subtype}: the recovered cut has moved. {result.describe()}. "
            f"The drift limit is {100 * max_leaf_drift:.0f}%. A cut that has genuinely moved is a "
            f"finding for the next VCM, not something to absorb"
        )
    if result.below_shortfall > max_below_outside:
        raise CutError(
            f"{pin.cycle}/{pin.subtype}: {result.below_outside} of {result.below_present} pinned "
            f"below-list leaves still in this tree are no longer below the recovered cut "
            f"({100 * result.below_shortfall:.1f}%, limit {100 * max_below_outside:.0f}%), so the "
            f"anchors' MRCA is not the clade that was cut. {result.describe()}"
        )
    return result


def _mrca(nodes: _Nodes, rows: Sequence[int]) -> int:
    common: set[int] | None = None
    for row in rows:
        chain = set()
        at = row
        while at != -1:
            chain.add(at)
            at = nodes.parent[at]
        common = chain if common is None else common & chain
    if not common:
        raise CutError("the anchors have no common ancestor; is this one tree?")
    return max(common, key=lambda row: nodes.depth[row])


def _child_towards(nodes: _Nodes, ancestor: int, row: int) -> int:
    """Which child of ``ancestor`` ``row`` descends from."""
    at = row
    while nodes.parent[at] != ancestor:
        at = nodes.parent[at]
        if at == -1:
            raise CutError(f"node {row} does not descend from {nodes.node_id[ancestor]}")
    return at


# -------------------------------------------------------------------------------------------------
# Has circulation above the cut resumed?


@dataclass(frozen=True)
class Resumption:
    """Leaves above the cut in a rebuilt tree, split by how they got into the selection."""

    leaves_total: int
    above_cut: int
    resumed: int  # above the cut and not in the below-list: kept because the cut never saw them
    moved_above: int  # in the below-list, yet above the cut now: the topology moved, not the data
    newest_resumed: datetime.date | None
    limit: float

    @property
    def fraction(self) -> float:
        return self.resumed / self.leaves_total if self.leaves_total else 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "leaves_total": self.leaves_total,
            "above_cut": self.above_cut,
            "resumed": self.resumed,
            "resumed_fraction": round(self.fraction, 6),
            "moved_above": self.moved_above,
            "newest_resumed": None if self.newest_resumed is None else str(self.newest_resumed),
            "limit": self.limit,
        }

    def describe(self) -> str:
        newest = self.newest_resumed.isoformat() if self.newest_resumed else "none dated"
        return (
            f"{self.above_cut} of {self.leaves_total} leaves are above the cut; {self.resumed} of "
            f"them ({100 * self.fraction:.2f}% of the tree) were kept because the cut never saw "
            f"them, newest collected {newest}; {self.moved_above} pinned below-list leaves now sit "
            f"above the cut"
        )


def check_resumption(
    directory: Path,
    pin: CutPin,
    node_id: str,
    *,
    limit: float = MAX_RESUMED,
    outgroup: str | None = None,
) -> Resumption:
    """How much of the retired part of the tree has grown back, counted after the build.

    Selection keeps a key it has never seen (``CutPin.keeps``), which is fail-safe — af never drops
    a virus it has not already placed — but open-ended: a backfill pull of old sequences is also
    something the cut never saw. This is where that is answerable exactly, because the tree now says
    where those leaves attached. Above ``limit`` it refuses, so a cycle cannot quietly regrow the
    part that was retired; the count is also the evidence for the next VCM's choice.

    The outgroup is exempt (it is above every cut by design, or CMAPLE would have nothing to root
    on): it is read from ``tree.json`` unless given.
    """
    nodes = _Nodes.read(directory)
    row = nodes.row_of(node_id)
    if outgroup is None:
        outgroup = i6.read_metadata(Path(directory)).get("outgroup")
    exempt = {_parse_key("outgroup", outgroup)} if outgroup else set()
    block = range(row, row + nodes.extent[row])
    above = [
        leaf for leaf in nodes.leaf_rows() if leaf not in block and nodes.key_of(leaf) not in exempt
    ]
    resumed = [leaf for leaf in above if nodes.key_of(leaf) not in pin.below]
    days = [nodes.last[leaf] for leaf in resumed if nodes.last[leaf] is not None]
    result = Resumption(
        leaves_total=nodes.n_leaves[0],
        above_cut=len(above),
        resumed=len(resumed),
        moved_above=len(above) - len(resumed),
        newest_resumed=max((day for day in days if day is not None), default=None),
        limit=limit,
    )
    if result.fraction > limit:
        raise CutError(
            f"{pin.cycle}/{pin.subtype}: circulation above the cut has resumed, or a backfill has "
            f"re-supplied the retired lineages. {result.describe()}. The limit is "
            f"{100 * limit:.0f}% of the tree. A backfill is a moment to re-choose the cut, not "
            f"something to regrow quietly"
        )
    return result


# -------------------------------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="python -m af.tree.cut", description=__doc__)
    sub = parser.add_subparsers(dest="what", required=True)

    table = sub.add_parser("propose", help="the candidate cut node at each tolerance")
    table.add_argument("version", type=Path, help="a tree-store version directory (I6)")
    table.add_argument("--window-months", type=int, default=WINDOW_MONTHS)
    table.add_argument("--json", action="store_true")

    pin = sub.add_parser("pin", help="the anchors and below-list for a chosen node")
    pin.add_argument("version", type=Path)
    pin.add_argument("node_id", help="the node Sarah chose, from `propose`")
    pin.add_argument("--anchors", type=int, default=ANCHORS)
    pin.add_argument("--below-out", type=Path, help="write the below-list here (one key per line)")

    check = sub.add_parser("recover", help="find a pinned cut in a tree and check it has not moved")
    check.add_argument("version", type=Path)
    check.add_argument("cuts", type=Path, help="the cuts file (acmacs-f-data trees/cuts.toml)")
    check.add_argument("cycle")
    check.add_argument("subtype")
    check.add_argument("--max-drift", type=float, default=MAX_LEAF_DRIFT)
    check.add_argument("--max-below-outside", type=float, default=MAX_BELOW_OUTSIDE)
    check.add_argument("--max-resumed", type=float, default=MAX_RESUMED)
    check.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.what == "propose":
        found = propose(args.version, window_months=args.window_months)
        if args.json:
            print(json.dumps([item.to_json() for item in found], indent=1))
        else:
            for item in found:
                print(item.describe())
        return 0
    if args.what == "pin":
        anchors = pick_anchors(args.version, args.node_id, count=args.anchors)
        below = below_keys(args.version, args.node_id)
        nodes = _Nodes.read(args.version)
        row = nodes.row_of(args.node_id)
        print("[[cut]]")
        print(f'node_id = "{args.node_id}"')
        label = nodes.clade[row]
        # TOML has no null: a node the clade engine did not label gets a comment, not a value.
        print(f'clade = "{label}"' if label else "# clade = (no clade on this node)")
        print(f"expected_leaves = {len(below)}")
        print(f"expected_children = {len(nodes.children[row])}")
        # The measured default is 15 of 20; with another count, the same three quarters.
        print(f"quorum = {max(2, round(QUORUM / ANCHORS * len(anchors)))}")
        print("anchors = [")
        for epi, accession in anchors:
            print(f'    "{epi}|{accession}",')
        print("]")
        print("# subtype, cycle, chosen_on, chosen_from, below_file and reason are yours to add")
        if args.below_out:
            args.below_out.write_text(
                "leaf_id\n" + "".join(f"{epi}|{acc}\n" for epi, acc in sorted(below))
            )
            print(f"# below-list: {len(below)} keys -> {args.below_out}")
        return 0
    loaded = load_cut_pin(args.cuts, args.cycle, args.subtype)
    recovered = recover(
        args.version,
        loaded,
        max_leaf_drift=args.max_drift,
        max_below_outside=args.max_below_outside,
    )
    resumption = check_resumption(args.version, loaded, recovered.node_id, limit=args.max_resumed)
    if args.json:
        both = {"recovered": recovered.to_json(), "resumption": resumption.to_json()}
        print(json.dumps(both, indent=1))
    else:
        print(recovered.describe())
        print(resumption.describe())
    return 0


__all__ = [
    "ANCHORS",
    "MAX_LEAF_DRIFT",
    "MAX_RESUMED",
    "QUORUM",
    "TOLERANCES",
    "WINDOW_MONTHS",
    "CutError",
    "CutPin",
    "CutProposal",
    "RecoveredCut",
    "Resumption",
    "below_keys",
    "check_resumption",
    "load_cut_pin",
    "pick_anchors",
    "propose",
    "read_below",
    "recover",
]
