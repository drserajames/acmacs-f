"""Clade brackets and lettered bands, derived from the clade assignments and the layout.

This replaces the hand-curated part of the old `.tal` files: which clades get a bracket, the
bracket slots, the per-clade inclusion/exclusion tolerances and the lettered hz bands. Every
threshold here is relative (to the drawn rows or to the clade's own size), so one set of
defaults works as trees grow. The only hand input is named overrides, counted and reported.

Measured against a real round's hand curation in notes/tree-figure/COMPARISON.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


class SectionOverrideError(ValueError):
    """An override names a clade or band that does not exist (design rule 1)."""


@dataclass
class Band:
    clade: str
    first: int  # first row, inclusive
    last: int  # last row, inclusive
    members: int  # rows in [first, last] that carry the clade

    @property
    def span(self) -> int:
        return self.last - self.first + 1


@dataclass
class CladeSections:
    clade: str
    bands: list[Band]
    dropped: list[Band] = field(default_factory=list)  # strays, not drawn, counted
    total: int = 0  # rows carrying the clade


def clade_membership(
    leaf_clades: list[str | None], parents: Mapping[str, str | None]
) -> dict[str, np.ndarray]:
    """Per clade, a bool per row: the row's clade is this clade or a descendant of it.

    The tree store gives each leaf one clade label; ancestry comes from the nomenclature
    hierarchy (``parents``: clade -> parent clade). A label missing from the hierarchy is an
    error, because it would silently drop the leaf from every ancestor's bracket.
    """
    n = len(leaf_clades)
    unknown = sorted({c for c in leaf_clades if c is not None and c not in parents})
    if unknown:
        raise SectionOverrideError(f"clade label(s) not in the nomenclature: {unknown[:5]}")
    lineage: dict[str, list[str]] = {}
    for clade in parents:
        chain: list[str] = []
        c: str | None = clade
        while c is not None:
            chain.append(c)
            c = parents.get(c)
        lineage[clade] = chain
    member: dict[str, np.ndarray] = {}
    for row, label in enumerate(leaf_clades):
        if label is None:
            continue
        for ancestor in lineage[label]:
            member.setdefault(ancestor, np.zeros(n, bool))[row] = True
    return member


def runs(member: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive True rows, as (first, last)."""
    padded = np.concatenate([[False], member, [False]]).astype(np.int8)
    d = np.diff(padded)
    return list(
        zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist(), strict=True)
    )


@dataclass
class BandParams:
    """Values and their reasons: defaults.toml [bands]."""

    merge_gap_fraction: float
    merge_min_purity: float
    min_gap: int
    drop_fraction: float


def clade_bands(member: np.ndarray, clade: str, p: BandParams) -> CladeSections:
    """Bands from contiguous runs, merged across gaps that are small relative to the clade.

    A merge is refused if the merged span would fall below ``merge_min_purity`` members, so a
    band never swallows a sibling clade. The old per-clade tolerances were absolute tip counts
    that had to be retuned as the tree grew; these are relative.
    """
    total = int(member.sum())
    if total == 0:
        return CladeSections(clade, [], [], 0)
    cum = np.concatenate([[0], np.cumsum(member)])

    def count(f: int, last: int) -> int:
        return int(cum[last + 1] - cum[f])

    max_gap = max(p.min_gap, int(p.merge_gap_fraction * total))
    merged: list[list[int]] = []
    for f, last in runs(member):
        if merged:
            pf, pl = merged[-1]
            gap = f - pl - 1
            pure = count(pf, last) >= p.merge_min_purity * (last - pf + 1)
            if gap <= p.min_gap or (gap <= max_gap and pure):
                merged[-1][1] = last
                continue
        merged.append([f, last])
    bands = [Band(clade, f, last, count(f, last)) for f, last in merged]
    largest = max(b.members for b in bands)
    kept = [b for b in bands if b.members >= p.drop_fraction * largest]
    dropped = [b for b in bands if b.members < p.drop_fraction * largest]
    return CladeSections(clade, kept, dropped, total)


@dataclass
class SelectParams:
    """Values and their reasons: defaults.toml [clades]."""

    min_share: float  # shown if on at least this share of the drawn rows ...
    min_window_leaves: int  # ... or with at least this many leaves in the time window
    max_share: float
    min_coherence: float
    cut_window_share: float


@dataclass
class Selection:
    shown: list[CladeSections]
    rejected: dict[str, str]  # clade -> reason; reported, never silent
    slots: dict[str, int]  # 0 = next to the matrix
    overrides: dict[str, int]  # override kind -> times applied


def select_clades(
    member: Mapping[str, np.ndarray],
    parents: Mapping[str, str | None],
    in_window: np.ndarray,
    p: SelectParams,
    bands: BandParams,
    force_show: frozenset[str] = frozenset(),
    force_hide: frozenset[str] = frozenset(),
) -> Selection:
    """Choose the clades that get a bracket; slots follow nesting depth among those chosen.

    A clade is "very small" (not shown) when it is on less than ``min_share`` of the rows AND
    has fewer than ``min_window_leaves`` leaves in the window: small clades that are still
    circulating keep their bracket.
    """
    n_rows = len(in_window)
    missing = sorted((force_show | force_hide) - set(member))
    if missing:
        raise SectionOverrideError(f"clade override(s) match no drawn clade: {missing}")
    shown, rejected = [], {}
    for clade in sorted(member):
        m = member[clade]
        share = m.sum() / n_rows
        in_win = int((m & in_window).sum())
        cs = clade_bands(m, clade, bands)
        coherence = max(b.members for b in cs.bands) / cs.total if cs.bands else 0.0
        if clade in force_hide:
            rejected[clade] = "override: hide"
        elif clade in force_show:
            shown.append(cs)
        elif share > p.max_share:
            rejected[clade] = f"on {share:.3f} of rows (> {p.max_share})"
        elif share < p.min_share and in_win < p.min_window_leaves:
            rejected[clade] = (
                f"very small: on {share:.4f} of rows (< {p.min_share}) and "
                f"{in_win} leaves in the window (< {p.min_window_leaves})"
            )
        elif coherence < p.min_coherence:
            rejected[clade] = f"scattered: largest band holds {coherence:.2f} of its rows"
        else:
            shown.append(cs)
    names = {cs.clade for cs in shown}

    def depth(c: str) -> int:
        d, a = 0, parents.get(c)
        while a is not None:
            d += a in names
            a = parents.get(a)
        return d

    max_depth = max((depth(c) for c in names), default=0)
    slots = {c: max_depth - depth(c) for c in names}
    overrides = {"force_show": len(force_show), "force_hide": len(force_hide)}
    return Selection(shown, rejected, slots, overrides)


@dataclass
class HzBand:
    letter: str
    clade: str
    first: int
    last: int


def hz_partition(
    sel: Selection,
    parents: Mapping[str, str | None],
    in_window: np.ndarray,
    p: SelectParams,
    min_rows_fraction: float = 0.002,
    hide_first_rows: frozenset[int] = frozenset(),
) -> list[HzBand]:
    """Lettered horizontal bands: each row belongs to the deepest drawn clade whose band cuts.

    A nested band cuts its ancestor's band when most of its rows are in the time window;
    otherwise it stays inside the ancestor's letter (bracketed, not lettered). Residual pieces
    smaller than ``min_rows_fraction`` of the rows are not lettered. ``hide_first_rows`` is the
    override (the caller resolves leaf names to rows and errors on a name that matches nothing).
    """
    n = len(in_window)
    owner = np.full(n, -1)
    clades: list[str] = []
    top = max(sel.slots.values(), default=0)
    for cs in sorted(sel.shown, key=lambda cs: -sel.slots[cs.clade]):  # ancestors first
        for b in cs.bands:
            cuts = in_window[b.first : b.last + 1].mean() >= p.cut_window_share
            nested = (owner[b.first : b.last + 1] >= 0).any()
            if cuts or sel.slots[cs.clade] == top or not nested:
                clades.append(cs.clade)
                owner[b.first : b.last + 1] = len(clades) - 1
    pieces: list[HzBand] = []
    for f, last in runs(owner >= 0):
        seg = owner[f : last + 1]
        change = np.flatnonzero(np.diff(seg)) + 1
        starts = np.concatenate([[0], change]).tolist()
        ends = np.concatenate([change, [len(seg)]]).tolist()
        for s, e in zip(starts, ends, strict=True):
            pieces.append(HzBand("", clades[seg[s]], f + s, f + e - 1))
    unmatched = hide_first_rows - {b.first for b in pieces}
    if unmatched:
        raise SectionOverrideError(
            f"hz hide override(s) match no band start: rows {sorted(unmatched)}"
        )
    out = [
        b
        for b in pieces
        if b.last - b.first + 1 >= min_rows_fraction * n and b.first not in hide_first_rows
    ]
    for k, hz in enumerate(out):
        hz.letter = _letter(k)
    return out


def _letter(k: int) -> str:
    """A, B, ... Z, AA, AB, ...: a figure never runs out of letters."""
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        s = chr(ord("A") + r) + s
    return s
