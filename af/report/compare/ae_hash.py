"""ae's sequence hash, so af's tree leaves can be matched to an ae reference tree by sequence.

ae writes a tree leaf as ``<name>_<passage>_<hash>``, the hash being xxHash32 (seed 0) of the
aligned nucleotide sequence as 8 upper-case hex digits (ae ``cc/utils/hash.hh``,
``sequences/align.cc`` ``calculate_hash``). Computed over af's ``nuc_aligned`` it reproduces
ae's: measured on 2,000 random af H1 sequences against the Sep 2026 report tree, 1,450 hit, 526
were strains not in that tree, and 24 (1.2%) were in it under another hash (another passage or
accession of the strain; trimming gaps rescued none). Matching on it pairs the same sequence
whatever either side calls the strain, which a name join cannot: af keeps a leaf per isolate,
ae one per identical sequence, named by one of them.

ae compatibility only: nothing in af's own figures depends on it. Pure Python, measured at
1.4 s per 10,000 sequences of ~1,650 nt.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from af.store import Store, StoreRef

_P1, _P2, _P3, _P4, _P5 = 2654435761, 2246822519, 3266489917, 668265263, 374761393
_MASK = 0xFFFFFFFF

REFERENCE_HASH = re.compile(r"_([0-9A-F]{8})$")
"""The hash at the end of an ae leaf id (``<name>_<passage>_1A2B3C4D``)."""


def _rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & _MASK


def xxh32(data: bytes, seed: int = 0) -> int:
    """xxHash32 of ``data`` (the reference algorithm, little-endian lanes)."""
    n, i = len(data), 0
    if n >= 16:
        v = [(seed + _P1 + _P2) & _MASK, (seed + _P2) & _MASK, seed & _MASK, (seed - _P1) & _MASK]
        while i <= n - 16:
            for k in range(4):
                lane = int.from_bytes(data[i : i + 4], "little")
                v[k] = (_rotl((v[k] + lane * _P2) & _MASK, 13) * _P1) & _MASK
                i += 4
        h = (_rotl(v[0], 1) + _rotl(v[1], 7) + _rotl(v[2], 12) + _rotl(v[3], 18)) & _MASK
    else:
        h = (seed + _P5) & _MASK
    h = (h + n) & _MASK
    while i + 4 <= n:
        lane = int.from_bytes(data[i : i + 4], "little")
        h = (_rotl((h + lane * _P3) & _MASK, 17) * _P4) & _MASK
        i += 4
    while i < n:
        h = (_rotl((h + data[i] * _P5) & _MASK, 11) * _P1) & _MASK
        i += 1
    h ^= h >> 15
    h = (h * _P2) & _MASK
    h ^= h >> 13
    h = (h * _P3) & _MASK
    h ^= h >> 16
    return h


def ae_hash(nuc_aligned: str) -> str:
    """ae's hash of an aligned nucleotide sequence, as ae prints it."""
    return f"{xxh32(nuc_aligned.encode('ascii')):08X}"


def reference_hash(leaf_id: str) -> str | None:
    """The hash an ae leaf id ends in, or None."""
    m = REFERENCE_HASH.search(leaf_id)
    return m.group(1) if m else None


def leaf_key(leaf_id: str) -> tuple[str, str | None] | None:
    """(epi_isl, accession) of an af leaf id ``EPI_ISL_<n>|<accession>``, or None."""
    epi, sep, accession = leaf_id.partition("|")
    if not epi.startswith("EPI_ISL_"):
        return None
    return epi, (accession or None) if sep else None


def sequences_under(store: Store, ref: StoreRef) -> list[StoreRef]:
    """The sequences versions a store version was built from, followed upstream."""
    from af.report.provenance import upstream_refs

    found, queue, seen = [], [ref], set()
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        if current.kind == "sequences":
            found.append(current)
            continue
        queue.extend(upstream_refs(store, current))
    return sorted(set(found), key=str)


def leaf_hashes(
    store: Store, sequences: StoreRef, leaf_ids: Iterable[str]
) -> tuple[dict[str, str], int]:
    """ae hash per af leaf id from ``sequences``; also how many ids had no aligned sequence."""
    from af.seq.processed import read_table

    wanted = {key: leaf for leaf in leaf_ids if (key := leaf_key(leaf))}
    table = read_table(store, sequences, "sequences", ["epi_isl", "accession", "nuc_aligned"])
    out: dict[str, str] = {}
    for epi, accession, nuc in zip(
        *(table[c].to_pylist() for c in ("epi_isl", "accession", "nuc_aligned")), strict=True
    ):
        leaf = wanted.get((epi, accession))
        if leaf is not None and nuc:
            out[leaf] = ae_hash(nuc)
    return out, len(wanted) - len(out)


def figure_hashes(
    store: Store | None, doc: dict[str, Any]
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """ae hash per drawn leaf of an af tree figure, or None when its leaves are not EPI keyed.

    A tree whose leaf ids are ``EPI_ISL_<n>|<accession>`` is matched by sequence, so the store is
    then required (missing inputs are fatal): the sequences are the single sequences version the
    figure's tree version was built from. Returns the hashes and what they came from.
    """
    ids = [str(leaf["id"]) for leaf in doc["tree"]["leaves"] if leaf["shown"]]
    if not any(leaf_key(i) for i in ids):
        return None, {}
    if store is None:
        raise ValueError(f"{doc.get('title')!r}: leaves are EPI_ISL keyed; --store is needed to "
                         "match them to the reference by sequence")  # fmt: skip
    trees = [StoreRef.from_json(r) for r in doc["provenance"].get("store_refs", [])
             if r.get("kind") == "trees"]  # fmt: skip
    if len(trees) != 1:
        raise ValueError(f"{doc.get('title')!r}: {len(trees)} tree store refs; exactly one needed")
    sequences = sequences_under(store, trees[0])
    if len(sequences) != 1:
        raise ValueError(f"{trees[0]} rests on {len(sequences)} sequences versions; exactly one "
                         "is needed to hash its leaves")  # fmt: skip
    hashes, without = leaf_hashes(store, sequences[0], ids)
    return hashes, {"sequences_version": str(sequences[0]), "leaves_without_sequence": without}
