"""Is a clades table behind its sequences in content, not just in version label?

A clades table's provenance names the sequences version it was labelled from. Comparing that
label with CURRENT says "behind" after every sequences republish, even one that changes nothing
the clade calls read (a name fixed, a flag added, a protein padded), and an identical refresh
can never clear it: the store returns the existing clades version, which still cites the old
label. So consumers that ask "do sequences have no clade row because the table is stale?"
need the content answer.

:func:`calls_fingerprint` hashes exactly what the fallback reads from a sequences version
(:data:`af.clades.fallback.STORE_COLUMNS`, every row, sorted) and the Nextclade datasets the
version cites. Two versions with the same fingerprint produce the same clade rows, so
:func:`clades_behind_in_content` compares the version a table cites with the one a consumer
read, and is behind only when a refresh would publish something.

Tree-based tables also depend on their tree. Tree versions are content-addressed (an identical
rebuild returns the same version), so for the tree a version comparison is already a content
comparison; it is the sequences side that needed this.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from af.clades.fallback import STORE_COLUMNS
from af.clades.store import CladeStoreError
from af.seq.processed import read_table
from af.store import Store, StoreRef


@dataclass(frozen=True)
class ContentBehind:
    """Whether a clades table's calls cover the sequences version a consumer read."""

    behind: bool
    labelled: StoreRef
    """The sequences version the table was labelled from (its provenance)."""
    read: StoreRef
    """The sequences version it was compared with."""

    @property
    def same_version(self) -> bool:
        return self.labelled.version == self.read.version


def calls_fingerprint(store: Store, sequences: StoreRef) -> str:
    """A hash of everything the fallback's clade calls depend on in ``sequences``.

    Every row's (epi_isl, accession, nextclade_subclade, nextclade_qc_status), sorted, and
    the ``raw/nextclade/...`` datasets in the version's provenance. Names, flags, proteins
    and pull layout are not included: they cannot change a clade row.
    """
    table = read_table(store, sequences, "sequences", STORE_COLUMNS)
    rows = sorted(
        tuple("" if value is None else f"={value}" for value in row)
        for row in zip(*(table[column].to_pylist() for column in STORE_COLUMNS), strict=True)
    )
    provenance = json.loads((store.resolve(sequences) / "PROVENANCE.json").read_text())
    datasets = sorted(
        f"{entry['dataset']}@{entry['version']}"
        for item in provenance.get("inputs", [])
        if (entry := item.get("store"))
        and entry.get("kind") == "raw"
        and entry.get("dataset", "").startswith("nextclade/")
    )
    digest = hashlib.sha256()
    digest.update(json.dumps(datasets).encode())
    for row in rows:
        digest.update(("\t".join(row) + "\n").encode())
    return digest.hexdigest()


def clades_behind_in_content(store: Store, clades: StoreRef, sequences: StoreRef) -> ContentBehind:
    """Is ``clades`` behind ``sequences`` in content? Same version, or same fingerprint: no.

    A table whose provenance names no sequences version of its own dataset cannot be checked,
    and that is an error rather than a guess either way.
    """
    provenance = json.loads((store.resolve(clades) / "PROVENANCE.json").read_text())
    cited = [
        StoreRef.from_json(item["store"])
        for item in provenance.get("inputs", [])
        if "store" in item
        and item["store"].get("kind") == "sequences"
        and item["store"].get("dataset") == sequences.dataset
    ]
    if len(cited) != 1:
        raise CladeStoreError(
            f"{clades}: its provenance names {len(cited)} sequences/{sequences.dataset} "
            "versions; exactly one is needed to tell whether it is behind"
        )
    labelled = cited[0]
    if labelled.version == sequences.version:
        return ContentBehind(False, labelled, sequences)
    behind = calls_fingerprint(store, labelled) != calls_fingerprint(store, sequences)
    return ContentBehind(behind, labelled, sequences)
