"""Store layout: raw, sequences, clades, tables, trees, chains, serology; snapshots, manifests.

Generated data lives in one store root outside git (path from config). Each kind holds
datasets; each dataset holds immutable, content-addressed versions, a CURRENT pointer,
an append-only HISTORY and a write-once cache. See :mod:`af.store.store`.
"""

from af.store.manifest import Manifest
from af.store.ref import KINDS, ExternalInput, StoreError, StoreRef
from af.store.snapshot import (
    check_refs,
    read_manifest,
    read_snapshot,
    write_manifest,
    write_snapshot,
)
from af.store.store import Provenance, Store, VersionBuilder
from af.store.work import DatasetWork, PathsConfig, Work

__all__ = [
    "DatasetWork",
    "PathsConfig",
    "Work",
    "KINDS",
    "ExternalInput",
    "Manifest",
    "Provenance",
    "Store",
    "StoreError",
    "StoreRef",
    "VersionBuilder",
    "check_refs",
    "read_manifest",
    "read_snapshot",
    "write_manifest",
    "write_snapshot",
]
