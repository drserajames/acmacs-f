"""References into a store, and the inputs a version records in its provenance.

A :class:`StoreRef` names one immutable version of one dataset, and carries the full
SHA-256 of that version's ``MANIFEST.json``. A reader on another machine can then check
that its copy is the one that was used. Report manifests, snapshots and provenance
all point into the store with these.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from af.util.artefacts import sha256_path

KINDS = ("raw", "sequences", "clades", "tables", "trees", "chains", "serology")
"""The store's top-level directories: one per af stage that writes into the store.

``raw`` holds inputs af cannot regenerate (e.g. GISAID pulls): never deleted, always
backed up. Everything else can be rebuilt from ``raw`` and the lab repositories.
"""

_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._=+-]*")
_RESERVED = {"versions", "CURRENT", "HISTORY.jsonl"}
_VERSION = re.compile(r"[0-9a-f]{16}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class StoreError(RuntimeError):
    """A store, dataset or version is missing, malformed or does not match its reference."""


def check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise StoreError(f"unknown store kind {kind!r}; expected one of {', '.join(KINDS)}")
    return kind


def check_dataset(dataset: str) -> str:
    """A dataset key is one or more '/'-separated readable segments, e.g. ``labx/h3-hi``."""
    segments = dataset.split("/")
    for segment in segments:
        if not _SEGMENT.fullmatch(segment) or segment in _RESERVED:
            raise StoreError(
                f"bad dataset key {dataset!r}: segment {segment!r} must start with a letter or "
                f"digit, use only letters, digits and ._=+- and not be one of {sorted(_RESERVED)}"
            )
    return dataset


@dataclass(frozen=True)
class StoreRef:
    kind: str
    dataset: str
    version: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        check_kind(self.kind)
        check_dataset(self.dataset)
        if not _VERSION.fullmatch(self.version):
            raise StoreError(f"bad version id {self.version!r}: expected 16 hex digits")
        if not _SHA256.fullmatch(self.manifest_sha256):
            raise StoreError(f"bad manifest_sha256 {self.manifest_sha256!r}")
        if not self.manifest_sha256.startswith(self.version):
            raise StoreError(f"version {self.version} is not the prefix of its manifest hash")

    def to_json(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "dataset": self.dataset,
            "version": self.version,
            "manifest_sha256": self.manifest_sha256,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> StoreRef:
        expected = {"kind", "dataset", "version", "manifest_sha256"}
        if set(data) != expected:
            raise StoreError(f"store ref must have exactly the keys {sorted(expected)}: {data}")
        return cls(**data)

    def __str__(self) -> str:
        return f"{self.kind}/{self.dataset}@{self.version}"


@dataclass(frozen=True)
class ExternalInput:
    """An input from outside the store (a pinned reference dataset, a lab file in git)."""

    path: Path
    sha256: str
    version: str | None = None

    @classmethod
    def of(cls, path: Path, version: str | None = None) -> ExternalInput:
        """Hash ``path`` now; a missing path is fatal."""
        if not Path(path).exists():
            raise FileNotFoundError(f"external input missing: {path}")
        return cls(Path(path), sha256_path(Path(path)), version)

    def to_json(self) -> dict[str, Any]:
        entry: dict[str, Any] = {"path": str(self.path), "sha256": self.sha256}
        if self.version is not None:
            entry["version"] = self.version
        return {"external": entry}


Input = StoreRef | ExternalInput


def input_to_json(item: Input) -> dict[str, Any]:
    """Provenance input entries are exactly one of ``{"store": ref}`` or ``{"external": ...}``."""
    if isinstance(item, StoreRef):
        return {"store": item.to_json()}
    return item.to_json()


def input_from_json(data: dict[str, Any]) -> Input:
    if set(data) == {"store"}:
        return StoreRef.from_json(data["store"])
    if set(data) == {"external"}:
        entry = data["external"]
        return ExternalInput(Path(entry["path"]), entry["sha256"], entry.get("version"))
    raise StoreError(f"provenance input must be {{'store': …}} or {{'external': …}}: {data}")
