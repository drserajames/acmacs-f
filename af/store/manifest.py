"""A version's MANIFEST.json: every file in it, with size and SHA-256.

The version id is the first 16 hex digits of the SHA-256 of the manifest's bytes, so
identical content always gives an identical id. The manifest lists the dataset's
own files only. ``MANIFEST.json`` and ``PROVENANCE.json`` are excluded, so a rebuild
from new inputs that produces the same files is recognised as the same version.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from af.store.ref import StoreError
from af.util.artefacts import sha256_path

MANIFEST = "MANIFEST.json"
PROVENANCE = "PROVENANCE.json"
RESERVED_FILES = (MANIFEST, PROVENANCE)


@dataclass(frozen=True)
class FileEntry:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Manifest:
    files: tuple[FileEntry, ...]

    def to_bytes(self) -> bytes:
        """Canonical bytes: files sorted by path, keys sorted, fixed separators."""
        entries = [
            {"path": entry.path, "sha256": entry.sha256, "size": entry.size}
            for entry in sorted(self.files, key=lambda entry: entry.path)
        ]
        return (json.dumps({"files": entries}, indent=1, sort_keys=True) + "\n").encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def version(self) -> str:
        return self.sha256()[:16]

    def by_path(self) -> dict[str, FileEntry]:
        return {entry.path: entry for entry in self.files}

    @classmethod
    def from_bytes(cls, data: bytes) -> Manifest:
        parsed = json.loads(data)
        return cls(tuple(FileEntry(**entry) for entry in parsed["files"]))


def dataset_files(directory: Path) -> list[str]:
    """Relative POSIX paths of the dataset's files (reserved top-level names excluded)."""
    files = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise StoreError(f"symlinks are not allowed in a store version: {path}")
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            if relative not in RESERVED_FILES:
                files.append(relative)
    return sorted(files)


def build_manifest(directory: Path) -> Manifest:
    entries = []
    for relative in dataset_files(directory):
        path = directory / relative
        entries.append(FileEntry(relative, path.stat().st_size, sha256_path(path)))
    return Manifest(tuple(entries))


def verify_files(directory: Path, manifest: Manifest, *, deep: bool) -> list[str]:
    """Problems with ``directory`` against ``manifest``: missing, extra, size, hash."""
    expected = manifest.by_path()
    present = set(dataset_files(directory))
    problems = [f"missing: {path}" for path in sorted(set(expected) - present)]
    problems += [f"not in manifest: {path}" for path in sorted(present - set(expected))]
    for path in sorted(present & set(expected)):
        entry = expected[path]
        actual = directory / path
        if actual.stat().st_size != entry.size:
            problems.append(f"size changed: {path}")
        elif deep and sha256_path(actual) != entry.sha256:
            problems.append(f"content changed: {path}")
    return problems
