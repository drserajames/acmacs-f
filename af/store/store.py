"""The store: datasets of immutable, content-addressed versions under one root.

Layout (the full description is in STORE-LAYOUT.md in the notes; this is the summary)::

    <root>/STORE.toml                       layout version; af refuses a store it doesn't know
    <root>/<kind>/<dataset key...>/
        CURRENT                             the version id in use (plain file, not a symlink)
        HISTORY.jsonl                       append-only log of publishes and reconfirmations
        versions/<version id>/              immutable: MANIFEST.json, PROVENANCE.json, files
        cache/<entry key>/                  write-once entries, looked up by input key
        (versions/.tmp-*, cache/.tmp-*)     work in progress; never synced, never read

Publishing is atomic. A version is built in ``versions/.tmp-*``, hashed into its
manifest and renamed into place, and only then do ``HISTORY.jsonl`` and ``CURRENT``
change. Published files are made read-only. That matters because unchanged files are
hard-linked from earlier versions, and writing into a linked file would silently change
the old version too.

A rebuild that produces files identical to an existing version publishes nothing new:
the manifest (and so the id) is the same. ``HISTORY.jsonl`` records a
``reconfirmed`` event with the new inputs.
"""

from __future__ import annotations

import datetime
import fcntl
import json
import os
import shutil
import socket
import stat
import tomllib
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import af
from af.store.manifest import (
    MANIFEST,
    PROVENANCE,
    Manifest,
    build_manifest,
    verify_files,
)
from af.store.ref import (
    Input,
    StoreError,
    StoreRef,
    check_dataset,
    check_kind,
    input_to_json,
)

LAYOUT_VERSION = 1
STORE_FILE = "STORE.toml"
CURRENT = "CURRENT"
HISTORY = "HISTORY.jsonl"
VERSIONS = "versions"
CACHE = "cache"
TMP_PREFIX = ".tmp-"


@dataclass(frozen=True)
class Provenance:
    """What a version was built from. Written to the version's PROVENANCE.json."""

    step: str
    inputs: tuple[Input, ...]
    parameters: Mapping[str, Any]
    started: datetime.datetime
    finished: datetime.datetime

    def to_json(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "af_version": af.__version__,
            "inputs": [input_to_json(item) for item in self.inputs],
            "parameters": json.loads(json.dumps(dict(self.parameters), sort_keys=True)),
            "started": _utc(self.started),
            "finished": _utc(self.finished),
        }


class Store:
    """An opened store. Open an existing one with :meth:`open`, make one with :meth:`create`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @classmethod
    def create(cls, root: Path) -> Store:
        """Make a new, empty store. Refuses a directory that already has content."""
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise StoreError(f"cannot create a store in non-empty directory {root}")
        root.mkdir(parents=True, exist_ok=True)
        (root / STORE_FILE).write_text(
            f'layout = {LAYOUT_VERSION}\ncreated_by = "af {af.__version__}"\n'
        )
        return cls(root)

    @classmethod
    def open(cls, root: Path) -> Store:
        """Open an existing store. A missing or unknown STORE.toml is fatal."""
        marker = Path(root) / STORE_FILE
        if not marker.is_file():
            raise StoreError(f"not a store (no {STORE_FILE}): {root}")
        layout = tomllib.loads(marker.read_text()).get("layout")
        if layout != LAYOUT_VERSION:
            raise StoreError(
                f"{marker}: layout {layout!r} is not supported by af {af.__version__} "
                f"(expects {LAYOUT_VERSION})"
            )
        return cls(Path(root))

    # ---- paths ------------------------------------------------------------------

    def dataset_dir(self, kind: str, dataset: str) -> Path:
        return self.root / check_kind(kind) / check_dataset(dataset)

    def version_dir(self, ref: StoreRef) -> Path:
        return self.dataset_dir(ref.kind, ref.dataset) / VERSIONS / ref.version

    # ---- reading ----------------------------------------------------------------

    def current(self, kind: str, dataset: str) -> StoreRef:
        """The version a dataset's CURRENT names. A dataset with no CURRENT is fatal."""
        path = self.dataset_dir(kind, dataset) / CURRENT
        if not path.is_file():
            raise StoreError(f"no such dataset (no {CURRENT}): {kind}/{dataset}")
        version = path.read_text().strip()
        manifest_path = self.dataset_dir(kind, dataset) / VERSIONS / version / MANIFEST
        if not manifest_path.is_file():
            raise StoreError(f"{kind}/{dataset}: CURRENT names missing version {version}")
        return StoreRef(
            kind, dataset, version, Manifest.from_bytes(manifest_path.read_bytes()).sha256()
        )

    def list_datasets(self, kind: str) -> list[StoreRef]:
        """Every dataset of ``kind`` with its CURRENT ref, sorted by dataset key."""
        base = self.root / check_kind(kind)
        if not base.is_dir():
            return []
        return [self.current(kind, dataset) for dataset in _dataset_keys(base)]

    def resolve(self, ref: StoreRef, *, verify: bool = False) -> Path:
        """The directory of ``ref``'s version, after checking its manifest hash.

        The manifest check re-hashes only MANIFEST.json, so it is cheap and always done.
        ``verify=True`` also re-hashes every file (see :meth:`verify`), which is slow
        for large versions: use it at report-manifest time, not on every access.
        """
        directory = self.version_dir(ref)
        manifest_path = directory / MANIFEST
        if not manifest_path.is_file():
            raise StoreError(f"{ref}: version not in this store ({directory})")
        actual = Manifest.from_bytes(manifest_path.read_bytes()).sha256()
        if actual != ref.manifest_sha256:
            raise StoreError(f"{ref}: manifest hash {actual} does not match the reference")
        if verify:
            self.verify(ref)
        return directory

    def verify(self, ref: StoreRef) -> None:
        """Re-hash every file of a version against its manifest; raise on any mismatch."""
        directory = self.version_dir(ref)
        manifest = Manifest.from_bytes((directory / MANIFEST).read_bytes())
        problems = verify_files(directory, manifest, deep=True)
        if problems:
            raise StoreError(f"{ref} does not match its manifest:\n  " + "\n  ".join(problems))

    def history(self, kind: str, dataset: str) -> list[dict[str, Any]]:
        path = self.dataset_dir(kind, dataset) / HISTORY
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # ---- writing a version --------------------------------------------------------

    @contextmanager
    def build(self, kind: str, dataset: str) -> Iterator[VersionBuilder]:
        """Build a new version in a temporary directory; discarded unless published."""
        dataset_dir = self.dataset_dir(kind, dataset)
        self._refuse_nesting(kind, dataset)
        staging = _new_tmp_dir(dataset_dir / VERSIONS)
        builder = VersionBuilder(self, kind, dataset, staging)
        try:
            yield builder
        finally:
            if staging.exists():
                _remove_tree(staging)

    def _refuse_nesting(self, kind: str, dataset: str) -> None:
        """A dataset may not sit inside another dataset's directory, nor contain one."""
        parts = dataset.split("/")
        for depth in range(1, len(parts)):
            parent = "/".join(parts[:depth])
            if (self.dataset_dir(kind, parent) / CURRENT).exists():
                raise StoreError(f"{kind}/{dataset} would be nested inside dataset {parent!r}")
        inner = [key for key in _dataset_keys(self.root / kind) if key.startswith(dataset + "/")]
        if inner:
            raise StoreError(f"{kind}/{dataset} would contain dataset {inner[0]!r}")

    # ---- write-once cache ---------------------------------------------------------

    def cache_entry(self, kind: str, dataset: str, key: str) -> Path | None:
        """A published cache entry, or None. Entries are looked up by an input key."""
        path = self.dataset_dir(kind, dataset) / CACHE / check_dataset(key)
        return path if path.is_dir() else None

    @contextmanager
    def build_cache_entry(self, kind: str, dataset: str, key: str) -> Iterator[Path]:
        """Write a cache entry into the yielded directory; it is published on success.

        Write-once: if the entry already exists when the block ends, the new one is
        discarded and the existing one kept (an entry is a pure function of its key).
        An empty entry is an error.
        """
        check_dataset(key)
        self._refuse_nesting(kind, dataset)
        cache = self.dataset_dir(kind, dataset) / CACHE
        staging = _new_tmp_dir(cache)
        try:
            yield staging
            if not any(path.is_file() for path in staging.rglob("*")):
                raise StoreError(f"cache entry {kind}/{dataset}:{key} is empty")
            target = cache / key
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                _make_read_only(staging)
                staging.rename(target)
        finally:
            if staging.exists():
                _remove_tree(staging)


class VersionBuilder:
    """Writes one new version. Use via :meth:`Store.build`, then call :meth:`publish`."""

    def __init__(self, store: Store, kind: str, dataset: str, path: Path) -> None:
        self.store = store
        self.kind = kind
        self.dataset = dataset
        self.path = path
        self.published: StoreRef | None = None

    def link(self, source: Path, relative: str) -> Path:
        """Put ``source`` (a file, or a directory tree) at ``relative`` without copying.

        Hard links when the filesystem allows, else copies. Use it for files unchanged
        since an earlier version, or for cache entries. Linked files are read-only, and
        must be replaced, never edited in place.
        """
        target = self.path / relative
        if source.is_dir():
            for file in sorted(path for path in source.rglob("*") if path.is_file()):
                self.link(file, (Path(relative) / file.relative_to(source)).as_posix())
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        return target

    def link_unchanged(self, relative: str) -> Path:
        """Hard-link ``relative`` from the dataset's CURRENT version."""
        current = self.store.current(self.kind, self.dataset)
        return self.link(self.store.version_dir(current) / relative, relative)

    def publish(self, provenance: Provenance, summary: Mapping[str, Any] | None = None) -> StoreRef:
        """Freeze the staged files as a version, make it CURRENT, and return its ref.

        If identical files are already a version of this dataset, nothing new is
        written; the existing version becomes (or stays) CURRENT and HISTORY records it.
        """
        if self.published is not None:
            raise StoreError("this builder has already published")
        manifest = build_manifest(self.path)
        if not manifest.files:
            raise StoreError(f"{self.kind}/{self.dataset}: refusing to publish an empty version")
        ref = StoreRef(self.kind, self.dataset, manifest.version(), manifest.sha256())
        dataset_dir = self.store.dataset_dir(self.kind, self.dataset)
        with _locked(dataset_dir):
            self._publish_locked(ref, manifest, provenance, summary)
        self.published = ref
        return ref

    def _publish_locked(
        self,
        ref: StoreRef,
        manifest: Manifest,
        provenance: Provenance,
        summary: Mapping[str, Any] | None,
    ) -> None:
        dataset_dir = self.store.dataset_dir(self.kind, self.dataset)
        previous = self._current_version()
        target = dataset_dir / VERSIONS / ref.version
        if target.exists():
            self.store.verify(ref)
            event = "reconfirmed" if previous == ref.version else "restored"
        else:
            (self.path / MANIFEST).write_bytes(manifest.to_bytes())
            (self.path / PROVENANCE).write_text(
                json.dumps(provenance.to_json(), indent=2, sort_keys=True) + "\n"
            )
            _make_read_only(self.path)
            self.path.rename(target)
            event = "published"
        _append_history(
            dataset_dir,
            {
                "event": event,
                "version": ref.version,
                "parent": previous,
                "created": _utc(datetime.datetime.now(datetime.UTC)),
                "af_version": af.__version__,
                "step": provenance.step,
                "inputs": [input_to_json(item) for item in provenance.inputs],
                "summary": dict(summary or {}),
            },
        )
        _write_atomically(dataset_dir / CURRENT, ref.version + "\n")

    def _current_version(self) -> str | None:
        path = self.store.dataset_dir(self.kind, self.dataset) / CURRENT
        return path.read_text().strip() if path.is_file() else None


# ---- helpers ------------------------------------------------------------------------


def _new_tmp_dir(parent: Path) -> Path:
    """A unique work directory. Host and pid in the name say who left it, if one is left."""
    parent.mkdir(parents=True, exist_ok=True)
    name = f"{TMP_PREFIX}{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    path = parent / name
    path.mkdir()
    return path


def _make_read_only(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            mode = path.stat().st_mode
            path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _remove_tree(directory: Path) -> None:
    """Remove a work directory. Files in it may be read-only hard links: unlink, never chmod,
    because chmod would also change the published file the link points at."""
    shutil.rmtree(directory)


def _append_history(dataset_dir: Path, entry: Mapping[str, Any]) -> None:
    with (dataset_dir / HISTORY).open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _utc(moment: datetime.datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("store times must be timezone-aware")
    return moment.astimezone(datetime.UTC).isoformat()


def _dataset_keys(base: Path) -> list[str]:
    """Dataset keys under a kind directory: directories holding a CURRENT file.

    Does not descend into versions/, cache/ or work directories, which can be large.
    """
    keys = []
    for directory, subdirectories, files in os.walk(base):
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if name not in (VERSIONS, CACHE) and not name.startswith(TMP_PREFIX)
        )
        if CURRENT in files:
            keys.append(Path(directory).relative_to(base).as_posix())
    return sorted(keys)


@contextmanager
def _locked(dataset_dir: Path) -> Iterator[None]:
    """Serialise publishes to one dataset (advisory flock on <dataset>/.lock).

    flock works on local filesystems and most modern NFS; on a filesystem where it
    doesn't, run one publisher per dataset.
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
