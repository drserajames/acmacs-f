"""Snapshots and report manifests: named sets of store refs.

Versions are already immutable, so a snapshot is only a list of refs. A report
records the same structure (its *manifest*) in the private data repo, which says
exactly which trees, chains and tables it was built from.

"Reproduce this report" starts with :func:`check_refs`: every ref must resolve in this
store and match its manifest hash. Every problem is reported, not just the first.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import af
from af.store.ref import StoreError, StoreRef
from af.store.store import Store

SNAPSHOTS = "snapshots"


def refs_document(refs: Sequence[StoreRef], description: str) -> dict[str, Any]:
    if not refs:
        raise StoreError("a snapshot or manifest must name at least one store ref")
    keys = [(ref.kind, ref.dataset) for ref in refs]
    repeated = sorted(
        {f"{kind}/{dataset}" for kind, dataset in keys if keys.count((kind, dataset)) > 1}
    )
    if repeated:
        raise StoreError(f"a dataset may appear only once: {repeated}")
    return {
        "description": description,
        "af_version": af.__version__,
        "created": datetime.datetime.now(datetime.UTC).isoformat(),
        "refs": [ref.to_json() for ref in sorted(refs, key=lambda ref: (ref.kind, ref.dataset))],
    }


def write_snapshot(store: Store, refs: Sequence[StoreRef], description: str) -> str:
    """Check every ref resolves, then write ``snapshots/<id>.json``; return the id.

    The id hashes the refs only (not the time or description), so snapshotting the
    same versions twice gives the same id and the first file is kept.
    """
    problems = check_refs(store, refs, deep=False)
    if problems:
        raise StoreError("cannot snapshot:\n  " + "\n  ".join(problems))
    document = refs_document(refs, description)
    refs_json = json.dumps(document["refs"], sort_keys=True).encode()
    snapshot_id = hashlib.sha256(refs_json).hexdigest()[:16]
    path = store.root / SNAPSHOTS / f"{snapshot_id}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    return snapshot_id


def read_snapshot(store: Store, snapshot_id: str) -> list[StoreRef]:
    return read_manifest(store.root / SNAPSHOTS / f"{snapshot_id}.json")


def write_manifest(path: Path, refs: Sequence[StoreRef], description: str) -> Path:
    """Write a report manifest (the snapshot format) at ``path``, e.g. in acmacs-f-data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(refs_document(refs, description), indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(path: Path, *, include_pinned: bool = False) -> list[StoreRef]:
    """Read a snapshot or report manifest. A missing file is fatal.

    ``refs`` names one version per dataset. A report manifest may also list ``pinned``:
    older versions of those datasets that only pinned report slots reach (a map kept on its
    older tables by a ruling), as ``[{"store": <ref>, "slots": [...]}]``. With
    ``include_pinned`` they follow the refs, so "reproduce this report"
    (:func:`check_refs` over the result) resolves every version the report rests on.
    A snapshot has no ``pinned``, and reads the same either way.
    """
    document = json.loads(Path(path).read_text())
    refs = [StoreRef.from_json(entry) for entry in document["refs"]]
    if not include_pinned:
        return refs
    return refs + pinned_refs(document, refs, Path(path))


def pinned_refs(document: dict[str, Any], refs: Sequence[StoreRef], path: Path) -> list[StoreRef]:
    """The manifest's pinned versions, checked against its refs.

    Each is an OLDER version of a dataset the refs name (so a different version of it), named
    once, with the slots that reach it. Anything else is a malformed manifest, refused rather
    than read past: reproducing a report must not quietly skip a version it rests on.
    """
    entries = document.get("pinned", [])
    current = {(ref.kind, ref.dataset): ref for ref in refs}
    pinned, problems = [], []
    for entry in entries:
        ref = StoreRef.from_json(entry["store"])
        key = (ref.kind, ref.dataset)
        if not entry.get("slots"):
            problems.append(f"{ref}: names no slots")
        if key not in current:
            problems.append(f"{ref}: its dataset is not among the manifest's refs")
        elif current[key].version == ref.version:
            problems.append(f"{ref}: is the refs' own version, not an older pinned one")
        if ref in pinned:
            problems.append(f"{ref}: listed twice")
        pinned.append(ref)
    if problems:
        raise StoreError(f"{path}: malformed 'pinned' list:\n  " + "\n  ".join(problems))
    return pinned


def check_refs(store: Store, refs: Sequence[StoreRef], *, deep: bool) -> list[str]:
    """Problems resolving ``refs`` in ``store``; an empty list means all are present.

    ``deep`` re-hashes every file (slow; use when checking a report manifest).
    """
    problems = []
    for ref in refs:
        try:
            store.resolve(ref, verify=deep)
        except StoreError as error:
            problems.append(str(error))
    return problems
