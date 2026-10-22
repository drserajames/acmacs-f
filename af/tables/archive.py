"""A lab's original workbooks, kept in the store's ``raw/`` kind once they have been read.

The originals a source_names set reads (af.tables.update) live where the lab or a person
put them: a fludata clone, a delivery folder. Neither is the store's to keep, and a folder
can be reorganised or deleted. So each set is archived as a raw dataset,
``raw/tables/<lab>/<set>``, holding real copies under the rows' ``file`` paths, and every
run reads the archived copy where there is one (as ``raw/gisaid`` holds the GISAID pulls).
The set's root can then go away.

An archived copy is never replaced: a root file that no longer hashes as its archived copy
is an error, since a re-sent workbook is a different source and needs its own row (another
path or set). New originals are read from the root and archived only by a run that
publishes: a dry run writes nothing to the store.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

from af.store import ExternalInput, Provenance, Store
from af.store.manifest import Manifest
from af.store.ref import StoreError, StoreRef
from af.util.artefacts import sha256_path

from .rules import Rule

RAW = "raw"
STEP = "tables-archive"


class ArchiveError(ValueError):
    pass


def dataset(lab: str, name: str) -> str:
    return f"tables/{lab.lower()}/{name}"


@dataclass
class SetFiles:
    """What one set's run reads: each row's ``file`` path mapped to the archived copy, or to
    the root original while it is not archived yet (``pending``)."""

    lab: str
    name: str
    current: StoreRef | None
    read: dict[str, Path] = field(default_factory=dict)
    pending: dict[str, Path] = field(default_factory=dict)

    @property
    def dataset(self) -> str:
        return dataset(self.lab, self.name)


def plan(store: Store, lab: str, name: str, rows: list[tuple[Rule, Path]]) -> SetFiles:
    """Where each row of a set is read from. A row whose original is in neither the archive
    nor the root is fatal (design rule 4); so is a root original that differs from its
    archived copy."""
    try:
        current: StoreRef | None = store.current(RAW, dataset(lab, name))
    except StoreError:
        current = None
    archived: dict[str, str] = {}
    directory = None
    if current is not None:
        directory = store.resolve(current)
        manifest = Manifest.from_bytes((directory / "MANIFEST.json").read_bytes())
        archived = {path: entry.sha256 for path, entry in manifest.by_path().items()}
    files = SetFiles(lab, name, current)
    missing = []
    for rule, original in rows:
        relative = rule["file"]
        if relative in archived and directory is not None:
            if original.is_file() and sha256_path(original) != archived[relative]:
                raise ArchiveError(
                    f"{rule.where}: {original} differs from its archived copy in "
                    f"raw/{files.dataset}@{current.version if current else ''}; a re-sent "
                    "workbook needs its own source_names row"
                )
            files.read[relative] = directory / relative
        elif original.is_file():
            files.read[relative] = files.pending[relative] = original
        else:
            missing.append(f"{rule.where}: {original}")
    if missing:
        raise FileNotFoundError("originals not found, nor archived: " + "; ".join(missing))
    return files


def archive(store: Store, files: SetFiles) -> StoreRef:
    """Publish the set with its pending originals added: unchanged copies are linked from
    CURRENT, new ones copied in. Returns CURRENT unchanged when nothing is pending."""
    if not files.pending and files.current is not None:
        return files.current
    started = dt.datetime.now(dt.UTC)
    with store.build(RAW, files.dataset) as build:
        for relative in sorted(files.read):
            if relative in files.pending:
                build.copy(files.pending[relative], relative)
            else:
                build.link_unchanged(relative)
        provenance = Provenance(
            step=STEP,
            inputs=tuple(ExternalInput.of(files.pending[r]) for r in sorted(files.pending)),
            parameters={"lab": files.lab, "set": files.name},
            started=started,
            finished=dt.datetime.now(dt.UTC),
        )
        ref = build.publish(provenance, summary={"files": len(files.read)})
    for relative, original in files.pending.items():
        copy = store.version_dir(ref) / relative
        if sha256_path(copy) != sha256_path(original):
            raise ArchiveError(f"raw/{files.dataset}: {relative} archived with other bytes")
    return ref
