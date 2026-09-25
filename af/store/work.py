"""The work area: durable working state for the steps that build store datasets.

The store holds what is *published*. The work area holds what a step needs in order
to resume or skip work: af.pipeline records, SLURM partial results, scratch. It is
kept apart because:

- it is never published and never named by a report manifest;
- it can be thrown away at the cost of recomputation, never of data;
- it must still survive a session or a reboot. A scratchpad is not durable enough,
  and a chain's step records are what make restart-from-change work.

Layout, mirroring the store's kind/dataset keys::

    <work>/WORK.toml                        marker (layout version), like STORE.toml
    <work>/<kind>/<dataset key...>/
        state/                              af.pipeline records (Pipeline(state_dir=...))
        tmp/                                the step's own scratch, e.g. SLURM partial results

Both roots come from config. A ``[paths]`` table with ``store`` and ``work`` is
required: :class:`PathsConfig` has no defaults, and a missing key is a config error.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import af
from af.store.ref import StoreError, check_dataset, check_kind

WORK_FILE = "WORK.toml"
WORK_LAYOUT = 1


@dataclass(frozen=True)
class PathsConfig:
    """The ``[paths]`` table of a step's config: both roots, both required.

    Use it as a field of your config schema::

        @dataclass(frozen=True)
        class MyConfig:
            paths: PathsConfig
            ...

    and in TOML::

        [paths]
        store = "~/AC/eu/store"
        work = "~/AC/eu/work"
    """

    store: Path
    work: Path


@dataclass(frozen=True)
class DatasetWork:
    """The working directories of one dataset's builder."""

    root: Path

    @property
    def state(self) -> Path:
        """Pass as ``Pipeline(state_dir=...)``."""
        return self.root / "state"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"


class Work:
    """An opened work area. Open with :meth:`open`; make one with :meth:`create`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @classmethod
    def create(cls, root: Path) -> Work:
        """Make a new work area. Refuses a directory that already has content."""
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise StoreError(f"cannot create a work area in non-empty directory {root}")
        root.mkdir(parents=True, exist_ok=True)
        (root / WORK_FILE).write_text(
            f'layout = {WORK_LAYOUT}\ncreated_by = "af {af.__version__}"\n'
        )
        return cls(root)

    @classmethod
    def open(cls, root: Path) -> Work:
        """Open an existing work area. A missing or unknown WORK.toml is fatal.

        No silent creation: a typo in the configured path must not start a fresh,
        empty work area and quietly re-run every step.
        """
        marker = Path(root) / WORK_FILE
        if not marker.is_file():
            raise StoreError(f"not a work area (no {WORK_FILE}): {root}")
        layout = tomllib.loads(marker.read_text()).get("layout")
        if layout != WORK_LAYOUT:
            raise StoreError(
                f"{marker}: layout {layout!r} is not supported (expects {WORK_LAYOUT})"
            )
        return cls(Path(root))

    def dataset(self, kind: str, dataset: str) -> DatasetWork:
        """The working directories for ``kind/dataset``, created if missing."""
        work = DatasetWork(self.root / check_kind(kind) / check_dataset(dataset))
        work.state.mkdir(parents=True, exist_ok=True)
        work.tmp.mkdir(parents=True, exist_ok=True)
        return work
