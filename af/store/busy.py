"""Batches of publishes, and reads that must not see half of one.

A single publish is atomic (af.store.store). A *batch* is not: a sweep that republishes 83
sequences datasets takes minutes, and a map or report built during it reads some datasets
before the sweep and some after (2 Oct 2026: a map read half-swept sequences against
older clades). So:

- a writer that publishes several datasets meant to land together wraps them in
  :meth:`Store.batch <af.store.store.Store.batch>`. That holds a marker,
  ``<store>/BUSY/<name>.json``, naming the holder, its datasets and when it started;
- a reader that builds something from the store (a map, a report) wraps its work in
  :meth:`Store.reading <af.store.store.Store.reading>`. It refuses to start while a marker
  is held, and at the end it fails if a dataset whose CURRENT it read has moved since. A
  publish to a dataset the reader never read does not fail it: long builds must not trip
  on unrelated work. ``override=True`` (a command-line flag, never an environment
  variable) reads anyway, for diagnosis, and the guard records that it did.

A crashed writer must not leave a permanent lock. A marker from this host whose process
is gone (pid plus the process's start time, so a reused pid does not count) is stale; a
marker from another host is stale once older than its ``max_age_hours``. Stale markers
are reported and ignored, and the next batch of the same name replaces one.
``python -m af.store busy <store>`` lists markers; ``--clear <name>`` removes one.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import socket
import subprocess
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from af.store.ref import StoreError

BUSY = "BUSY"
DEFAULT_MAX_AGE_HOURS = 12.0
log = logging.getLogger("af.store")


class StoreBusy(StoreError):
    """A batch holds the store, or a dataset this read used moved during it."""


@dataclass(frozen=True)
class Marker:
    """Who holds a batch: written to ``BUSY/<name>.json``."""

    name: str
    host: str
    pid: int
    process_started: str  # the holder process's own start time, so a reused pid is not it
    started: str  # UTC, ISO 8601
    datasets: tuple[str, ...]
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS

    def age_hours(self) -> float:
        started = datetime.datetime.fromisoformat(self.started)
        return (datetime.datetime.now(datetime.UTC) - started).total_seconds() / 3600

    def describe(self) -> str:
        datasets = ", ".join(self.datasets) or "(none named)"
        return (
            f"{self.name!r} (pid {self.pid} on {self.host}, started {self.started}, "
            f"{self.age_hours():.1f} h ago; datasets: {datasets})"
        )

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "datasets": list(self.datasets)}


def stale_reason(marker: Marker) -> str | None:
    """Why ``marker`` no longer holds the store, or None if it is live.

    On the marker's own host the process decides. Elsewhere, or where processes cannot be
    inspected (a sandbox that forbids ``ps``), only the marker's age can.
    """
    why_unchecked = f"held from another host, {marker.host}"
    if marker.host == socket.gethostname():
        started = process_start(marker.pid)
        if started is None:
            return f"its process {marker.pid} is gone"
        if started is not UNKNOWN and marker.process_started is not UNKNOWN:
            if started != marker.process_started:
                return f"pid {marker.pid} is now a different process"
            return None
        why_unchecked = "processes cannot be inspected here"
    if marker.age_hours() > marker.max_age_hours:
        return (
            f"it is {marker.age_hours():.1f} h old, past its {marker.max_age_hours:g} h limit "
            f"({why_unchecked}, so its process cannot be checked)"
        )
    return None


UNKNOWN = ""  # a process start that could not be read (the marker's process_started too)


def process_start(pid: int) -> str | None:
    """A token for when process ``pid`` started; None if there is no such process;
    :data:`UNKNOWN` if processes cannot be inspected here."""
    proc = Path(f"/proc/{pid}/stat")
    if proc.exists():
        try:
            fields = proc.read_text().rsplit(")", 1)[1].split()
        except OSError:
            return None
        return fields[19]  # starttime, in clock ticks since boot
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # another user's process: it exists
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, check=False
        )
    except OSError:
        return UNKNOWN  # e.g. a sandbox that forbids ps; the pid exists (kill 0 above)
    return result.stdout.strip() or (UNKNOWN if result.returncode == 0 else None)


def busy_dir(root: Path) -> Path:
    return Path(root) / BUSY


def read_markers(root: Path) -> list[Marker]:
    directory = busy_dir(root)
    if not directory.is_dir():
        return []
    markers = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            markers.append(Marker(**{**data, "datasets": tuple(data["datasets"])}))
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise StoreError(f"unreadable batch marker {path}: {error}") from error
    return markers


def live_markers(root: Path, *, report_stale: bool = True) -> list[Marker]:
    """The markers that hold the store now; stale ones are logged (once per call)."""
    live = []
    for marker in read_markers(root):
        reason = stale_reason(marker)
        if reason is None:
            live.append(marker)
        elif report_stale:
            log.warning("ignoring stale batch marker %s: %s", marker.describe(), reason)
    return live


def check_name(name: str) -> str:
    if not name or not all(c.isalnum() or c in "-_." for c in name) or name.startswith("."):
        raise StoreError(f"batch name {name!r}: use letters, digits, '-', '_' and '.' only")
    return name


@contextmanager
def batch(
    root: Path,
    name: str,
    datasets: Iterable[str],
    *,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> Iterator[Marker]:
    """Hold ``BUSY/<name>.json`` for the duration; see :meth:`Store.batch`."""
    path = busy_dir(root) / f"{check_name(name)}.json"
    path.parent.mkdir(exist_ok=True)
    marker = Marker(
        name=name,
        host=socket.gethostname(),
        pid=os.getpid(),
        process_started=process_start(os.getpid()) or UNKNOWN,
        started=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        datasets=tuple(datasets),
        max_age_hours=max_age_hours,
    )
    _create(path, marker)
    try:
        yield marker
    finally:
        _release(path, marker)


def _create(path: Path, marker: Marker) -> None:
    for _attempt in range(2):
        try:
            with path.open("x") as handle:  # O_EXCL: two writers cannot both hold one name
                json.dump(marker.to_json(), handle, indent=2)
                handle.write("\n")
            return
        except FileExistsError:
            (existing,) = [m for m in read_markers(path.parent.parent) if m.name == marker.name]
            reason = stale_reason(existing)
            if reason is None:
                raise StoreBusy(f"batch {existing.describe()} is already running") from None
            log.warning("replacing stale batch marker %s: %s", existing.describe(), reason)
            path.unlink(missing_ok=True)
    raise StoreBusy(f"could not take batch marker {path}")


def _release(path: Path, marker: Marker) -> None:
    """Remove the marker only if it is still ours (it may have been cleared and retaken)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if data.get("pid") == marker.pid and data.get("started") == marker.started:
        path.unlink(missing_ok=True)


def clear(root: Path, name: str) -> Marker:
    """Remove a marker by name (diagnosis; the CLI's --clear)."""
    path = busy_dir(root) / f"{check_name(name)}.json"
    found = [m for m in read_markers(root) if m.name == name]
    if not found:
        raise StoreError(f"no batch marker named {name!r} in {busy_dir(root)}")
    path.unlink()
    return found[0]


# ---- reads ------------------------------------------------------------------------------


@dataclass
class ReadGuard:
    """What one guarded read used: every CURRENT it read, and any markers it overrode."""

    name: str
    root: Path
    started: str
    overridden: list[Marker] = field(default_factory=list)
    reads: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, kind: str, dataset: str, version: str) -> None:
        with self._lock:
            self.reads.setdefault((kind, dataset), set()).add(version)

    def to_json(self) -> dict[str, Any]:
        """For the build's provenance: what it read, and whether it overrode a batch."""
        with self._lock:
            reads = {f"{k}/{d}": sorted(v) for (k, d), v in sorted(self.reads.items())}
        return {
            "read": self.name,
            "started": self.started,
            "overrode_batches": [m.to_json() for m in self.overridden],
            "currents_read": reads,
        }


_active: dict[Path, list[ReadGuard]] = {}
_active_lock = threading.Lock()


def record_current(root: Path, kind: str, dataset: str, version: str) -> None:
    """Called by Store.current: note the read in every guard open on this store."""
    key = Path(root).resolve()
    with _active_lock:
        guards = list(_active.get(key, ()))
    for guard in guards:
        guard.record(kind, dataset, version)


@contextmanager
def reading(
    root: Path,
    name: str,
    *,
    override: bool = False,
    current: Callable[[str, str], str | None],
) -> Iterator[ReadGuard]:
    """Guard a read of the store; see :meth:`Store.reading`. ``current(kind, dataset)``
    gives a dataset's CURRENT version id at exit, without recording it as a read."""
    guard = ReadGuard(
        name=name,
        root=Path(root),
        started=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    )
    held = live_markers(root)
    if held:
        if not override:
            raise StoreBusy(
                f"{name}: the store {root} is being written by "
                + "; ".join(m.describe() for m in held)
                + ". Wait for it to finish, or pass the override to read anyway (diagnosis)."
            )
        holders = "; ".join(m.describe() for m in held)
        log.warning("%s: reading despite %s (override)", name, holders)
        guard.overridden = held
    key = Path(root).resolve()
    with _active_lock:
        _active.setdefault(key, []).append(guard)
    try:
        yield guard
    finally:
        with _active_lock:
            _active[key].remove(guard)
            if not _active[key]:
                del _active[key]
    # Only on a normal exit: an exception already on its way out says more than this.
    moved = _moved(guard, current)
    if moved:
        message = f"{name}: " + "; ".join(moved) + " during this read"
        if override:
            log.warning("%s (override)", message)
        else:
            raise StoreBusy(message + ". Its output mixes versions; build it again.")


def _moved(guard: ReadGuard, current: Callable[[str, str], str | None]) -> list[str]:
    moved = []
    for (kind, dataset), versions in sorted(guard.reads.items()):
        now = current(kind, dataset)
        seen = sorted(versions)
        if len(seen) > 1:
            moved.append(f"{kind}/{dataset} read as {' and '.join(seen)}")
        elif now is not None and now != seen[0]:
            moved.append(f"{kind}/{dataset} moved from {seen[0]} to {now}")
    return moved


def main(argv: list[str] | None = None) -> int:
    """``python -m af.store busy <store> [--clear NAME]``: list or clear batch markers."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m af.store busy", description=main.__doc__)
    parser.add_argument("store", type=Path)
    parser.add_argument("--clear", metavar="NAME", help="remove this batch marker")
    args = parser.parse_args(argv)
    if args.clear:
        removed = clear(args.store, args.clear)
        print(f"removed {removed.describe()}")
        return 0
    markers = read_markers(args.store)
    if not markers:
        print(f"no batch markers in {busy_dir(args.store)}")
    for marker in markers:
        reason = stale_reason(marker)
        state = f"STALE {marker.describe()}: {reason}" if reason else f"LIVE  {marker.describe()}"
        print(state)
    return 0
