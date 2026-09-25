"""Declare and verify the outputs of a step, and record where they came from.

Why: in the old pipeline a step could exit 0 having written nothing, an empty file or
half a file, and the next step carried on with stale output (design rule 3). Every af
step declares its outputs as :class:`Artefact` objects and calls
:func:`check_artefacts` before reporting success. A failure names the step and the
file, and all failures are reported at once.

Provenance (design rule 5): :func:`write_provenance` writes
``<output>.provenance.json`` next to an output, recording the step, the af version,
every input with its SHA-256, the parameters and the start and finish times. That is
enough to tell later whether an output is stale.

An artefact may be a file or a directory. A directory's hash covers the relative path
and content of every file under it, so renaming a file changes the hash.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import af

PROVENANCE_SUFFIX = ".provenance.json"
_CHUNK = 1 << 20


class ArtefactError(RuntimeError):
    """One or more declared outputs of a step are missing, empty or unparsable."""

    def __init__(self, step: str, failures: list[tuple[Path, str]]) -> None:
        self.step = step
        self.failures = failures
        lines = "\n".join(f"  - {path}: {reason}" for path, reason in failures)
        super().__init__(f"step {step!r} did not produce valid output:\n{lines}")


@dataclass(frozen=True)
class Artefact:
    """An output a step promises to produce.

    ``parse`` is an optional check that reads the file and raises if it is not valid
    (for example ``json.loads`` on the text); its return value is ignored.
    ``allow_empty`` is for outputs that are legitimately empty, and must be explicit.
    """

    path: Path
    parse: Callable[[Path], object] | None = None
    allow_empty: bool = False


@dataclass(frozen=True)
class CheckedArtefact:
    """An artefact that passed its checks, with its content hash."""

    path: Path
    sha256: str
    size: int


def sha256_path(path: Path) -> str:
    """SHA-256 of a file, or of a directory tree (relative paths and file contents)."""
    path = Path(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for file in _files_under(path):
            relative = file.relative_to(path).as_posix()
            digest.update(relative.encode() + b"\0" + sha256_path(file).encode() + b"\n")
        return digest.hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def check_artefact(step: str, artefact: Artefact) -> CheckedArtefact:
    """Verify one artefact; raise :class:`ArtefactError` if it fails."""
    return check_artefacts(step, [artefact])[0]


def check_artefacts(step: str, artefacts: Iterable[Artefact]) -> list[CheckedArtefact]:
    """Verify every artefact of ``step``: exists, non-empty, parses. Returns their hashes.

    Raises :class:`ArtefactError` listing every failure. A step that declares no
    artefacts at all is also an error: a step with nothing to check cannot be checked.
    """
    artefacts = list(artefacts)
    if not artefacts:
        raise ArtefactError(step, [(Path("-"), "step declared no artefacts")])
    checked: list[CheckedArtefact] = []
    failures: list[tuple[Path, str]] = []
    for artefact in artefacts:
        reason = _problem(artefact)
        if reason:
            failures.append((artefact.path, reason))
        else:
            checked.append(_checked(artefact.path))
    if failures:
        raise ArtefactError(step, failures)
    return checked


def _problem(artefact: Artefact) -> str | None:
    path = artefact.path
    if not path.exists():
        return "missing"
    size = _size(path)
    if size == 0 and not artefact.allow_empty:
        return "empty directory" if path.is_dir() else "empty file"
    if artefact.parse is not None:
        try:
            artefact.parse(path)
        except Exception as error:
            return f"does not parse: {type(error).__name__}: {error}"
    return None


def _checked(path: Path) -> CheckedArtefact:
    return CheckedArtefact(path=path, sha256=sha256_path(path), size=_size(path))


def _size(path: Path) -> int:
    if path.is_dir():
        return sum(file.stat().st_size for file in _files_under(path))
    return path.stat().st_size


def _files_under(directory: Path) -> list[Path]:
    """Files under ``directory``, in a fixed order (sorted by relative POSIX path)."""
    files = [path for path in directory.rglob("*") if path.is_file()]
    return sorted(files, key=lambda path: path.relative_to(directory).as_posix())


def provenance_path(output: Path) -> Path:
    """Where the provenance record for ``output`` lives: beside it."""
    return output.with_name(output.name + PROVENANCE_SUFFIX)


def write_provenance(
    step: str,
    output: CheckedArtefact,
    *,
    inputs: Iterable[Path],
    parameters: Mapping[str, Any],
    started: datetime.datetime,
    finished: datetime.datetime,
) -> Path:
    """Write ``<output>.provenance.json`` and return its path.

    Inputs are hashed now, so the record says exactly what the output was built from.
    A missing input is fatal. ``parameters`` must be JSON-serialisable; they are
    written with sorted keys so the same parameters always give the same text.
    """
    record = {
        "step": step,
        "af_version": af.__version__,
        "output": {"path": str(output.path), "sha256": output.sha256, "size": output.size},
        "inputs": [{"path": str(path), "sha256": sha256_path(path)} for path in inputs],
        "parameters": dict(parameters),
        "started": _utc(started),
        "finished": _utc(finished),
    }
    target = provenance_path(output.path)
    _write_json_atomically(target, record)
    return target


def read_provenance(output: Path) -> dict[str, Any]:
    """Read the provenance record of ``output``; a missing record is fatal."""
    with provenance_path(output).open() as stream:
        record: dict[str, Any] = json.load(stream)
    return record


def _utc(moment: datetime.datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("provenance times must be timezone-aware")
    return moment.astimezone(datetime.UTC).isoformat()


def _write_json_atomically(target: Path, record: Mapping[str, Any]) -> None:
    """Write JSON so a reader never sees half a file (write a temporary, then rename)."""
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
