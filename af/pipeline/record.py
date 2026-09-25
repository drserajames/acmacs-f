"""Incremental rebuilds: one small JSON record per step says what its last run used.

A step is up to date when all three hold:

- its inputs have the same SHA-256 as in its last successful run,
- its parameters are the same (compared as canonical JSON),
- its outputs still exist with the hashes that run recorded. An output someone
  edited or deleted by hand makes the step run again, rather than being trusted.

Anything else means the step runs again, and :func:`check_up_to_date` says why, so a
log shows *which* input or parameter caused the re-run. Records are plain JSON under
the pipeline's state directory, one file per step, readable with ``cat``.

Why not Snakemake or Nextflow: af's steps are Python functions that mostly call
af code, and what decides a re-run here is content (hashes), not file times. A
hundred lines that Sarah can read beat a framework whose re-run rules have to be
learnt, and SLURM is already handled by :mod:`af.run`.

To force a re-run after changing a step's code, give the step a ``code_version``
parameter and bump it.

**Records are keyed by role, not by absolute path**, so moving a tree (a checkout, a
store synced to the HPC or the production server under another root) does not re-run
anything whose content is unchanged. A role is either a name the step gives an input
(``inputs={"table": path}``), or a path made relative to the pipeline's ``root``.
Absolute paths are kept in the record only for people reading it. Without a ``root``,
unnamed inputs and outputs fall back to absolute paths, and then a move re-runs them.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import af
from af.util.artefacts import Artefact, CheckedArtefact, sha256_path


class StepInputMissing(FileNotFoundError):
    def __init__(self, step: str, path: Path) -> None:
        super().__init__(f"step {step!r}: input missing: {path}")
        self.step = step
        self.path = path


RECORD_FORMAT = 2
"""Bumped when the record layout changes. A record in another format is not trusted."""

Inputs = Mapping[str, Path] | Sequence[Path]


@dataclass(frozen=True)
class Fingerprint:
    """What a step run depends on: input hashes by role, and canonical parameters."""

    inputs: dict[str, str]
    parameters: dict[str, Any]
    paths: dict[str, str]


def role_of(path: Path, root: Path | None) -> str:
    """A path's role: relative to ``root`` when it lies under it, else the absolute path."""
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return str(path)


def input_roles(step: str, inputs: Inputs, root: Path | None) -> dict[str, Path]:
    """Role -> path for a step's inputs. Named inputs keep their names."""
    if isinstance(inputs, Mapping):
        return dict(inputs)
    return _unique_roles(step, "input", [(role_of(path, root), path) for path in inputs])


def output_roles(step: str, outputs: Sequence[Artefact], root: Path | None) -> dict[str, Path]:
    pairs = [(role_of(artefact.path, root), artefact.path) for artefact in outputs]
    return _unique_roles(step, "output", pairs)


def _unique_roles(step: str, what: str, pairs: list[tuple[str, Path]]) -> dict[str, Path]:
    roles: dict[str, Path] = {}
    for role, path in pairs:
        if role in roles:
            raise ValueError(f"step {step!r}: two {what}s share the role {role!r}")
        roles[role] = path
    return roles


def fingerprint(
    step: str, inputs: Mapping[str, Path], parameters: Mapping[str, Any]
) -> Fingerprint:
    """Hash every input now, by role. A missing input is fatal (design rule 4)."""
    hashes: dict[str, str] = {}
    for role, path in inputs.items():
        if not path.exists():
            raise StepInputMissing(step, path)
        hashes[role] = sha256_path(path)
    paths = {role: str(path) for role, path in inputs.items()}
    return Fingerprint(inputs=hashes, parameters=canonical(parameters), paths=paths)


def canonical(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Parameters as they will read back from JSON (tuples become lists, keys sorted).

    Raises TypeError for values JSON can't represent, so an unrecordable parameter is
    caught before the step runs, not after.
    """
    result: dict[str, Any] = json.loads(json.dumps(dict(parameters), sort_keys=True))
    return result


def record_path(state_dir: Path, step: str) -> Path:
    return state_dir / f"{step}.json"


def check_up_to_date(
    state_dir: Path, step: str, current: Fingerprint, outputs: Mapping[str, Path]
) -> tuple[bool, str]:
    """Return (up_to_date, reason). ``outputs`` maps role -> where the output is now."""
    path = record_path(state_dir, step)
    if not path.exists():
        return False, "no previous run recorded"
    record = json.loads(path.read_text())
    if record.get("format") != RECORD_FORMAT:
        return False, "record written by an older af (different format)"
    changed = _changed_keys(record["inputs"], current.inputs)
    if changed:
        return False, f"inputs changed: {', '.join(changed)}"
    changed = _changed_keys(record["parameters"], current.parameters)
    if changed:
        return False, f"parameters changed: {', '.join(changed)}"
    recorded_outputs: dict[str, str] = record["outputs"]
    if sorted(outputs) != sorted(recorded_outputs):
        return False, "declared outputs changed"
    for role, sha256 in recorded_outputs.items():
        output = outputs[role]
        if not output.exists():
            return False, f"output missing: {role}"
        if sha256_path(output) != sha256:
            return False, f"output modified since last run: {role}"
    return True, "inputs, parameters and outputs unchanged"


def save_record(
    state_dir: Path,
    step: str,
    current: Fingerprint,
    outputs: Mapping[str, CheckedArtefact],
    started: datetime.datetime,
    finished: datetime.datetime,
) -> Path:
    """Record a successful run, keyed by role. Written to a temporary file and renamed."""
    state_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "format": RECORD_FORMAT,
        "step": step,
        "af_version": af.__version__,
        "inputs": current.inputs,
        "parameters": current.parameters,
        "outputs": {role: output.sha256 for role, output in outputs.items()},
        "paths": {
            "inputs": current.paths,
            "outputs": {role: str(output.path) for role, output in outputs.items()},
        },
        "started": started.isoformat(),
        "finished": finished.isoformat(),
    }
    path = record_path(state_dir, step)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def forget(state_dir: Path, step: str) -> None:
    """Drop a step's record before it runs, so a crash mid-run can't look up to date."""
    record_path(state_dir, step).unlink(missing_ok=True)


def _changed_keys(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    keys = set(before) | set(after)
    return sorted(key for key in keys if before.get(key, _ABSENT) != after.get(key, _ABSENT))


_ABSENT = object()
