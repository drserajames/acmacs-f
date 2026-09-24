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
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Iterable, Mapping, Sequence
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


@dataclass(frozen=True)
class Fingerprint:
    """What a step run depends on: input hashes and canonical parameters."""

    inputs: dict[str, str]
    parameters: dict[str, Any]


def fingerprint(step: str, inputs: Iterable[Path], parameters: Mapping[str, Any]) -> Fingerprint:
    """Hash every input now. A missing input is fatal (design rule 4)."""
    hashes: dict[str, str] = {}
    for path in inputs:
        if not path.exists():
            raise StepInputMissing(step, path)
        hashes[str(path)] = sha256_path(path)
    return Fingerprint(inputs=hashes, parameters=canonical(parameters))


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
    state_dir: Path, step: str, current: Fingerprint, outputs: Sequence[Artefact]
) -> tuple[bool, str]:
    """Return (up_to_date, reason). The reason is for the log."""
    path = record_path(state_dir, step)
    if not path.exists():
        return False, "no previous run recorded"
    record = json.loads(path.read_text())
    changed = _changed_keys(record["inputs"], current.inputs)
    if changed:
        return False, f"inputs changed: {', '.join(changed)}"
    changed = _changed_keys(record["parameters"], current.parameters)
    if changed:
        return False, f"parameters changed: {', '.join(changed)}"
    recorded_outputs: dict[str, str] = record["outputs"]
    declared = sorted(str(artefact.path) for artefact in outputs)
    if declared != sorted(recorded_outputs):
        return False, "declared outputs changed"
    for name, sha256 in recorded_outputs.items():
        output = Path(name)
        if not output.exists():
            return False, f"output missing: {name}"
        if sha256_path(output) != sha256:
            return False, f"output modified since last run: {name}"
    return True, "inputs, parameters and outputs unchanged"


def save_record(
    state_dir: Path,
    step: str,
    current: Fingerprint,
    outputs: Sequence[CheckedArtefact],
    started: datetime.datetime,
    finished: datetime.datetime,
) -> Path:
    """Record a successful run. Written to a temporary file and renamed."""
    state_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "step": step,
        "af_version": af.__version__,
        "inputs": current.inputs,
        "parameters": current.parameters,
        "outputs": {str(output.path): output.sha256 for output in outputs},
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
