"""The figure's rule thresholds, read from ``defaults.toml`` (one editable copy, design rule 6).

The parameter classes have no numeric defaults of their own, so a value can only come from a
TOML file: the shipped one, or a copy passed to :func:`load_defaults`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

from af.util.config import load_config

from .aa_labels import LabelParams
from .sections import BandParams, SelectParams


@dataclass(frozen=True)
class Defaults:
    clades: SelectParams
    bands: BandParams
    labels: LabelParams


def load_defaults(path: Path | None = None) -> Defaults:
    """The shipped defaults, or those in ``path`` (every key required)."""
    if path is None:
        return _shipped()
    return load_config(path, Defaults)


@cache
def _shipped() -> Defaults:
    with resources.as_file(resources.files(__package__).joinpath("defaults.toml")) as path:
        return load_config(path, Defaults)
