"""Ancestral reconstruction, with the method chosen from config.

    from af.tree import asr

    backend = asr.get_backend("treetime")          # or from config
    asr.check_gap_capability(backend, deletion_positions)
    states = backend.reconstruct(tree, alignment, work_dir)
    subs = states.substitutions(tree, leaf_sequences)

The default is TreeTime, chosen by Sarah on the measurements in notes/trees/ASR.md. Backends are
interchangeable because their nodes are matched back to af's by clade, not by name or position.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from af.tree.asr.base import (
    AncestralStates,
    Backend,
    BackendUnavailable,
    GapCapabilityError,
    Substitution,
    check_gap_capability,
    translate,
)
from af.tree.asr.iqtree import IqTreeBackend
from af.tree.asr.matching import (
    MatchError,
    match_states_by_clade,
    unmatched_count,
    unmatched_ids,
)
from af.tree.asr.parsimony import ParsimonyBackend
from af.tree.asr.raxml import RaxmlBackend
from af.tree.asr.treetime import TreeTimeBackend

DEFAULT_BACKEND = "treetime"

_BACKENDS: dict[str, type] = {
    "treetime": TreeTimeBackend,
    "iqtree": IqTreeBackend,
    "parsimony": ParsimonyBackend,
    "raxml": RaxmlBackend,
}


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_backend(name: str | None = None, **options: Any) -> Backend:
    """Build a backend by name. An unknown name is an error, never a silent fallback."""
    chosen = name or DEFAULT_BACKEND
    try:
        factory = _BACKENDS[chosen]
    except KeyError:
        raise ValueError(
            f"unknown ASR backend {chosen!r}; available: {', '.join(available_backends())}"
        ) from None
    backend: Backend = factory(**options)
    return backend


def backend_from_config(config: Mapping[str, Any]) -> Backend:
    """Build the backend described by a config section.

    ``{"backend": "treetime", "executable": "...", "gtr": "infer"}``. Absent, the default is used;
    an unknown key is an error rather than being ignored, because a silently dropped option is how
    a run ends up not doing what its config says.
    """
    options = dict(config)
    name = options.pop("backend", None)
    return get_backend(name, **options)


__all__ = [
    "DEFAULT_BACKEND",
    "AncestralStates",
    "Backend",
    "BackendUnavailable",
    "GapCapabilityError",
    "IqTreeBackend",
    "MatchError",
    "ParsimonyBackend",
    "RaxmlBackend",
    "Substitution",
    "TreeTimeBackend",
    "available_backends",
    "backend_from_config",
    "check_gap_capability",
    "get_backend",
    "match_states_by_clade",
    "translate",
    "unmatched_count",
    "unmatched_ids",
]
