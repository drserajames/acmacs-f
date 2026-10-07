"""Section intro text: the report's own words from config, facts filled from recorded artefacts.

A section may open with an intro page (``intro`` in the report config), as the delivered VCM
report's geographic and tree sections do. Its words are the report's choice, so they live in
config; anything in them that states a fact about the data or a method is a placeholder filled
here from a recorded artefact, never typed into config, because the trees and their methods
change from round to round:

- ``{period}``: the report period, "April 2026 to September 2026" (from the config's period);
- ``{tree_method}`` (tree sections only): what was aligned and how the tree was built, from the
  tree store version's ``tree.json`` ("aligned" and "build", written by af.tree).

A fact no artefact records is a gap: named, counted in the build record, and drawn in red as
NOT RECORDED in a bring-up report; a final report with a gap is refused.
"""

from __future__ import annotations

import json
import string
from typing import Any

from af.store import Store, StoreRef

PLACEHOLDERS = {"period": ("geo", "trees", "maps"), "tree_method": ("trees",)}


def placeholders(text: str) -> list[str]:
    """The ``{name}`` fields a text uses; a malformed one is an error (ValueError)."""
    return [name for _, name, _, _ in string.Formatter().parse(text) if name is not None]


def period_text(months: list[str]) -> str:
    """ "April 2026 to September 2026" for a period given as YYYY-MM months."""
    import datetime as dt

    first, last = (dt.date(int(m[:4]), int(m[5:]), 1) for m in (months[0], months[-1]))
    if first == last:
        return first.strftime("%B %Y")
    return f"{first.strftime('%B %Y')} to {last.strftime('%B %Y')}"


def tree_json(store: Store, figure: dict[str, Any]) -> tuple[StoreRef, dict[str, Any]]:
    """The tree store version a tree figure was drawn from, and its ``tree.json``."""
    refs = [StoreRef.from_json(r) for r in figure["provenance"].get("store_refs", [])
            if r.get("kind") == "trees"]  # fmt: skip
    if len(refs) != 1:
        raise ValueError(f"{figure.get('title')!r}: {len(refs)} tree store refs; one needed")
    return refs[0], json.loads((store.resolve(refs[0]) / "tree.json").read_text())


def tree_method(tree: dict[str, Any]) -> tuple[str, list[str]]:
    """The method sentences from a ``tree.json``, and the fields it lacks (empty when complete).

    Only what the version records: the aligned region, the build tool as it reports its
    version, its substitution model, and whether zero-length branches were collapsed. What is
    recorded is said even when something else is not; each missing field is a gap.
    """
    aligned, build = tree.get("aligned") or {}, tree.get("build") or {}
    region, version = aligned.get("region"), build.get("version")
    model = (build.get("parameters") or {}).get("model")
    collapsed = build.get("zero_length_collapsed")
    missing = [
        f"tree.json {name}"
        for name, value in (
            ("aligned.region", region),
            ("build.version", version),
            ("build.parameters.model", model),
            ("build.zero_length_collapsed", collapsed),
        )
        if value is None
    ]
    parts = []
    if region is not None:
        parts.append(f"Nucleotide sequences of the {region} were aligned.")
    if version is not None:
        sentence = f"The phylogenetic tree was constructed using {version}"
        if model is not None:
            sentence += f" under the {model} substitution model"
        if collapsed:
            sentence += ", and zero-length branches were collapsed into multifurcations"
        parts.append(sentence + ".")
    return " ".join(parts), missing


def fill(text: str, values: dict[str, str]) -> str:
    """``text`` with its placeholders replaced; every one must have a value."""
    return text.format_map(values)
