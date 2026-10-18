"""Section intro text: the report's own words from config, facts filled from recorded artefacts.

A section may open with an intro page (``intro`` in the report config), as the delivered VCM
report's geographic and tree sections do. Its words are the report's choice, so they live in
config; anything in them that states a fact about the data or a method is a placeholder filled
here from a recorded artefact, never typed into config, because the trees and their methods
change from round to round:

- ``{period}``: the report period, "April 2026 to September 2026" (from the config's period);
- ``{tree_method}`` (tree sections only): what was aligned and how the tree was built, from the
  tree store version's ``tree.json`` ("aligned" and "build", written by af.tree). A tree af
  imported rather than built (its provenance says "not an af build") records neither, and says
  so instead.
- ``{clade_not_defined}`` (geo sections with a clade key only): how many viruses over the period
  are drawn as "clade not defined", with the noun ("1 virus", "2 viruses"), from the geo build's
  clade-key.json (Sarah, Q215). When there is none, the paragraph that says it is left out.

A fact no artefact records is a gap: named, and the build is refused, bring-up or final. Internal
identifiers never reach the page in place of a sentence (00, 18 Oct 2026, after a review found
a NOT RECORDED line on two tree pages).
"""

from __future__ import annotations

import json
import string
from typing import Any

from af.store import Store, StoreRef

PLACEHOLDERS = {
    "period": ("geo", "trees", "maps"),
    "tree_method": ("trees",),
    "clade_not_defined": ("geo",),
}


def placeholders(text: str) -> list[str]:
    """The ``{name}`` fields a text uses; a malformed one is an error (ValueError)."""
    return [name for _, name, _, _ in string.Formatter().parse(text) if name is not None]


def without_paragraphs_using(text: str, name: str) -> str:
    """``text`` without the paragraphs (blank-line separated) that use the placeholder ``name``."""
    kept = [p for p in text.split("\n\n") if name not in placeholders(p)]
    return "\n\n".join(kept)


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


# A tree version published from another toolkit's build says so in its provenance's caveats.
IMPORTED_CAVEAT = "not an af build"
IMPORTED_METHOD = ("This tree was imported from the round's own tree build; af did not align its "
                   "sequences or build it.")  # fmt: skip


def tree_caveats(store: Store, ref: StoreRef) -> list[str]:
    """The caveats a tree version's provenance records (``parameters.caveats``), or none."""
    path = store.resolve(ref) / "PROVENANCE.json"
    caveats = json.loads(path.read_text()).get("parameters", {}).get("caveats") or []
    return [str(c) for c in caveats]


def tree_method(tree: dict[str, Any], caveats: list[str] | None = None) -> tuple[str, list[str]]:
    """The method sentences from a ``tree.json``, and the fields it lacks (empty when complete).

    Only what the version records: the aligned region, the build tool as it reports its
    version, its substitution model, and whether zero-length branches were collapsed. What is
    recorded is said even when something else is not; each missing field is a gap.

    A version whose provenance says it is not an af build, and whose ``tree.json`` records no
    alignment or build, is an imported tree: af knows only that, and says it (no gap). One that
    says so but records a build anyway is contradictory, and every field is checked as usual.
    """
    aligned, build = tree.get("aligned") or {}, tree.get("build") or {}
    imported = any(c.startswith(IMPORTED_CAVEAT) for c in caveats or [])
    if imported and not aligned and not build:
        return IMPORTED_METHOD, []
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
