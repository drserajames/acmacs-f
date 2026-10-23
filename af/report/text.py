"""Section intro text: the report's own words from config, facts filled from recorded artefacts.

A section may open with an intro page (``intro`` in the report config), as the delivered VCM
report's geographic and tree sections do. Its words are the report's choice, so they live in
config; anything in them that states a fact about the data or a method is a placeholder filled
here from a recorded artefact, never typed into config, because the trees and their methods
change from round to round:

- ``{period}``: the report period, "April 2026 to September 2026" (from the config's period);
- ``{tree_method}`` (tree sections only): what was aligned and how the tree was built, from the
  tree store version's ``tree.json`` ("aligned" and "build", written by af.tree), in the
  delivered report's words ("CMAPLE v2.0.0"). A tree af imported rather than built (its
  provenance says "not an af build") records neither: the round declares them, each fact with
  its source (``imported_tree_methods``), and the sentence is the same.
- ``{clade_not_defined}`` (geo sections with a clade key only): how many viruses over the period
  are drawn as "clade not defined", with the noun ("1 virus", "2 viruses"), from the geo build's
  clade-key.json (Sarah, Q215). When there is none, the paragraph that says it is left out.
- ``{data_cutoff}`` (the report's about page only): the report's data cut-off, "23 October 2026".

The report's about page (``[report.about]``), which says what a report is, takes ``{period}`` and
``{data_cutoff}``: both come from the report config, so it has no gaps.

A fact no artefact records is a gap: named, and the build is refused, bring-up or final. Internal
identifiers never reach the page in place of a sentence (00, 18 Oct 2026, after a review found
a NOT RECORDED line on two tree pages).
"""

from __future__ import annotations

import datetime as dt
import json
import string
from typing import Any

from af.report.config import TreeMethod
from af.store import Store, StoreRef

PLACEHOLDERS = {
    "period": ("geo", "trees", "maps", "about"),
    "tree_method": ("trees",),
    "clade_not_defined": ("geo",),
    "data_cutoff": ("about",),
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
    first, last = (dt.date(int(m[:4]), int(m[5:]), 1) for m in (months[0], months[-1]))
    if first == last:
        return first.strftime("%B %Y")
    return f"{first.strftime('%B %Y')} to {last.strftime('%B %Y')}"


def date_text(date: dt.date) -> str:
    """ "23 October 2026", as the cover prints a meeting day."""
    return f"{date.day} {date.strftime('%B %Y')}"


def tree_json(store: Store, figure: dict[str, Any]) -> tuple[StoreRef, dict[str, Any]]:
    """The tree store version a tree figure was drawn from, and its ``tree.json``."""
    refs = [StoreRef.from_json(r) for r in figure["provenance"].get("store_refs", [])
            if r.get("kind") == "trees"]  # fmt: skip
    if len(refs) != 1:
        raise ValueError(f"{figure.get('title')!r}: {len(refs)} tree store refs; one needed")
    return refs[0], json.loads((store.resolve(refs[0]) / "tree.json").read_text())


# A tree version published from another toolkit's build says so in its provenance's caveats.
IMPORTED_CAVEAT = "not an af build"
# Build tools as the delivered report names them ("CMAPLE v1.0.0"); a tree records the tool as
# it reports itself, "cmaple/2.0.0" (af.tree.build.cmaple.version).
TOOL_NAMES = {"cmaple": "CMAPLE"}
DECLARED_IN = "imported_tree_methods"  # the report config table


def tree_caveats(store: Store, ref: StoreRef) -> list[str]:
    """The caveats a tree version's provenance records (``parameters.caveats``), or none."""
    path = store.resolve(ref) / "PROVENANCE.json"
    caveats = json.loads(path.read_text()).get("parameters", {}).get("caveats") or []
    return [str(c) for c in caveats]


def imported(tree: dict[str, Any], caveats: list[str]) -> bool:
    """An imported tree: its provenance says it is not an af build, and it records no alignment
    or build. One that says so but records a build anyway is checked as an af build."""
    flagged = any(c.startswith(IMPORTED_CAVEAT) for c in caveats)
    return flagged and not tree.get("aligned") and not tree.get("build")


def tool_text(version: str) -> str:
    """ "cmaple/2.0.0" -> "CMAPLE v2.0.0", the delivered report's form; anything else as is."""
    tool, _, number = version.partition("/")
    if not number or number == "unknown":
        return version
    return f"{TOOL_NAMES.get(tool, tool)} v{number}"


def tree_method(
    tree: dict[str, Any], caveats: list[str] | None = None, declared: TreeMethod | None = None
) -> tuple[str, list[str]]:
    """The method sentences from a ``tree.json``, and the fields it lacks (empty when complete).

    Only what the version records: the aligned region, the build tool as it reports its
    version, its substitution model, and whether zero-length branches were collapsed. What is
    recorded is said even when something else is not; each missing field is a gap.

    An imported tree (:func:`imported`) records none of them; the round declares them, with
    their sources (``imported_tree_methods`` in the report config), and the sentences are the
    same. Undeclared, it is one gap. A declaration for a tree af built is a gap too: the
    recorded facts are the ones to print.
    """
    if imported(tree, caveats or []):
        if declared is None:
            return "", [f"an imported tree: no method recorded, none declared ({DECLARED_IN})"]
        tree = {
            "aligned": {"region": declared.region},
            "build": {"version": declared.version, "parameters": {"model": declared.model},
                      "zero_length_collapsed": declared.zero_length_collapsed},
        }  # fmt: skip
    elif declared is not None:
        return "", [f"a declared method for a tree that is not imported ({DECLARED_IN})"]
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
        sentence = f"The phylogenetic tree was constructed using {tool_text(version)}"
        if model is not None:
            sentence += f" under the {model} substitution model"
        if collapsed:
            sentence += ", and zero-length branches were collapsed into multifurcations"
        parts.append(sentence + ".")
    return " ".join(parts), missing


def fill(text: str, values: dict[str, str]) -> str:
    """``text`` with its placeholders replaced; every one must have a value."""
    return text.format_map(values)
