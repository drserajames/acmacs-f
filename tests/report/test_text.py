"""Intro pages: the report's words from config, facts only from recorded artefacts."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest

from af.report import build, placeholder, text
from af.report.config import load
from af.store import Provenance, Store
from af.util.config import ConfigError

build_needs_latex = pytest.mark.tool(build.LATEX)

T0 = dt.datetime(2026, 9, 1, 12, tzinfo=dt.UTC)
CONFIG = """
[report]
id = "r"
kind = "monthly"
title = "Test"
period = { first = "2026-04", last = "2026-09" }
data_cutoff = 2026-09-30
end_page = true

[figures]
allow_placeholders = ALLOW

[[sections]]
kind = "geo"
title = "Geographic data"
slots = ["geo/x"]
intro = "Month by month from {period}.\\n\\nEach dot is a strain."

[[sections]]
kind = "trees"
title = "Tree"
slots = ["tree/x/report"]
intro = "The tree is coloured by region.\\n\\n{tree_method}"
"""

FULL = {"aligned": {"region": "mature HA (HA1 and HA2)"},
        "build": {"version": "toolname/9.9", "parameters": {"model": "GTR"},
                  "zero_length_collapsed": True}}  # fmt: skip


def test_tree_method_says_only_what_tree_json_records() -> None:
    sentence, missing = text.tree_method(FULL)
    assert missing == []
    assert sentence == (
        "Nucleotide sequences of the mature HA (HA1 and HA2) were aligned. The phylogenetic tree "
        "was constructed using toolname/9.9 under the GTR substitution model, and zero-length "
        "branches were collapsed into multifurcations."
    )
    partial = {"aligned": FULL["aligned"], "build": {"version": "toolname/9.9",
               "parameters": {}, "zero_length_collapsed": True}}  # fmt: skip
    assert text.tree_method(partial) == (
        "Nucleotide sequences of the mature HA (HA1 and HA2) were aligned. The phylogenetic tree "
        "was constructed using toolname/9.9, and zero-length branches were collapsed into "
        "multifurcations.",
        ["tree.json build.parameters.model"],
    )
    assert text.tree_method({"asr": {"backend": "x"}})[1] == [
        "tree.json aligned.region", "tree.json build.version",
        "tree.json build.parameters.model", "tree.json build.zero_length_collapsed",
    ]  # fmt: skip


def test_period_text() -> None:
    assert text.period_text(["2026-04", "2026-05", "2026-09"]) == "April 2026 to September 2026"
    assert text.period_text(["2026-04"]) == "April 2026"


def _setup(tmp: Path, tree: dict[str, Any] | None, allow: bool) -> tuple[Path, Path, Store]:
    cfg = tmp / "r.toml"
    cfg.write_text(CONFIG.replace("ALLOW", "true" if allow else "false"))
    root = tmp / "figs"
    for slot in load(cfg).all_slots():
        placeholder.make(root, slot, slot, T0)
    store = Store.create(tmp / "store")
    with store.build("trees", "x/report") as version:
        (version.path / "tree.json").write_text(json.dumps(tree or {}))
        ref = version.publish(Provenance("test", (), {}, T0, T0))
    i7 = next((root / "tree/x/report").glob("*/figure.i7.json"))
    doc = json.loads(i7.read_text())
    doc["provenance"]["store_refs"] = [ref.to_json()]
    i7.write_text(json.dumps(doc))
    return cfg, root, store


def test_intro_pages_fill_facts_from_the_tree_version(tmp_path: Path) -> None:
    cfg_path, root, store = _setup(tmp_path, FULL, allow=True)
    cfg = load(cfg_path)
    figs = build.resolve_all(cfg, root)
    intros = build.intro_texts(cfg, figs, store)
    assert intros[0] == (["Month by month from April 2026 to September 2026.",
                          "Each dot is a strain."], [])  # fmt: skip
    assert intros[1][0][1].startswith("Nucleotide sequences of the mature HA")
    out = tmp_path / "build"
    out.mkdir()
    tex = build.write_tex(cfg, figs, out, T0, 0, intros).read_text()
    assert tex.count(r"\section{Tree}") == 1  # the heading moves to the intro page
    assert tex.index("coloured by region") < tex.index("fig:tree-x-report")
    assert "Report generated: 2026-09-01 12:00 UTC" in tex


def test_an_unrecorded_method_is_red_in_bring_up_and_refused_in_a_final_report(
    tmp_path: Path,
) -> None:
    cfg_path, root, store = _setup(tmp_path, {"asr": {}}, allow=True)
    cfg = load(cfg_path)
    figs = build.resolve_all(cfg, root)
    _, gaps = build.intro_texts(cfg, figs, store)[1]
    assert len(gaps) == 1 and gaps[0].startswith("tree/x/report (trees/x/report@")
    assert gaps[0].endswith(
        "tree.json build.parameters.model, tree.json build.zero_length_collapsed"
    )
    out = tmp_path / "build"
    out.mkdir()
    tex = build.write_tex(cfg, figs, out, T0, 0, build.intro_texts(cfg, figs, store))
    assert r"\textcolor{red}{NOT RECORDED: tree/x/report (trees/x/report@" in tex.read_text()


def test_intro_config_is_checked(tmp_path: Path) -> None:
    cfg = tmp_path / "r.toml"
    bad = CONFIG.replace("ALLOW", "true").replace(
        'intro = "Month by month from {period}.\\n\\nEach dot is a strain."',
        'intro = "From {period}: {tree_method}"',
    )
    cfg.write_text(bad)
    with pytest.raises(ConfigError, match="tree_method.* is not a placeholder for a geo section"):
        load(cfg)


@build_needs_latex
def test_a_report_with_an_end_page_builds(tmp_path: Path) -> None:
    """The closing page follows the last figure: the page check must allow it (it did not)."""
    cfg_path, root, _ = _setup(tmp_path, FULL, allow=True)
    pdf = build.build(cfg_path, root, tmp_path / "out")
    record = json.loads((tmp_path / "out" / "r.build.json").read_text())
    assert pdf.is_file() and record["output"]["pages"] >= 4
    assert record["text_gaps"]  # no --store: the tree's method cannot be read, so it is a gap
