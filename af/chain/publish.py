"""Publish a finished chain as an immutable version in the store (kind `chains`).

A version holds `chain.json`, the current steps' directories (`steps/NNNN/`: merge,
incremental, scratch and chosen maps, step.json) and the review page, in the same
relative layout as the chain's working area, so the review page's links work in both.
Step files are hard-linked (af.store `link`), so versions that share steps cost nothing;
the small files (chain.json, the review page) are copied. Its provenance names the tables
version the chain was built from.

Dataset keys mirror the tables datasets with a variant segment (STORE-LAYOUT.md):
`<lab>/<group>/main` for the production chain, another variant for anything else (for
example a comparison built from ae-era tables).
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Any

from af.store.ref import ExternalInput, Input, StoreRef
from af.store.store import Provenance, Store

KIND = "chains"


class PublishError(RuntimeError):
    pass


def publish_chain(store_root: Path, dataset: str, chain_root: Path) -> StoreRef:
    """Publish the chain in `chain_root` as the next version of `chains/<dataset>`."""
    doc = json.loads((chain_root / "chain.json").read_text())
    if not doc["complete"]:
        raise PublishError(f"{chain_root}: the chain did not run to its last step")
    review = chain_root / "review" / "index.html"
    if not review.exists():
        raise PublishError(f"{chain_root}: no review page; build it before publishing")
    started = datetime.datetime.now(datetime.UTC)
    store = Store.open(store_root)
    with store.build(KIND, dataset) as build:
        for step in doc["steps"]:
            build.link(chain_root / step["directory"], step["directory"])
        shutil.copy2(chain_root / "chain.json", build.path / "chain.json")
        shutil.copytree(chain_root / "review", build.path / "review")
        provenance = Provenance(
            step="af.chain",
            inputs=_inputs(doc["config"]),
            parameters={k: v for k, v in doc["config"].items() if k != "tables"}
            | {
                "tables": [t["table_id"] for t in doc["config"]["tables"]],
                "optimiser": doc["optimiser"],
            },
            started=started,
            finished=datetime.datetime.now(datetime.UTC),
        )
        return build.publish(provenance, summary=_summary(doc))


def _inputs(config: dict[str, Any]) -> tuple[Input, ...]:
    inputs: list[Input] = []
    if config.get("tables_source"):
        inputs.append(StoreRef.from_json(config["tables_source"]))
    else:  # tables from a directory (ae-era comparison): record each file
        inputs.extend(ExternalInput.of(Path(t["path"])) for t in config["tables"])
    if config.get("first_map"):
        inputs.append(ExternalInput.of(Path(config["first_map"])))
    return tuple(inputs)


def _summary(doc: dict[str, Any]) -> dict[str, Any]:
    remade = [s["index"] for s in doc["steps"] if not s["reused"]]
    return {
        "steps": len(doc["steps"]),
        "last_table": doc["steps"][-1]["table_id"],
        "remade": len(remade),
        "restarted_at_step": remade[0] if remade else None,
    }
