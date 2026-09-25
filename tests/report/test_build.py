"""The report builder must fail loudly in every case B-report-layer §4.1 lists as silent today."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import pytest

from af.report import build, placeholder
from af.report.config import load
from af.report.provenance import ProvenanceError
from af.store import Provenance, Store, StoreError, StoreRef, read_manifest
from af.util.artefacts import sha256_path
from af.util.config import ConfigError

T0 = dt.datetime(2026, 9, 1, 12, tzinfo=dt.UTC)
CONFIG = """
[report]
id = "test-report"
kind = "monthly"
title = "Test"
centre = "Lab X"
period = { first = "2026-09", last = "2026-09" }
data_cutoff = 2026-09-30

[figures]
allow_placeholders = ALLOW

[[sections]]
kind = "trees"
title = "Tree"
slots = ["tree/a/x"]

[[sections]]
kind = "maps"
title = "Maps"
slots = ["map/m1", "map/m2"]
windows = [{ name = "all", title = "all" }]
"""

needs_latex = pytest.mark.skipif(shutil.which(build.LATEX) is None, reason="no pdflatex")


def _setup(tmp: Path, allow: bool = True) -> tuple[Path, Path]:
    cfg = tmp / "r.toml"
    cfg.write_text(CONFIG.replace("ALLOW", "true" if allow else "false"))
    root = tmp / "figs"
    for slot in load(cfg).all_slots():
        placeholder.make(root, slot, slot, T0)
    return cfg, root


@needs_latex
def test_builds_and_writes_manifest(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path)
    pdf = build.build(cfg, root, tmp_path / "out")
    record = json.loads((tmp_path / "out" / "test-report.build.json").read_text())
    assert pdf.is_file()
    assert record["output"]["pages"] == 4  # cover, contents, tree, one map page
    assert record["placeholders"] == 3 and len(record["figures"]) == 3
    assert "/Users/" not in (tmp_path / "out" / "build" / "report.tex").read_text()


def test_every_missing_figure_is_listed(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path)
    shutil.rmtree(root / "map" / "m1")
    shutil.rmtree(root / "map" / "m2")
    with pytest.raises(build.BuildError, match="2 figure"):
        build.resolve_all(load(cfg), root)


def test_placeholders_refused_unless_allowed(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path, allow=False)
    with pytest.raises(build.BuildError, match="placeholder"):
        build.build(cfg, root, tmp_path / "out")


def test_pdf_changed_behind_its_i7_is_fatal(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path)
    (root / "map" / "m1" / "all" / "placeholder" / "figure.pdf").write_bytes(b"%PDF-1.4 other")
    with pytest.raises(build.BuildError, match="sha256"):
        build.resolve_all(load(cfg), root)


def test_latest_is_by_created_time_not_by_name(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path)
    placeholder.make(root, "tree/a/x", "newer", T0 + dt.timedelta(days=1), version="aaa-first")
    assert build.resolve_all(load(cfg), root)["tree/a/x"].version == "aaa-first"


def test_pin_overrides_latest(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path)
    placeholder.make(root, "tree/a/x", "newer", T0 + dt.timedelta(days=1), version="newer")
    cfg.write_text(
        cfg.read_text().replace("[figures]", '[figures]\npins = { "tree/a/x" = "placeholder" }')
    )
    assert build.resolve_all(load(cfg), root)["tree/a/x"].version == "placeholder"


def test_same_created_time_is_ambiguous(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path)
    placeholder.make(root, "tree/a/x", "twin", T0, version="twin")
    with pytest.raises(build.BuildError, match="same created time"):
        build.resolve_all(load(cfg), root)


def test_broken_newer_figure_does_not_fall_back(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path)
    bad = root / "tree" / "a" / "x" / "broken"
    bad.mkdir()
    (bad / "figure.i7.json").write_text("{not json")
    with pytest.raises(build.BuildError, match="not JSON"):
        build.resolve_all(load(cfg), root)


def test_unknown_config_key_is_an_error(tmp_path: Path) -> None:
    cfg = tmp_path / "r.toml"
    cfg.write_text(CONFIG.replace("ALLOW", "true").replace('centre = "Lab X"', 'center = "Lab X"'))
    with pytest.raises(ConfigError, match="center"):
        load(cfg)


def test_month_must_be_yyyy_mm(tmp_path: Path) -> None:
    cfg = tmp_path / "r.toml"
    cfg.write_text(CONFIG.replace("ALLOW", "true").replace('first = "2026-09"', 'first = "2026-9"'))
    with pytest.raises(ConfigError, match="YYYY-MM"):
        load(cfg)


def test_pin_for_an_unused_slot_is_an_error(tmp_path: Path) -> None:
    cfg = tmp_path / "r.toml"
    cfg.write_text(CONFIG.replace("ALLOW", "true").replace(
        "[figures]", '[figures]\npins = { "map/nowhere/all" = "v1" }'))  # fmt: skip
    with pytest.raises(ConfigError, match="nowhere"):
        load(cfg)


@needs_latex
def test_latex_failure_is_fatal_and_leaves_no_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, root = _setup(tmp_path)
    original = build.write_tex

    def broken(*args: object, **kwargs: object) -> Path:
        tex = original(*args, **kwargs)  # type: ignore[arg-type]
        tex.write_text(tex.read_text().replace(r"\end{document}", r"\undefinedmacro\end{document}"))
        return tex

    monkeypatch.setattr(build, "write_tex", broken)
    with pytest.raises(build.BuildError, match="pass 1 failed"):
        build.build(cfg, root, tmp_path / "out")
    assert not (tmp_path / "out" / "test-report.pdf").exists()


# ---- store provenance -------------------------------------------------------------------


def _publish(
    store: Store, kind: str, dataset: str, text: str, inputs: tuple[StoreRef, ...] = ()
) -> StoreRef:
    started = dt.datetime(2026, 9, 1, tzinfo=dt.UTC)
    with store.build(kind, dataset) as version:
        (version.path / "data.txt").write_text(text)
        return version.publish(Provenance("test", inputs, {}, started, started))


def _real_figure(
    root: Path, slot: str, refs: list[StoreRef], created: dt.datetime, version: str
) -> None:
    """A minimal non-placeholder figure (an empty map) naming ``refs`` in its provenance."""
    directory = root / slot / version
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / "figure.pdf"
    placeholder.write_pdf(pdf, ["real"], 400, 400)
    doc = {
        "i7_version": 1, "kind": "map", "title": slot, "placeholder": False,
        "figure": {"pdf": pdf.name, "sha256": sha256_path(pdf), "pages": 1},
        "provenance": {"producer": "test", "created": created.isoformat(),
                       "store_refs": [ref.to_json() for ref in refs]},
        "map": {"chart": "c", "window": {"name": "all"}, "viewport": [0, 0, 1, 1],
                "clade_scheme": "s", "antigens": [], "sera": [], "legend": []},
    }  # fmt: skip
    (directory / "figure.i7.json").write_text(json.dumps(doc))


def _store_setup(tmp: Path) -> tuple[Path, Path, Store, StoreRef, StoreRef]:
    cfg, root = _setup(tmp, allow=False)
    store = Store.create(tmp / "store")
    tree = _publish(store, "trees", "a/report", "tree v1")
    chain = _publish(store, "chains", "labx/m", "chain v1")
    later = T0 + dt.timedelta(hours=1)
    _real_figure(root, "tree/a/x", [tree], later, "v1")
    for slot in ("map/m1/all", "map/m2/all"):
        _real_figure(root, slot, [chain, tree], later, "v1")
    return cfg, root, store, tree, chain


@needs_latex
def test_report_manifest_records_store_refs(tmp_path: Path) -> None:
    cfg, root, store, tree, chain = _store_setup(tmp_path)
    manifest = tmp_path / "data" / "reports" / "test-report" / "manifest.json"
    build.build(cfg, root, tmp_path / "out", store_root=store.root, manifest_path=manifest)
    assert sorted(read_manifest(manifest), key=lambda r: r.kind) == [chain, tree]
    record = json.loads((tmp_path / "out" / "test-report.build.json").read_text())
    assert record["store"]["used_by"]["chains/labx/m"] == ["map/m1/all", "map/m2/all"]


def test_refs_need_a_store_and_a_manifest_path(tmp_path: Path) -> None:
    cfg, root, *_ = _store_setup(tmp_path)
    with pytest.raises(build.BuildError, match="--store and --manifest"):
        build.build(cfg, root, tmp_path / "out")


def test_stale_figure_is_refused_unless_pinned(tmp_path: Path) -> None:
    cfg, root, store, _, _ = _store_setup(tmp_path)
    _publish(store, "chains", "labx/m", "chain v2")  # CURRENT moves on; the maps are now stale
    manifest = tmp_path / "manifest.json"
    with pytest.raises(ProvenanceError, match="stale: map/m1/all, map/m2/all"):
        build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                               manifest, deep=False)  # fmt: skip
    cfg.write_text(cfg.read_text().replace(
        "[figures]", '[figures]\npins = { "map/m1/all" = "v1", "map/m2/all" = "v1" }'))  # fmt: skip
    build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                           manifest, deep=False)  # fmt: skip


def test_two_versions_of_one_dataset_is_an_error(tmp_path: Path) -> None:
    cfg, root, store, tree, chain = _store_setup(tmp_path)
    newer = _publish(store, "chains", "labx/m", "chain v2")
    _real_figure(root, "map/m2/all", [newer, tree], T0 + dt.timedelta(hours=2), "v2")
    with pytest.raises(ProvenanceError, match="different versions of the same dataset"):
        build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                               tmp_path / "m.json", deep=False)  # fmt: skip


def test_ref_missing_from_the_store_is_an_error(tmp_path: Path) -> None:
    cfg, root, *_ = _store_setup(tmp_path)
    other = Store.create(tmp_path / "other")  # a store that has never seen these versions
    with pytest.raises((ProvenanceError, StoreError)):
        build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), other.root,
                               tmp_path / "m.json", deep=False)  # fmt: skip


def test_stand_in_figure_needs_bring_up_mode(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path, allow=False)
    for slot in load(cfg).all_slots():
        _real_figure(root, slot, [], T0 + dt.timedelta(hours=1), "standin")
    with pytest.raises(build.BuildError, match="3 not drawn from the store"):
        build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), None, None,
                               deep=False)  # fmt: skip


@needs_latex
def test_manifest_includes_upstream_tables(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path, allow=False)
    store = Store.create(tmp_path / "store")
    tables = _publish(store, "tables", "labx/m", "tables v1")
    chain = _publish(store, "chains", "labx/m/main", "chain v1", (tables,))
    later = T0 + dt.timedelta(hours=1)
    for slot in load(cfg).all_slots():
        _real_figure(root, slot, [chain], later, "v1")
    manifest = tmp_path / "manifest.json"
    build.build(cfg, root, tmp_path / "out", store_root=store.root, manifest_path=manifest)
    assert set(read_manifest(manifest)) == {chain, tables}


def test_current_chain_on_old_tables_is_stale(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path, allow=False)
    store = Store.create(tmp_path / "store")
    tables = _publish(store, "tables", "labx/m", "tables v1")
    chain = _publish(store, "chains", "labx/m/main", "chain v1", (tables,))
    for slot in load(cfg).all_slots():
        _real_figure(root, slot, [chain], T0 + dt.timedelta(hours=1), "v1")
    _publish(store, "tables", "labx/m", "tables v2")  # new tables; the chain was not rebuilt
    with pytest.raises(ProvenanceError, match="stale: .* rest on tables/labx/m"):
        build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                               tmp_path / "m.json", deep=False)  # fmt: skip


# ---- VCM layout: geo months, blank cells, landscape grids, groups, meeting ----------------

VCM = """
[report]
id = "vcm-test"
kind = "vcm"
title = "Test consultation"
subtitle = "for a test hemisphere"
period = { first = "2025-11", last = "2026-02" }
data_cutoff = 2026-02-20
meeting = { start = 2026-02-23, end = 2026-02-26 }

[figures]
allow_placeholders = true

[[sections]]
group = "Subtype X"
kind = "geo"
title = "X geographic data"
slots = ["geo/x"]
grid = [1, 3]

[[sections]]
group = "Subtype X"
kind = "maps"
title = "X maps"
landscape = true
grid = [3, 2]
slots = ["map/a", "map/b", "-", "map/c", "-", "-"]
windows = [{ name = "all", title = "" }]
"""


def test_geo_expands_over_the_period_across_a_year_end(tmp_path: Path) -> None:
    cfg = tmp_path / "v.toml"
    cfg.write_text(VCM)
    slots = load(cfg).all_slots()
    assert [s for s in slots if s.startswith("geo/")] == [
        "geo/x/2025-11",
        "geo/x/2025-12",
        "geo/x/2026-01",
        "geo/x/2026-02",
    ]
    assert [s for s in slots if s.startswith("map/")] == ["map/a/all", "map/b/all", "map/c/all"]


def test_blank_cells_only_in_map_grids(tmp_path: Path) -> None:
    cfg = tmp_path / "v.toml"
    cfg.write_text(VCM.replace('slots = ["geo/x"]', 'slots = ["geo/x", "-"]'))
    with pytest.raises(ConfigError, match="blank cells"):
        load(cfg)


def test_meeting_dates_on_the_cover() -> None:
    from af.report.config import Meeting

    assert (
        build._meeting(Meeting(dt.date(2026, 9, 21), dt.date(2026, 9, 24)))
        == "21--24 September 2026"
    )
    assert build._meeting(Meeting(dt.date(2026, 9, 30), dt.date(2026, 10, 2))) == (
        "30 September -- 2 October 2026"
    )


@needs_latex
def test_vcm_layout_builds(tmp_path: Path) -> None:
    cfg = tmp_path / "v.toml"
    cfg.write_text(VCM)
    root = tmp_path / "figs"
    for slot in load(cfg).all_slots():
        placeholder.make(root, slot, slot, T0)
    pdf = build.build(cfg, root, tmp_path / "out")
    tex = (tmp_path / "out" / "build" / "report.tex").read_text()
    assert r"\begin{landscape}" in tex and r"\setcounter{secnumdepth}{0}" in tex
    assert tex.count(r"\makebox[0.327\linewidth]{}") == 3  # the three blank cells
    record = json.loads((tmp_path / "out" / "vcm-test.build.json").read_text())
    # cover, contents, 2 geo pages (4 months, 3 per page), 1 landscape map page
    assert record["output"]["pages"] == 5 and pdf.is_file()
