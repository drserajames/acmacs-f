"""The report builder must fail loudly in every case B-report-layer §4.1 lists as silent today."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from af.report import build, placeholder
from af.report.config import load
from af.report.provenance import ProvenanceError
from af.store import Provenance, Store, StoreError, StoreRef, check_refs, read_manifest
from af.util.artefacts import sha256_path
from af.util.config import ConfigError

from ..clades import test_coverage as coverage

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

needs_latex = pytest.mark.tool(build.LATEX)


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


def _clade_coloured(tmp: Path, grow: bool) -> tuple[Path, Path, Store, StoreRef, StoreRef]:
    """Figures coloured from the store: sequences republished after the clades were labelled.

    ``grow`` adds a clade call (behind in content); otherwise only a name changes (same content).
    """
    cfg, root = _setup(tmp, allow=False)
    store, dataset, first, clades = coverage.labelled(tmp)
    rows = [(*row[:4], row[4].upper()) for row in coverage.ROWS]
    if grow:
        rows.append(("EPI_ISL_920003", "EPI920003", "P", "good", "virus three"))
    read = coverage.version(store, rows, dataset)
    for slot in load(cfg).all_slots():
        _real_figure(root, slot, [read, clades], T0 + dt.timedelta(hours=1), "v1")
    return cfg, root, store, first, read


def test_clades_labelled_from_same_content_sequences_are_recorded_not_refused(
    tmp_path: Path,
) -> None:
    """Two sequences labels under one figure, identical clade-call inputs: info, not a conflict."""
    cfg, root, store, first, read = _clade_coloured(tmp_path, grow=False)
    use = build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                                 tmp_path / "m.json", deep=False)  # fmt: skip
    assert use.clades_same_content == {"h3": (first.version, read.version)}
    assert first not in use.refs and read in use.refs  # the older label is not followed
    assert use.to_json()["clades_same_content"] == {"h3": [first.version, read.version]}


def test_clades_behind_in_content_are_refused(tmp_path: Path) -> None:
    cfg, root, store, _, read = _clade_coloured(tmp_path, grow=True)
    with pytest.raises(ProvenanceError, match=f"behind {read.version}.* in content"):
        build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                               tmp_path / "m.json", deep=False)  # fmt: skip


def test_clades_naming_no_sequences_version_are_refused(tmp_path: Path) -> None:
    cfg, root = _setup(tmp_path, allow=False)
    store = Store.create(tmp_path / "store")
    read = _publish(store, "sequences", "h3", "sequences v1")
    clades = _publish(store, "clades", "h3", "clades with no sequences input")
    for slot in load(cfg).all_slots():
        _real_figure(root, slot, [read, clades], T0 + dt.timedelta(hours=1), "v1")
    with pytest.raises(ProvenanceError, match="names 0 sequences/h3 versions"):
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
        == "21-24 September 2026"  # hyphen, as delivered
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
    # the three blank cells keep the panel width; panels framed as delivered
    assert tex.count(r"\hspace*{\dimexpr(\textheight-70pt)/2\relax}") == 3
    assert r"\includegraphics[width=\dimexpr\textheight*18/30\relax,frame]" in tex
    record = json.loads((tmp_path / "out" / "vcm-test.build.json").read_text())
    # cover, contents, 2 geo pages (4 months, 3 per page), 1 landscape map page
    assert record["output"]["pages"] == 5 and pdf.is_file()


def test_a_window_continues_in_the_same_landscape_and_the_geometry_is_set_in_the_body(
    tmp_path: Path,
) -> None:
    """As delivered: a window's next page is \\newpage\\vspace*{-2pt} in the same landscape (a
    fresh landscape put both rows 24pt high), and \\newgeometry after \\begin{document}
    (review, 18 Oct 2026). Two windows, the first titled "", 7 panels: 2 pages each."""
    cfg = tmp_path / "v.toml"
    cfg.write_text(VCM.replace(
        'slots = ["map/a", "map/b", "-", "map/c", "-", "-"]\n'
        'windows = [{ name = "all", title = "" }]',
        'slots = ["map/a", "map/b", "map/c", "map/d", "map/e", "map/f", "map/g"]\n'
        'windows = [{ name = "all", title = "" }, { name = "6m", title = "(6m)" }]',
    ))  # fmt: skip
    section = load(cfg).sections[1]
    assert section.run_starts([]) == [True, False, True, False]
    root = tmp_path / "figs"
    for slot in load(cfg).all_slots():
        placeholder.make(root, slot, slot, T0)
    cfg_ = load(cfg)
    out = tmp_path / "build"
    out.mkdir()
    tex = build.write_tex(cfg_, build.resolve_all(cfg_, root), out, T0).read_text()
    maps = tex[tex.index(r"\begin{landscape}") :]
    assert maps.count(r"\begin{landscape}") == maps.count(r"\end{landscape}") == 2
    assert maps.count(r"\newpage\vspace*{-2pt}") == 2
    assert maps.index(r"\newpage\vspace*{-2pt}") < maps.index(r"\end{landscape}")
    body = tex[tex.index(r"\begin{document}") :]
    assert body.index(r"\newgeometry{top=3em,bottom=5em,left=2.5em,right=2.5em}") < body.index(
        r"\begin{quotation}"
    )


def test_build_record_says_which_af_built_it() -> None:
    import af

    record = build.af_source()
    assert record["version"] == af.__version__
    # In this checkout (editable install) the commit is known; from a wheel it would be null.
    assert record["commit"] is None or len(record["commit"]) == 40


def test_build_record_uses_the_release_commit_when_not_a_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    import sys
    import types

    def no_git(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subprocess, "run", no_git)
    fake = types.ModuleType("af.run.runtime")
    fake.release_info = lambda: {"commit": "a" * 40, "rev": "HEAD"}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "af.run.runtime", fake)
    record = build.af_source()
    assert record["commit"] == "a" * 40 and record["dirty"] is False
    assert record["release"]["rev"] == "HEAD"


def test_a_store_being_published_refuses_the_build(tmp_path: Path) -> None:
    from af.store.busy import StoreBusy

    cfg, root, store, _, _ = _store_setup(tmp_path)
    manifest = tmp_path / "manifest.json"
    with store.batch("a-sweep", ["chains/labx/m"]), pytest.raises(StoreBusy, match="a-sweep"):
        build.build(cfg, root, tmp_path / "out", store_root=store.root, manifest_path=manifest)
    assert not (tmp_path / "out" / "test-report.pdf").exists()
    assert not manifest.exists()


@needs_latex
def test_the_build_record_says_what_the_store_read_saw(tmp_path: Path) -> None:
    cfg, root, store, tree, chain = _store_setup(tmp_path)
    manifest = tmp_path / "manifest.json"
    build.build(cfg, root, tmp_path / "out", store_root=store.root, manifest_path=manifest)
    seen = json.loads((tmp_path / "out" / "test-report.build.json").read_text())["store_read"]
    assert seen["read"] == "report-build" and seen["overrode_batches"] == []
    assert seen["currents_read"]["chains/labx/m"] == [chain.version]
    # --ignore-busy: builds anyway (diagnosis), and the record names the batch it overrode
    with store.batch("a-sweep", ["chains/labx/m"]):
        build.build(cfg, root, tmp_path / "out2", store_root=store.root, manifest_path=manifest,
                    ignore_busy=True)  # fmt: skip
    seen = json.loads((tmp_path / "out2" / "test-report.build.json").read_text())["store_read"]
    assert [m["name"] for m in seen["overrode_batches"]] == ["a-sweep"]


def _two_chains_on_two_tables(
    tmp: Path, pin_m2: bool
) -> tuple[Path, Path, Store, StoreRef, StoreRef]:
    """m1 on a chain over tables v2 (CURRENT); m2 on a chain over tables v1, optionally pinned."""
    cfg, root = _setup(tmp, allow=False)
    store = Store.create(tmp / "store")
    tree = _publish(store, "trees", "a/report", "tree v1")
    old = _publish(store, "tables", "labx/m", "tables v1")
    kept = _publish(store, "chains", "labx/m/kept", "chain on v1", (old,))
    new = _publish(store, "tables", "labx/m", "tables v2")  # CURRENT moves on
    redrawn = _publish(store, "chains", "labx/m/main", "chain on v2", (new,))
    later = T0 + dt.timedelta(hours=1)
    _real_figure(root, "tree/a/x", [tree], later, "v1")
    _real_figure(root, "map/m1/all", [redrawn], later, "v1")
    _real_figure(root, "map/m2/all", [kept], later, "v1")
    if pin_m2:
        cfg.write_text(cfg.read_text().replace(
            "[figures]", '[figures]\npins = { "map/m2/all" = "v1" }'))  # fmt: skip
    return cfg, root, store, old, new


def test_a_pinned_slot_may_rest_on_an_older_version_and_it_is_recorded(tmp_path: Path) -> None:
    cfg, root, store, old, new = _two_chains_on_two_tables(tmp_path, pin_m2=True)
    use = build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                                 tmp_path / "m.json", deep=False)  # fmt: skip
    assert new in use.refs and old not in use.refs  # the report's own version is CURRENT
    assert use.pinned_versions == {old: ["map/m2/all"]}
    assert old in use.all_refs() and new in use.all_refs()
    assert use.to_json()["pinned_versions"] == {f"tables/labx/m@{old.version}": ["map/m2/all"]}


def test_two_versions_under_unpinned_slots_are_still_refused(tmp_path: Path) -> None:
    cfg, root, store, _, _ = _two_chains_on_two_tables(tmp_path, pin_m2=False)
    with pytest.raises(ProvenanceError, match="rest on different versions of the same dataset"):
        build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                               tmp_path / "m.json", deep=False)  # fmt: skip


def test_a_pinned_slot_may_be_drawn_from_an_older_version_of_a_shared_dataset(
    tmp_path: Path,
) -> None:
    """The direct refs too: two maps on two versions of one chain, the older one pinned."""
    cfg, root, store, tree, chain = _store_setup(tmp_path)
    newer = _publish(store, "chains", "labx/m", "chain v2")
    _real_figure(root, "map/m1/all", [newer, tree], T0 + dt.timedelta(hours=2), "v2")
    cfg.write_text(cfg.read_text().replace(
        "[figures]", '[figures]\npins = { "map/m2/all" = "v1" }'))  # fmt: skip
    use = build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                                 tmp_path / "m.json", deep=False)  # fmt: skip
    assert newer in use.refs and use.pinned_versions == {chain: ["map/m2/all"]}


@needs_latex
def test_the_manifest_lists_every_version_a_pinned_report_rests_on(tmp_path: Path) -> None:
    cfg, root, store, old, new = _two_chains_on_two_tables(tmp_path, pin_m2=True)
    manifest = tmp_path / "manifest.json"
    build.build(cfg, root, tmp_path / "out", store_root=store.root, manifest_path=manifest)
    assert new in read_manifest(manifest) and old not in read_manifest(manifest)  # a snapshot
    document = json.loads(manifest.read_text())
    assert document["pinned"] == [{"store": old.to_json(), "slots": ["map/m2/all"]}]
    record = json.loads((tmp_path / "out" / "test-report.build.json").read_text())
    assert record["store"]["pinned_versions"] == {f"tables/labx/m@{old.version}": ["map/m2/all"]}
    # and af.store reads back every version the report rests on, which all resolve
    every = read_manifest(manifest, include_pinned=True)
    assert old in every and new in every and check_refs(store, every, deep=True) == []


@needs_latex
def test_the_build_record_carries_the_map_steps_warnings(tmp_path: Path) -> None:
    cfg, root, store, _, _ = _store_setup(tmp_path)
    i7 = root / "map/m1/all/v1/figure.i7.json"
    doc = json.loads(i7.read_text())
    doc["provenance"]["decisions"] = {"unpainted_clades": [{"warning": "WARNING: a sentence."}]}
    i7.write_text(json.dumps(doc))
    manifest = tmp_path / "manifest.json"
    build.build(cfg, root, tmp_path / "out", store_root=store.root, manifest_path=manifest)
    record = json.loads((tmp_path / "out" / "test-report.build.json").read_text())
    assert record["warnings"] == [{"slot": "map/m1/all", "warning": "WARNING: a sentence."}]


def test_with_every_version_pinned_the_reports_own_is_current_or_refused(tmp_path: Path) -> None:
    """Version ids are hashes: when no unpinned slot picks the report's version, CURRENT does."""
    cfg, root, store, old, new = _two_chains_on_two_tables(tmp_path, pin_m2=True)
    cfg.write_text(cfg.read_text().replace(
        'pins = { "map/m2/all" = "v1" }',
        'pins = { "map/m1/all" = "v1", "map/m2/all" = "v1" }'))  # fmt: skip
    use = build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                                 tmp_path / "m.json", deep=False)  # fmt: skip
    assert new in use.refs and use.pinned_versions == {old: ["map/m2/all"]}
    assert use.pinned_resolved_by_current == ["tables/labx/m"]
    _publish(store, "tables", "labx/m", "tables v3")  # now neither pinned version is CURRENT
    with pytest.raises(ProvenanceError, match="every version is under pinned slots and none"):
        build.check_provenance(load(cfg), build.resolve_all(load(cfg), root), store.root,
                               tmp_path / "m.json", deep=False)  # fmt: skip


def _tree_on_older_sequences(tmp: Path) -> tuple[Path, Path, Store, StoreRef, StoreRef, dict]:
    """Figures read sequences v2; their clades sit on a tree exported from v1 (as B/Vic, 11 Oct)."""
    from af.store.ref import ExternalInput

    cfg, root = _setup(tmp, allow=False)
    store = Store.create(tmp / "store")
    old = _publish(store, "sequences", "bx", "sequences v1")
    read = _publish(store, "sequences", "bx", "sequences v2")
    export = tmp / "export"
    export.mkdir()
    (export / "alignment.fasta").write_text(">a\nACGT\n")
    (export / "leaves.parquet").write_text("leaves")
    files = {name: sha256_path(export / name) for name in ("alignment.fasta", "leaves.parquet")}
    started = dt.datetime(2026, 9, 1, tzinfo=dt.UTC)
    with store.build("trees", "bx/weekly") as version:
        (version.path / "data.txt").write_text("a tree")
        inputs = (ExternalInput.of(export / "alignment.fasta"),
                  ExternalInput.of(export / "leaves.parquet"), old)  # fmt: skip
        tree = version.publish(Provenance("test", inputs, {}, started, started))
    clades = _publish(store, "clades", "bx", "clades", (tree, read))
    for slot in load(cfg).all_slots():
        _real_figure(root, slot, [read, clades], T0 + dt.timedelta(hours=1), "v1")
    return cfg, root, store, old, read, files


def _export_record(records: Path, read: StoreRef, files: dict, **change: Any) -> Path:
    path = records / "bx" / files["alignment.fasta"][:16] / f"{read.version}.export.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"export": {"sequences": read.to_json(), "outgroup": "an outgroup"},
              "files": files, "selection_sha256": "0" * 64, "af_release": "a release",
              "written": "2026-10-11T00:00:00+00:00", "how": "back-filled", **change}  # fmt: skip
    path.write_text(json.dumps(record))
    return path


def test_a_tree_exported_from_other_sequences_needs_an_export_record(tmp_path: Path) -> None:
    cfg, root, store, old, read, files = _tree_on_older_sequences(tmp_path)
    figs, records = build.resolve_all(load(cfg), root), tmp_path / "exports"

    def check(export_records: Path | None) -> Any:
        return build.check_provenance(load(cfg), figs, store.root, tmp_path / "m.json",
                                      deep=False, export_records=export_records)  # fmt: skip

    with pytest.raises(ProvenanceError, match="--export-records"):
        check(None)
    with pytest.raises(ProvenanceError, match="has no export record"):
        check(records)
    path = _export_record(records, read, files)
    use = check(records)
    assert use.trees_same_content == {"bx/weekly": (old.version, read.version, str(path),
                                                    "back-filled")}  # fmt: skip
    assert old not in use.refs and read in use.refs  # the older label is not followed
    assert use.to_json()["trees_same_content"]["bx/weekly"][3] == "back-filled"


def test_a_malformed_or_other_export_record_is_refused_never_a_pass(tmp_path: Path) -> None:
    cfg, root, store, old, read, files = _tree_on_older_sequences(tmp_path)
    figs, records = build.resolve_all(load(cfg), root), tmp_path / "exports"

    def check() -> Any:
        return build.check_provenance(load(cfg), figs, store.root, tmp_path / "m.json",
                                      deep=False, export_records=records)  # fmt: skip

    path = _export_record(records, read, files, how=None)
    with pytest.raises(ProvenanceError, match="malformed export record: how is None"):
        check()
    _export_record(records, read, files, export={"sequences": old.to_json()})
    with pytest.raises(ProvenanceError, match="malformed export record: export.sequences"):
        check()
    path.write_text("{not json")
    with pytest.raises(ProvenanceError, match="unreadable export record"):
        check()
    path.unlink()
    other = {**files, "alignment.fasta": "f" * 64}  # v2 exports something else
    _export_record(records, read, other)
    with pytest.raises(ProvenanceError, match="behind in content"):
        check()


# ---- the delivered report's page (Q201 a, Q203 a) --------------------------------------

SHIPPED_STYLE = VCM.replace(
    'title = "Test consultation"\nsubtitle = "for a test hemisphere"',
    'title = ["Test consultation", "on a test question"]\nsubtitle = "for a test hemisphere"\n'
    'centre = ["Lab of X", "University of Y"]\ntimezone = "Europe/London"\nend_page = true',
)


def test_title_and_centre_may_be_given_as_cover_lines(tmp_path: Path) -> None:
    cfg_path = tmp_path / "v.toml"
    cfg_path.write_text(SHIPPED_STYLE)
    cfg = load(cfg_path)
    cover = "\n".join(build._cover(cfg, 0, 0, T0))
    assert r"\textbf{Test consultation\\ on a test question\\ for a test hemisphere}" in cover
    assert r"Lab of X\par\vspace{10pt}University of Y\par" in cover
    # Q203 a: the period, the data cutoff and the build time are not on the cover
    for gone in ("Data up to", "Built", "November 2025", "titlepage"):
        assert gone not in cover, gone


def test_an_empty_cover_line_or_an_unknown_time_zone_is_refused(tmp_path: Path) -> None:
    cfg_path = tmp_path / "v.toml"
    cfg_path.write_text(SHIPPED_STYLE.replace('"University of Y"', '""').replace(
        "Europe/London", "Mars/Olympus"))  # fmt: skip
    with pytest.raises(ConfigError) as err:
        load(cfg_path)
    assert "report.centre: an empty line" in str(err.value)
    assert "report.timezone: 'Mars/Olympus' is not an IANA time zone" in str(err.value)


def test_no_caption_under_a_real_figure_but_its_label_and_a_placeholder_mark_stay(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "v.toml"
    cfg_path.write_text(SHIPPED_STYLE)
    root = tmp_path / "figs"
    for slot in load(cfg_path).all_slots():
        placeholder.make(root, slot, slot, T0)
    figs = build.resolve_all(load(cfg_path), root)
    slot = "map/a/all"
    mark = build._figure_mark(figs[slot], slot, r"\scriptsize")
    assert mark.endswith(r"\zlabel{fig:map-a-all}")
    assert r"\textcolor{red}{PLACEHOLDER " in mark  # a stand-in stays marked on the page
    real = build.Resolved.__new__(build.Resolved)  # the same figure, not a placeholder
    object.__setattr__(real, "figure", type("F", (), {"placeholder": False,
                                                      "title": "t"})())  # fmt: skip
    assert build._figure_mark(real, slot, r"\scriptsize") == r"\zlabel{fig:map-a-all}"


def test_provenance_goes_into_the_pdf_metadata(tmp_path: Path) -> None:
    cfg_path = tmp_path / "v.toml"
    cfg_path.write_text(SHIPPED_STYLE)
    meta = build._pdf_metadata(load(cfg_path), T0)
    assert meta.startswith(r"\hypersetup{pdftitle={Test consultation on a test question for a")
    assert "pdfsubject={vcm-test (vcm), period 2025-11 to 2026-02, data up to 2026-02-20}" in meta
    assert "built 2026-09-01T12:00:00+00:00" in meta


@needs_latex
def test_the_delivered_page_style_builds(tmp_path: Path) -> None:
    cfg_path = tmp_path / "v.toml"
    cfg_path.write_text(SHIPPED_STYLE)
    root = tmp_path / "figs"
    for slot in load(cfg_path).all_slots():
        placeholder.make(root, slot, slot, T0)
    build.build(cfg_path, root, tmp_path / "out")
    tex = (tmp_path / "out" / "build" / "report.tex").read_text()
    assert r"\documentclass[a4paper,12pt]{article}" in tex and r"\usepackage{times}" in tex
    assert "hidelinks" not in tex and r"\contentsmargin{120pt}" in tex
    assert "Report generated: 2026-" in tex and ("BST" in tex or "GMT" in tex)
    record = json.loads((tmp_path / "out" / "vcm-test.build.json").read_text())
    # cover (page 1, counted), contents, 2 geo pages, 1 map page, and the closing page
    assert record["output"]["pages"] == 6


# ---- the geo intro page's clade key (style step 16) ---------------------------------------

GEO_KEY = VCM.replace(
    'slots = ["geo/x"]\ngrid = [1, 3]',
    'slots = ["geo/x"]\ngrid = [1, 3]\nintro = "Months {period}\\n\\nEach dot is a test strain. '
    'Undefined: {clade_not_defined}."\nclade_key = "x"\n'
    'clade_key_classes = {not_sequenced = "no seq", clade_not_defined = "no clade"}\n'
    'clade_not_defined_states = ["not_observable"]',
)
NOT_SEQ = {"fill": "transparent", "edge": "#a0a0a0", "edge_width": 0.5}
UNDEFINED = {"fill": "#ffffff", "edge": "#606060", "edge_width": 0.6}
KEY = {"x": {"entries": [
    {"legend": "Old (q1)", "keys": ["Old"], "colour": "#112233", "per_month": {}, "window": 0},
    {"legend": "New 9Z", "keys": ["New"], "colour": "#a0b0c0", "per_month": {}, "window": 3},
], "uncoloured_classes": [
    {"class": "not_sequenced", **NOT_SEQ, "per_month": {"2026-01": 4}, "window": 4},
    {"class": "clade_not_defined", **UNDEFINED, "per_month": {"2026-01": 2}, "window": 2,
     "states": {"not_observable": 2}},
]}}  # fmt: skip


def _geo_key_setup(tmp: Path, key: dict[str, Any] | None = KEY) -> tuple[Path, Path]:
    cfg = tmp / "v.toml"
    cfg.write_text(GEO_KEY)
    root = tmp / "figs"
    for slot in load(cfg).all_slots():
        placeholder.make(root, slot, slot, T0)
    if key is not None:
        (root / "geo").mkdir(parents=True, exist_ok=True)
        (root / build.GEO_CLADE_KEY).write_text(json.dumps(key))
    return cfg, root


def _dots(*dots: dict[str, Any]) -> dict[str, Any]:
    return {"locations": [{"name": "P", "points": list(dots)}]}


def test_the_clade_key_is_listed_newest_first_with_the_uncoloured_classes_last(
    tmp_path: Path,
) -> None:
    cfg_path, root = _geo_key_setup(tmp_path)
    cfg = load(cfg_path)
    keys, record = build.clade_keys(cfg, build.resolve_all(cfg, root), root)
    assert keys == {0: [
        ("New 9Z", "a0b0c0", "", None), ("Old (q1)", "112233", "", None),
        ("no seq", "", "a0a0a0", 0.5), ("no clade", "ffffff", "606060", 0.6),
    ]}  # fmt: skip
    assert record is not None and record["sha256"] == sha256_path(root / build.GEO_CLADE_KEY)
    assert record["sections"] == {"X geographic data": "x"}
    assert record["uncoloured_classes"] == {
        "X geographic data": {"not_sequenced": 4, "clade_not_defined": 2}
    }
    page = "\n".join(build._intro_page(cfg.sections[0], (["Months a", "Each dot"], []), keys[0]))
    # as delivered: the period sentence, the key, then the description
    assert page.index("Months a") < page.index(r"\begin{tabular}{l l}") < page.index("Each dot")
    assert r"\hspace*{\afkeyindent}\begin{tabular}{l l}" in page  # the class's indent, as delivered
    assert r"\cbox[afkey0x1] & Old (q1) \\" in page
    # Q215: each uncoloured class on its own line, in the swatch its dots are drawn in
    assert r"\ubox{none}{afkey0x2e}{0.5} & no seq \\" in page
    assert r"\ubox{afkey0x3}{afkey0x3e}{0.6} & no clade \\" in page
    assert r"\definecolor{afkey0x2}" not in page


def test_a_class_with_no_dot_in_the_window_has_no_row_and_no_paragraph(tmp_path: Path) -> None:
    """00, 18 Oct 2026: "clade not defined (0 viruses)" said nothing; both go when it is 0."""
    cfg_path, root = _geo_key_setup(tmp_path, {"x": {**KEY["x"], "uncoloured_classes": [
        KEY["x"]["uncoloured_classes"][0],
        {**KEY["x"]["uncoloured_classes"][1], "per_month": {}, "window": 0, "states": {}},
    ]}})  # fmt: skip
    cfg_path.write_text(GEO_KEY.replace("test strain. Undefined", "test strain.\\n\\nUndefined"))
    cfg = load(cfg_path)
    figs = build.resolve_all(cfg, root)
    keys, record = build.clade_keys(cfg, figs, root)
    assert [row.label for row in keys[0]] == ["New 9Z", "Old (q1)", "no seq"]
    assert record is not None
    assert record["uncoloured_classes"]["X geographic data"]["clade_not_defined"] == 0
    paragraphs, gaps = build.intro_texts(cfg, figs, None, record)[0]
    assert (
        paragraphs == ["Months November 2025 to February 2026", "Each dot is a test strain."]
        and gaps == []
    )


def test_uncoloured_dots_are_checked_against_their_class_in_the_key(tmp_path: Path) -> None:
    cfg_path, root = _geo_key_setup(tmp_path)
    cfg = load(cfg_path)
    figs = build.resolve_all(cfg, root)
    ok = [
        {"count": 4, "uncoloured": "not_sequenced", **NOT_SEQ},
        {"count": 2, "uncoloured": "clade_not_defined", **UNDEFINED},
    ]
    for dot in ok:
        dot["color"] = dot.pop("fill")  # an I7 point carries its fill as "color"
    figs["geo/x/2026-01"].figure.doc["geo"] = _dots(*ok)
    build.clade_keys(cfg, figs, root)
    figs["geo/x/2026-01"].figure.doc["geo"] = _dots({**ok[1], "edge": "#a0a0a0"})
    with pytest.raises(build.BuildError, match="'clade_not_defined' drawn .*key has"):
        build.clade_keys(cfg, figs, root)
    figs["geo/x/2026-01"].figure.doc["geo"] = _dots({**ok[0], "uncoloured": "clade_not_in_scheme"})
    with pytest.raises(build.BuildError, match="'clade_not_in_scheme' drawn, the key does not"):
        build.clade_keys(cfg, figs, root)
    figs["geo/x/2026-01"].figure.doc["geo"] = _dots({"color": "transparent", "count": 1})
    with pytest.raises(build.BuildError, match="uncoloured dot with no class"):
        build.clade_keys(cfg, figs, root)


def test_every_listed_class_needs_words_and_every_worded_class_a_listing(tmp_path: Path) -> None:
    extra = {"class": "clade_not_in_scheme", **UNDEFINED, "per_month": {}, "window": 1}
    key = {"x": {**KEY["x"], "uncoloured_classes": [*KEY["x"]["uncoloured_classes"], extra]}}
    cfg_path, root = _geo_key_setup(tmp_path, key=key)
    cfg = load(cfg_path)
    with pytest.raises(build.BuildError, match="lists uncoloured class 'clade_not_in_scheme'"):
        build.clade_keys(cfg, build.resolve_all(cfg, root), root)
    # a key from before Q215 lists no classes; clade_not_in_scheme alone may go unlisted
    (root / build.GEO_CLADE_KEY).write_text(json.dumps({"x": {"entries": KEY["x"]["entries"]}}))
    with pytest.raises(build.BuildError) as err:
        build.clade_keys(cfg, build.resolve_all(cfg, root), root)
    assert "2 clade key problem(s)" in str(err.value) and "before Q215" in str(err.value)


def test_a_clade_not_defined_for_an_unexplained_reason_is_refused(tmp_path: Path) -> None:
    key = json.loads(json.dumps(KEY))
    key["x"]["uncoloured_classes"][1]["states"] = {"not_observable": 1, "no_assignment": 1}
    cfg_path, root = _geo_key_setup(tmp_path, key=key)
    cfg = load(cfg_path)
    with pytest.raises(build.BuildError, match=r"does not explain \{'no_assignment': 1\}"):
        build.clade_keys(cfg, build.resolve_all(cfg, root), root)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("clade_not_defined = ", "sequenced = ", r"clade_key_classes: unknown \['sequenced'\]"),
        ('clade_key = "x"\n', "", r"clade_key_classes: needs clade_key"),
    ],
)
def test_key_class_config_problems(tmp_path: Path, old: str, new: str, message: str) -> None:
    cfg_path = tmp_path / "v.toml"
    cfg_path.write_text(GEO_KEY.replace(old, new, 1))
    with pytest.raises(ConfigError, match=message):
        load(cfg_path)


def test_the_clade_not_defined_count_and_reasons_need_its_key_row(tmp_path: Path) -> None:
    cfg_path = tmp_path / "v.toml"
    cfg_path.write_text(GEO_KEY.replace(', clade_not_defined = "no clade"', ""))
    with pytest.raises(ConfigError) as err:
        load(cfg_path)
    assert "clade_not_defined_states: needs a clade_not_defined" in str(err.value)
    assert "intro: {clade_not_defined} needs a clade_not_defined row" in str(err.value)


def test_a_clade_key_from_another_geo_build_is_refused(tmp_path: Path) -> None:
    cfg_path, root = _geo_key_setup(tmp_path)
    cfg = load(cfg_path)
    figs = build.resolve_all(cfg, root)
    dots = [{"color": "#a0b0c0", "count": 2, "clade": "New 9Z"},
            {"color": "#ffffff", "count": 1, "clade": "Other"},
            {"color": "transparent", "count": 4, "clade": None, "uncoloured": "not_sequenced",
             "edge": "#a0a0a0", "edge_width": 0.5}]  # fmt: skip
    figs["geo/x/2026-01"].figure.doc["geo"] = _dots(*dots)
    with pytest.raises(build.BuildError, match="'Other' drawn #ffffff, key has None"):
        build.clade_keys(cfg, figs, root)


def test_a_missing_clade_key_or_subtype_is_refused(tmp_path: Path) -> None:
    cfg_path, root = _geo_key_setup(tmp_path, key=None)
    cfg = load(cfg_path)
    with pytest.raises(build.BuildError, match="clade-key.json: missing"):
        build.clade_keys(cfg, build.resolve_all(cfg, root), root)
    (root / "geo").mkdir(parents=True, exist_ok=True)
    (root / build.GEO_CLADE_KEY).write_text(json.dumps({"y": KEY["x"]}))
    with pytest.raises(build.BuildError, match="'x' is not in"):
        build.clade_keys(cfg, build.resolve_all(cfg, root), root)


def test_a_clade_key_only_on_a_geo_intro_page(tmp_path: Path) -> None:
    cfg_path = tmp_path / "v.toml"
    cfg_path.write_text(GEO_KEY.replace("grid = [3, 2]", 'grid = [3, 2]\nclade_key = "x"'))
    with pytest.raises(ConfigError, match=r"sections\[1\].clade_key: only for a geo section"):
        load(cfg_path)


@needs_latex
def test_the_geo_intro_page_with_its_clade_key_builds(tmp_path: Path) -> None:
    cfg_path, root = _geo_key_setup(tmp_path)
    build.build(cfg_path, root, tmp_path / "out")
    tex = (tmp_path / "out" / "build" / "report.tex").read_text()
    assert r"\definecolor{afkey0x0}{HTML}{a0b0c0}" in tex and r"\usepackage{tikz}" in tex
    assert r"\ubox{none}{afkey0x2e}{0.5} & no seq" in tex and "Undefined: 2 viruses." in tex
    record = json.loads((tmp_path / "out" / "vcm-test.build.json").read_text())
    assert record["clade_key"]["sections"] == {"X geographic data": "x"}
    # cover, contents, the geo intro page, 2 geo pages, 1 map page
    assert record["output"]["pages"] == 6
