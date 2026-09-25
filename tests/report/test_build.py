"""The report builder must fail loudly in every case B-report-layer §4.1 lists as silent today."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import pytest

from af.report import build, placeholder
from af.report.config import load
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
    record = json.loads((tmp_path / "out" / "test-report.manifest.json").read_text())
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
        build.resolve_all(load(cfg), root)


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
