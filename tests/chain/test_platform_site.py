"""A step's platform record can say WHERE it ran: a label from the run config, never the host.

A map is repeatable on one machine, not across machines (13 Oct 2026: one map, same tables, options,
seed and starting layouts, built at two sites, differed by 0.46% in stress), so a reader of a
published map needs the site, and must not have to guess it from a compiler string.
"""

import json

import pytest

from af.chain.__main__ import RunSettings
from af.chain.backend import StubOptimiser
from af.chain.engine import Mapper, chain_steps, platform_info, run_chain
from af.util.config import load_config

from .test_engine import config, tables  # noqa: F401  (tables is a fixture)


def _platforms(root):
    return [json.loads(p.read_text())["platform"] for p in sorted(root.rglob("step.json"))]


def test_the_site_is_recorded_only_when_the_run_config_names_one(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    run_chain(cfg, tmp_path / "plain", optimiser=StubOptimiser())
    run_chain(cfg, tmp_path / "sited", optimiser=StubOptimiser(), site="site-a")
    plain, sited = _platforms(tmp_path / "plain"), _platforms(tmp_path / "sited")
    assert plain and all("site" not in p for p in plain)  # records as they always were
    assert [p["site"] for p in sited] == ["site-a"] * len(plain)
    assert [{k: v for k, v in p.items() if k != "site"} for p in sited] == plain


def test_the_site_changes_no_step_parameters(tables, tmp_path):  # noqa: F811
    cfg = config(tables)
    a = chain_steps(cfg, tmp_path / "x", Mapper(StubOptimiser(), None))
    b = chain_steps(cfg, tmp_path / "x", Mapper(StubOptimiser(), None, "site-a"))
    assert [dict(s.parameters) for s in a] == [dict(s.parameters) for s in b]
    assert "site" not in platform_info() and platform_info("site-a")["site"] == "site-a"


def test_the_label_is_a_short_name_not_a_host(tmp_path):
    path = tmp_path / "run.toml"
    body = 'optimiser = "stub"\n{}[paths]\nstore = "s"\nwork = "w"\n'
    path.write_text(body.format('site = "site-a"\n'))
    assert load_config(path, RunSettings).site == "site-a"
    path.write_text(body.format(""))
    assert load_config(path, RunSettings).site is None
    for bad in ("node1.example.org", "Site A", ""):
        path.write_text(body.format(f'site = "{bad}"\n'))
        with pytest.raises(Exception, match="site"):  # noqa: B017, PT011
            load_config(path, RunSettings)
