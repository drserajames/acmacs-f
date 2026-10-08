"""Vaccine rule dates derived from WHO's recommendations or git, else "not known" (synthetic)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from af.report import vaccine_dates as vd


def _virus(place: str, number: str, year: str, kind: str = "A") -> str:
    return "/".join([kind, place, number, year])


def _who(tmp: Path, seasons: list[tuple[str, str, list[str]]]) -> Path:
    doc = {"seasons": [
        {"season_id": sid, "recommendation_date": date,
         "components": [{"vaccine_virus": v, "platform": "egg"} for v in viruses]}
        for sid, date, viruses in seasons
    ]}  # fmt: skip
    path = tmp / "who.json"
    path.write_text(json.dumps(doc))
    return path


def test_names_compare_without_type_case_or_padding() -> None:
    assert vd.normalise(_virus("Someplace", "02", "2018")) == "/".join(["SOMEPLACE", "2", "2018"])
    assert vd.normalise(_virus("Someplace", "02", "2018")) == vd.normalise(
        "/".join(["SOMEPLACE", "2", "2018"])
    )


def test_superseded_is_the_first_recommendation_after_the_last_naming_it(tmp_path: Path) -> None:
    old, new = _virus("Someplace", "1", "2015"), _virus("Otherplace", "7", "2018")
    who = vd.load_who(_who(tmp_path, [
        ("2019-SH", "2018-09-27", [old]), ("2019-2020-NH", "2019-02-21", [old]),
        ("2020-SH", "2019-09-27", [new]), ("2020-2021-NH", "2020-02-28", [new]),
    ]))  # fmt: skip
    date, source = vd.superseded(who, "/".join(["SOMEPLACE", "1", "2015"])) or ("", "")
    assert date == "2019-09-27" and "last recommended in 2019-2020-NH" in source
    assert vd.superseded(who, "/".join(["OTHERPLACE", "7", "2018"])) is None  # still recommended
    assert vd.superseded(who, "/".join(["NOWHERE", "9", "2010"])) is None  # never a WHO vaccine


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_precedence_hand_then_who_then_git_then_not_known(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = repo / "maps.toml"
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    local = "/".join(["LOCALONLY", "3", "2019"])
    cfg.write_text(f'name = "{local}"\n')
    subprocess.run([*git, "add", "maps.toml"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "rule", "--date=2026-09-25T12:00:00"],
                   check=True, env={"GIT_COMMITTER_DATE": "2026-09-25T12:00:00",
                                    "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})  # fmt: skip
    old = _virus("Someplace", "1", "2015")
    who = vd.load_who(_who(tmp_path, [("2019-SH", "2018-09-27", [old]),
                                      ("2020-SH", "2019-09-27", [])]))  # fmt: skip
    dates = vd.VaccineDates(who, {"map": cfg})
    whole = {
        "rule": "disable",
        "passage": "any",
        "scope": "map",
        "name": "/".join(["SOMEPLACE", "1", "2015"]),
    }
    assert dates.resolve(whole)[0] == "2019-09-27"
    hand = dates.resolve({**whole, "decided": "2026-10-01"})
    assert hand == ("2026-10-01", "config decided, overriding 2019-09-27 (WHO: last recommended "
                                  "in 2019-SH (2018-09-27); 2020-SH (2019-09-27) is the first "
                                  "recommendation after it)")  # fmt: skip
    one_prep = {"rule": "disable", "passage": "reassortant", "scope": "map",
                "name": local}  # fmt: skip
    date, source = dates.resolve(one_prep)  # not a whole vaccine: WHO says nothing; git does
    assert date == "2026-09-25" and source.startswith("git: commit ")
    assert dates.resolve({**one_prep, "name": "/".join(["UNSEEN", "4", "2020"])}) == (
        "not known",
        "neither WHO's recommendations nor git history give a date",
    )
