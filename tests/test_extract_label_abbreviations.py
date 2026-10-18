"""tools/extract-label-abbreviations.py on invented checkouts: which codes become rows, the
header's provenance, and the refusals. Every place, code and country spelling here is invented."""

from __future__ import annotations

import importlib.util
import json
import lzma
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from af.map.labels import LabelAbbreviations

TOOL = Path(__file__).resolve().parents[1] / "tools" / "extract-label-abbreviations.py"

COUNTRIES = """code\tspelling\tsource\tevidence\tadded_by\tadded_on
USA\tExampleland\tgisaid\tsame name\tt\t2026-01-01
XBB\tOtherland\tgisaid\tsame name\tt\t2026-01-01
"""
REGIONS = """scheme\tcountry\tgroup\tevidence\tadded_by\tadded_on
gisaid\tUSA\tExample Continent\tthe store\tt\t2026-01-01
gisaid\tXBB\tOther Continent\tthe store\tt\t2026-01-01
"""
PLACE_COLUMNS = ("location\tcountry\tadmin\tlatitude\tlongitude\tprecision\tsource\tevidence"
                 "\tsame_as\tflags\tadded_by\tadded_on")  # fmt: skip
PLACES = [("NORTH EXAMPLE", "USA"), ("SOUTH EXAMPLE", "USA"), ("FARTOWN", "XBB")]
CODES = {"NE": "NORTH EXAMPLE", "SE": "SOUTH EXAMPLE", "FT": "FARTOWN", "ZZ": "NOWHERE"}


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extract_label_abbreviations", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip


def checkouts(tmp_path: Path, codes: dict[str, str]) -> tuple[Path, Path]:
    db, data = tmp_path / "acmacs-data", tmp_path / "acmacs-f-data"
    db.mkdir()
    locationdb = json.dumps({"cdc_abbreviations": codes}).encode()
    (db / "locationdb.json.xz").write_bytes(lzma.compress(locationdb))
    locations = data / "rules/locations"
    locations.mkdir(parents=True)
    (locations / "countries.tsv").write_text(COUNTRIES)
    (locations / "regions.tsv").write_text(REGIONS)
    rows = [f"{loc}\t{c}\t\t1.0\t2.0\tcity\thand\tinvented\t\t\tt\t2026-01-01" for loc, c in PLACES]
    (locations / "places.tsv").write_text("\n".join([PLACE_COLUMNS, *rows]) + "\n")
    for repo in (db, data):
        git(repo, "init", "-q")
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "invented")
    return db, data


def test_a_code_becomes_a_row_only_for_a_place_in_the_usa(tool, tmp_path, capsys):
    db, data = checkouts(tmp_path, CODES)
    out = tmp_path / "label-abbreviations.tsv"
    assert tool.main([str(db), str(data), str(out)]) == 0
    assert capsys.readouterr().out.strip() == f"2 rows -> {out}"
    # FARTOWN has a code but is not in the USA; NOWHERE is no place at all: neither is a row
    assert LabelAbbreviations.read(out).table == {"NORTH EXAMPLE": "NE", "SOUTH EXAMPLE": "SE"}


def test_the_header_names_the_versions_it_was_made_from(tool, tmp_path):
    db, data = checkouts(tmp_path, CODES)
    out = tmp_path / "label-abbreviations.tsv"
    tool.main([str(db), str(data), str(out)])
    text = out.read_text()
    db_commit = git(db, "log", "-1", "--format=%h")
    assert f"locationdb.json.xz @ {db_commit} " in text
    assert f"places.tsv @ {git(data, 'log', '-1', '--format=%h')}\n# puts in USA" in text
    assert "(4 codes)" in text and "never edit by hand" in text
    # the same inputs give the same bytes
    again = tmp_path / "again.tsv"
    tool.main([str(db), str(data), str(again)])
    assert again.read_bytes() == out.read_bytes()


def test_a_place_with_two_codes_is_refused(tool, tmp_path):
    db, data = checkouts(tmp_path, {**CODES, "NX": "NORTH EXAMPLE"})
    with pytest.raises(SystemExit, match="'NORTH EXAMPLE' has two CDC codes"):
        tool.main([str(db), str(data), str(tmp_path / "out.tsv")])


def test_no_us_place_with_a_code_is_refused(tool, tmp_path):
    db, data = checkouts(tmp_path, {"FT": "FARTOWN"})
    with pytest.raises(SystemExit, match="no CDC code names a place"):
        tool.main([str(db), str(data), str(tmp_path / "out.tsv")])


def test_an_uncommitted_input_is_refused(tool, tmp_path):
    db, data = checkouts(tmp_path, CODES)
    fresh = tmp_path / "fresh"  # a checkout whose locationdb was never committed
    fresh.mkdir()
    (fresh / "locationdb.json.xz").write_bytes((db / "locationdb.json.xz").read_bytes())
    git(fresh, "init", "-q")
    git(fresh, "commit", "-q", "--allow-empty", "-m", "empty")
    with pytest.raises(SystemExit, match="locationdb.json.xz has no commit"):
        tool.main([str(fresh), str(data), str(tmp_path / "out.tsv")])
