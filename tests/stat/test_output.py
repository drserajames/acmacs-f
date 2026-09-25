import datetime
import json
from pathlib import Path

import pytest

from af.geo.records import Month
from af.serology.query import Preparation, SerumUse
from af.stat.counts import StatCounts, stat_counts
from af.stat.output import FORMAT, StatOutputError, read_previous, write_stat

d = datetime.date


def _counts() -> StatCounts:
    preps = [
        Preparation("B", "VICTORIA", "Alpha-1", "", (), "E3", d(2021, 1, 9), "LABX", d(2021, 2, 1)),
        Preparation("B", "VICTORIA", "Beta-2", "", (), "E3", None, "LABY", d(2021, 2, 1)),
    ]
    uses = [SerumUse("B", "VICTORIA", "k1", "Alpha-1", "t1", "LABY", d(2021, 2, 1), 1)]
    continents = {"Alpha": "CONTINENT-1"}
    return stat_counts(
        preps,
        uses,
        Month(2021, 1),
        Month(2021, 2),
        lambda name: name.split("-")[0],
        continents.get,
        split_by_lineage={"B"},
    )


def test_write_stat_data_and_page(tmp_path: Path) -> None:
    paths = write_stat(_counts(), Month(2021, 1), Month(2021, 2), tmp_path)
    assert [p.name for p in paths] == ["stat.json", "index.html"]
    doc = json.loads(paths[0].read_text())
    assert doc["format"] == FORMAT
    assert doc["window"] == {"first": "2021-01", "last": "2021-02"}
    assert doc["undated"] == {"antigens": 1}
    cell = {"measure": "antigens", "subtype": "B", "lab": "LABX", "period": "2021-01"}
    assert {**cell, "continent": "CONTINENT-1", "count": 1} in doc["cells"]
    assert all(c["count"] > 0 for c in doc["cells"])
    page = paths[1].read_text()
    # every subtype section, every lab found in the data, the notes, and no scripts
    for text in (
        "<h2>all</h2>",
        "<h2>B</h2>",
        "<h2>B/VICTORIA</h2>",
        "LABX",
        "LABY",
        "no collection date",
    ):
        assert text in page
    assert "<script" not in page


def test_page_escapes_names(tmp_path: Path) -> None:
    preps = [Preparation("B", "", "A-1", "", (), "E3", d(2021, 1, 9), "<b>LAB</b>", d(2021, 1, 9))]
    counts = stat_counts(preps, [], Month(2021, 1), Month(2021, 1), lambda n: "A", {"A": "C"}.get)
    page = write_stat(counts, Month(2021, 1), Month(2021, 1), tmp_path)[1].read_text()
    assert "<b>LAB</b>" not in page and "&lt;b&gt;LAB&lt;/b&gt;" in page


def _prep(name: str, lab: str, day: datetime.date) -> Preparation:
    return Preparation("B", "VICTORIA", name, "", (), "E3", day, lab, day)


def test_changes_since_previous_round(tmp_path: Path) -> None:
    location, continents = (lambda n: n.split("-")[0]), {"Alpha": "CONTINENT-1"}.get
    # previous round: window Nov..Jan; one LABX antigen in Jan, two LABY antigens in Jan
    before = stat_counts(
        [
            _prep("Alpha-1", "LABX", d(2021, 1, 9)),
            _prep("Alpha-2", "LABY", d(2021, 1, 9)),
            _prep("Alpha-3", "LABY", d(2021, 1, 10)),
        ],
        [],
        Month(2020, 11),
        Month(2021, 1),
        location,
        continents,
    )
    old = write_stat(before, Month(2020, 11), Month(2021, 1), tmp_path / "old")[0]
    # this round: window Jan..Feb; a late LABX antigen in Jan, one LABY antigen withdrawn
    now = stat_counts(
        [
            _prep("Alpha-1", "LABX", d(2021, 1, 9)),
            _prep("Alpha-4", "LABX", d(2021, 1, 20)),
            _prep("Alpha-2", "LABY", d(2021, 1, 9)),
            _prep("Alpha-5", "LABX", d(2021, 2, 2)),
        ],
        [],
        Month(2021, 1),
        Month(2021, 2),
        location,
        continents,
    )
    previous = read_previous(old)
    paths = write_stat(now, Month(2021, 1), Month(2021, 2), tmp_path / "new", previous)
    doc = json.loads(paths[0].read_text())
    found = {
        (c["lab"], c["continent"]): (c["now"], c["before"])
        for c in doc["previous"]["changes"]
        if c["measure"] == "antigens" and c["subtype"] == "B" and c["period"] == "2021-01"
    }
    assert found[("LABX", "all")] == (2, 1)
    assert found[("LABY", "all")] == (1, 2)  # a decrease is reported, not clamped to zero
    assert found.get(("all", "all")) is None  # 3 before, 3 now: no change
    # February and the totals are not in the previous window: never compared
    assert not any(c["period"] in {"2021-02", "2021", "all"} for c in doc["previous"]["changes"])
    page = paths[1].read_text()
    assert "<small>(+1)</small>" in page and "class='down'>1 <small>(-1)</small>" in page


def test_previous_of_another_format_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "stat.json"
    other.write_text(json.dumps({"antigens": {}}))
    with pytest.raises(StatOutputError, match="format"):
        read_previous(other)
    with pytest.raises(StatOutputError, match="cannot read"):
        read_previous(tmp_path / "missing.json")
