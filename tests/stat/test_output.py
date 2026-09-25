import datetime
import json
from pathlib import Path

from af.geo.records import Month
from af.serology.query import Preparation, SerumUse
from af.stat.counts import StatCounts, stat_counts
from af.stat.output import FORMAT, write_stat

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
