"""`.ace` round trip and compression detection."""

import gzip
import json
import lzma
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from af.chart.ace import (
    AceError,
    chart_json_bytes,
    chart_to_json,
    content_hash,
    read_chart,
    write_chart,
)
from af.chart.model import Antigen, Chart, Projection, Serum, Titres, empty_table
from af.chart.titre import Titre


def small_chart() -> Chart:
    table = empty_table(3, 2)
    for (i, j), t in {(0, 0): "40", (0, 1): "<10", (1, 0): "160", (2, 1): ">1280"}.items():
        table[i][j] = Titre.parse(t)
    return Chart(
        info={"V": "A(H9N2)", "A": "HI", "l": "TEST", "D": "20210115"},
        antigens=[
            Antigen("TEST-1", passage="E3", extra={"T": {"R": True}}),
            Antigen("TEST-2", passage="MDCK1"),
            Antigen("TEST-3"),
        ],
        sera=[
            Serum("TEST-1", serum_id="F001"),
            Serum("TEST-5", serum_id="F002"),
        ],
        titres=Titres(table),
        projections=[
            Projection(
                layout=np.array([[0.0, 1.0], [np.nan, np.nan], [2.0, 3.0], [1.0, 1.0], [0.5, 0.5]]),
                stress=1.5,
                disconnected=(1,),
            )
        ],
        extra={"R": {"style": {}}},
    )


def test_round_trip(tmp_path):
    path = write_chart(small_chart(), tmp_path / "c.ace")
    assert path.read_bytes()[:6] == b"\xfd7zXZ\x00"
    back = read_chart(path)
    assert chart_to_json(back) == chart_to_json(small_chart())
    assert np.isnan(back.projections[0].layout[1]).all()
    assert back.antigens[0].extra == {"T": {"R": True}}


@pytest.mark.parametrize("compress", [lambda b: b, gzip.compress, lzma.compress])
def test_compression_detected_by_magic(tmp_path, compress):
    data = json.dumps(chart_to_json(small_chart())).encode()
    (tmp_path / "x.ace").write_bytes(compress(data))
    assert read_chart(tmp_path / "x.ace").n_antigens == 3


def test_content_hash_ignores_compression(tmp_path):
    data = json.dumps(chart_to_json(small_chart())).encode()
    (tmp_path / "a.ace").write_bytes(data)
    (tmp_path / "b.ace").write_bytes(lzma.compress(data))
    assert content_hash(tmp_path / "a.ace") == content_hash(tmp_path / "b.ace")


def test_layout_size_mismatch_is_an_error(tmp_path):
    doc = chart_to_json(small_chart())
    doc["c"]["P"][0]["l"].pop()
    (tmp_path / "bad.ace").write_text(json.dumps(doc))
    with pytest.raises(AceError):
        read_chart(tmp_path / "bad.ace")


def test_optimiser_arrays_i1():
    a = small_chart().optimiser_arrays()
    assert a["titre_type"].tolist() == [[1, 2], [1, 0], [0, 3]]
    assert np.isnan(a["titre_value"][1, 1])
    assert a["column_bases"].tolist() == pytest.approx([4.0, 8.0])  # >1280 counts one dilution up
    assert a["disconnected"].all()  # every point has fewer than 3 regular titres


def test_concurrent_writers_of_one_path_all_succeed(tmp_path, monkeypatch):
    """Several processes do write one chart path: array tasks loading one dataset into a shared
    inputs directory. Renames are held here until every writer has its temporary on disk, the
    interleaving a temporary named after the target alone could not survive."""
    path = tmp_path / "c.ace"
    writers = 6
    at_rename = threading.Barrier(writers, timeout=60)
    real_replace = Path.replace

    def replace(self, target):
        if Path(target) == path:
            at_rename.wait()
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    with ThreadPoolExecutor(max_workers=writers) as pool:
        futures = [pool.submit(write_chart, small_chart(), path) for _ in range(writers)]
        assert [f.result() for f in futures] == [path] * writers
    assert read_chart(path).n_antigens == 3
    assert [p.name for p in tmp_path.iterdir()] == ["c.ace"]  # no temporary left behind


def test_a_failed_rename_leaves_no_temporary(tmp_path, monkeypatch):
    """A unique temporary name is never reused, so the write has to clean up after itself."""
    path = tmp_path / "c.ace"
    real_replace = Path.replace

    def replace(self, target):
        if Path(target) == path:
            raise OSError("no space left on device")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="no space left"):
        write_chart(small_chart(), path)
    assert list(tmp_path.iterdir()) == []


def test_chart_json_bytes_are_what_is_written(tmp_path):
    chart = small_chart()
    assert lzma.decompress(write_chart(chart, tmp_path / "c.ace").read_bytes()) == chart_json_bytes(
        chart
    )
