"""ae's sequence hash and matching tree leaves by sequence (synthetic data only)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from af.report.compare import ae_hash, trees
from af.store import Provenance, Store

T0 = dt.datetime(2026, 10, 6, tzinfo=dt.UTC)


def test_xxh32_matches_the_published_vectors() -> None:
    # xxHash32, seed 0 (the reference implementation's test values)
    assert ae_hash.xxh32(b"") == 0x02CC5D05
    assert ae_hash.xxh32(b"a") == 0x550D7456
    assert ae_hash.xxh32(b"abc") == 0x32D153FF
    assert ae_hash.xxh32(b"Nobody inspects the spammish repetition") == 0xE2293B2F  # >= 16 bytes
    assert ae_hash.ae_hash("abc") == "32D153FF"  # as ae prints it: 8 upper-case hex digits


def _name(n: int) -> str:
    return "/".join(["SOMEPLACE", str(n), "2020"])


def _leaf(leaf_id: str, name: str, order: int) -> dict[str, Any]:
    return {"id": leaf_id, "name": name, "shown": True, "order": order, "clade": "X",
            "depth": float(order)}  # fmt: skip


def _doc(leaves: list[dict[str, Any]]) -> dict[str, Any]:
    return {"title": "t", "tree": {"leaves": leaves, "sections": [], "time_series": []},
            "provenance": {}}  # fmt: skip


def _epi(n: int) -> str:
    return f"EPI_ISL_{900000 + n}|EPI{900000 + n}"


def test_leaves_pair_by_sequence_then_by_isolate_name() -> None:
    seq = {i: ae_hash.ae_hash("ACGT" * (i + 5)) for i in range(6)}

    def ae(n: int, h: int) -> dict[str, Any]:  # ae writes the passage after the year
        return _leaf(f"{_name(n)}_E1/E1_{seq[h]}", f"{_name(n)}_E1/E1", n)

    def af(n: int) -> dict[str, Any]:  # af writes the type before the place
        return _leaf(_epi(n), "/".join(["A", _name(n)]), n)

    ref = _doc([ae(1, 1), ae(6, 1), ae(2, 2), ae(3, 3)])  # 1 and 6: one sequence, two isolates
    new = _doc([af(1), af(11), af(12), af(2), af(4), af(5)])
    hashes = {_epi(1): seq[1], _epi(11): seq[1], _epi(12): seq[1],  # three isolates of it here
              _epi(2): seq[0], _epi(4): seq[4]}  # 2's sequence differs; 5 has none  # fmt: skip
    res = trees.compare_figures(ref, new, new_hashes=hashes)
    m = res["matching"]
    assert m["how"] == "sequence, then name"
    # 1 pairs with 1 by name within the sequence, 6 with 11 by tree order; 2 by isolate name
    assert (m["by_sequence"], m["by_sequence_named"], m["by_name"]) == (2, 1, 1)
    assert m["new_without_sequence"] == 1
    assert res["common"] == 3 and res["only_ref_keys"] == [f"{_name(3)}_E1/E1"]
    assert len(res["only_new_keys"]) == 3  # 12 (a third isolate of 1's sequence), 4 and 5
    assert res["jaccard"] == pytest.approx(3 / 7)
    assert m["sequences"] == {"ref": 3, "new": 3, "common": 1, "jaccard": pytest.approx(1 / 5)}
    # by name alone the passage and type spellings keep every leaf apart
    by_name = trees.compare_figures(ref, new)
    assert by_name["matching"] == {"how": "name"} and by_name["common"] == 0


def test_isolate_key_drops_the_type_and_whatever_follows_the_year() -> None:
    assert trees.isolate_key(f"{_name(7)}_E1/E1") == trees.isolate_key("A/" + _name(7))
    assert trees.isolate_key("A(H1N1)/" + _name(7)) == trees.isolate_key(_name(7))


def test_a_figure_without_epi_ids_keeps_name_matching(tmp_path: Path) -> None:
    doc = _doc([_leaf("L1|x", _name(1), 0)])
    assert ae_hash.figure_hashes(None, doc) == (None, {})


def test_epi_keyed_leaves_need_the_store_and_hash_its_sequences(tmp_path: Path) -> None:
    doc = _doc([_leaf(_epi(1), _name(1), 0), _leaf(_epi(2), _name(2), 1)])
    with pytest.raises(ValueError, match="--store is needed"):
        ae_hash.figure_hashes(None, doc)
    store = Store.create(tmp_path / "store")
    with store.build("sequences", "h3") as build:
        part = build.path / "sequences" / "pull=test"
        part.mkdir(parents=True)
        epi, accession = _epi(1).split("|")
        table = {"epi_isl": [epi], "accession": [accession], "nuc_aligned": ["abc"]}
        pq.write_table(pa.table(table), part / "part-0.parquet")
        sequences = build.publish(Provenance("test", (), {}, T0, T0))
    with store.build("trees", "h3/test") as build:
        (build.path / "tree.json").write_text("{}")
        tree = build.publish(Provenance("test", (sequences,), {}, T0, T0))
    doc["provenance"]["store_refs"] = [tree.to_json()]
    hashes, source = ae_hash.figure_hashes(store, doc)
    assert hashes == {_epi(1): "32D153FF"}
    assert source == {"sequences_version": str(sequences), "leaves_without_sequence": 1}


def test_the_comparison_says_how_tree_leaves_were_paired(tmp_path: Path) -> None:
    from af.report.compare.run import Limits, markdown
    from af.util.config import parse_config

    limits = parse_config(
        {"adoption": {"status": "provisional", "adopted_by": "a reviewer",
                      "adopted": dt.date(2026, 9, 25), "review": "later"}},
        Limits, base_dir=tmp_path,
    )  # fmt: skip
    detail = {"only_ref_keys": [], "only_new_keys": [], "sections": {"unresolved": {"ref": [],
              "new": []}, "only_ref": [], "only_new": []},
              "matching": {"how": "sequence, then name", "by_sequence": 7, "by_sequence_named": 2,
                           "by_name": 3, "new_without_sequence": 5,
                           "sequences_version": "sequences/x@v",
                           "sequences": {"ref": 9, "new": 8, "common": 6,
                                         "jaccard": 0.5454}}}  # fmt: skip
    rows = [{"slot": "tree/x/report", "status": "ok", "tree_checks": [], "detail": detail}]
    text = markdown({"report": "r", "built": "b"}, rows, "l.toml", "identity", limits.adoption)
    assert ("tree/x/report: leaves paired by ae sequence hash 7 (2 of them by isolate name within "
            "a sequence), by isolate name 3; af leaves with no sequence 5 (sequences/x@v). "
            "Distinct sequences drawn: ref 9, af 8, both 6 (jaccard 0.545)") in text  # fmt: skip
