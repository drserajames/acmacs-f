"""Tree selection rules. Every name, id, date and sequence here is invented."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from af.seq import processed
from af.seq import select as S
from af.seq.dates import parse as parse_date
from af.seq.gisaid import SequenceRecord
from af.seq.nextclade import Aligned
from af.store import Store
from af.store.ref import StoreRef

from .test_store_build import aligned, record

OUTGROUP = S.Outgroup("EPI_ISL_1", "EPI1", "invented outgroup")


def publish(
    store: Store,
    pull: str,
    records: list[SequenceRecord],
    results: dict[str, Aligned] | None = None,
) -> StoreRef:
    results = results or {}
    found = {processed.seq_id(r): results.get(r.epi_isl) or aligned(r) for r in records}
    return processed.publish_pull(
        store, "h3", pull, processed.isolates_table(records, "h3", pull),
        processed.sequences_table(records, found, pull), inputs=[], parameters={},
        started=datetime.datetime.now(datetime.UTC),
    )  # fmt: skip


@pytest.fixture
def store(tmp_path: Path) -> Store:
    st = Store.create(tmp_path / "store")
    recs = [
        record(1, accession="EPI1", collection_date=parse_date("2009-05-01")),  # the outgroup
        record(2, accession="EPI2", collection_date=parse_date("2021-03-05")),
        record(3, accession="EPI3", collection_date=parse_date("2021")),  # year only
        record(4, accession="EPI4", collection_date=parse_date("2021-06-01"), host="Swine"),
        record(5, accession="EPI5", collection_date=parse_date("2021-02-01")),
    ]
    publish(st, "p1", recs, {"EPI_ISL_5": aligned(recs[4], covers_mature=False)})
    return st


def rules(*items: S.Rule) -> S.SubtypeRules:
    return S.SubtypeRules("h3", OUTGROUP, list(items))


HOST = S.Rule("host", "host", "human only", allow=["Human"])
QC = S.Rule("qc", "qc", "R3", max_unknown_aa=10, max_deleted_aa=6)


def listing(path: Path, kind: str, *keys: tuple[str, str]) -> Path:
    path.write_text("# invented\nepi_isl\taccession\tlist\treason\n"
                    + "".join(f"{e}\t{a}\t{kind}\twhy\n" for e, a in keys))  # fmt: skip
    return path


def test_rules_apply_in_order_and_are_counted(store: Store) -> None:
    sel = S.select(store, rules(HOST, QC))
    assert sel.keys == [("EPI_ISL_1", "EPI1"), ("EPI_ISL_3", "EPI3"), ("EPI_ISL_2", "EPI2")]
    assert [(c.rule, c.removed, c.remaining) for c in sel.counts] == [
        ("host", 1, 3),
        ("qc", 1, 2),
    ]  # the outgroup is not counted
    assert sel.total == 5


def test_a_date_floor_needs_the_whole_interval(store: Store) -> None:
    floor = S.Rule("floor", "date_floor", "R4", floor="2021-03-01")
    sel = S.select(store, rules(floor))
    assert ("EPI_ISL_3", "EPI3") not in sel.keys  # 2021 might be January
    assert ("EPI_ISL_5", "EPI5") not in sel.keys
    assert sel.keys[0] == OUTGROUP.key  # older than the floor, exempt


def test_include_adds_back_but_qc_and_exclude_still_apply(store: Store, tmp_path: Path) -> None:
    inc = listing(tmp_path / "inc.tsv", "include", ("EPI_ISL_4", "EPI4"), ("EPI_ISL_5", "EPI5"))
    exc = listing(tmp_path / "exc.tsv", "exclude", ("EPI_ISL_4", "EPI4"))
    sel = S.select(store, rules(
        HOST,
        S.Rule("inc", "include_list", "named", file=inc),
        QC,
        S.Rule("exc", "exclude_list", "named", file=exc),
    ))  # fmt: skip
    assert [(c.rule, c.removed, c.added) for c in sel.counts] == [
        ("host", 1, 0), ("inc", 0, 1), ("qc", 1, 0), ("exc", 1, 0),
    ]  # fmt: skip
    assert ("EPI_ISL_4", "EPI4") not in sel.keys  # exclude beats include


def test_a_rule_that_changes_nothing_is_an_error_unless_optional(store: Store) -> None:
    never = S.Rule("floor", "date_floor", "R4", floor="2000-01-01")
    with pytest.raises(S.SelectionError, match="changed nothing.*floor"):
        S.select(store, rules(never))
    S.select(store, rules(S.Rule("floor", "date_floor", "R4", floor="2000-01-01", optional=True)))


def test_a_missing_outgroup_is_fatal(store: Store) -> None:
    config = S.SubtypeRules("h3", S.Outgroup("EPI_ISL_9", "EPI9", "x"), [])
    with pytest.raises(S.SelectionError, match="cannot be rooted"):
        S.select(store, config)


def test_a_rule_without_a_reason_is_refused(store: Store) -> None:
    with pytest.raises(S.SelectionError, match="no reason"):
        S.select(store, rules(S.Rule("host", "host", " ", allow=["Human"])))


def test_aa_deletion_at_a_position(tmp_path: Path) -> None:
    st = Store.create(tmp_path / "s")
    recs = [record(1, accession="EPI1"), record(2, accession="EPI2")]
    publish(st, "p", recs, {"EPI_ISL_2": aligned(recs[1], amino_acids="T-")})
    sel = S.select(st, rules(S.Rule("del2", "aa_deletion", "R5", position=2)))
    assert sel.keys == [("EPI_ISL_1", "EPI1")]


@dataclass(frozen=True)
class Pin:
    below: frozenset[tuple[str, str]]
    chosen_from: StoreRef


def test_cut_keeps_below_and_what_the_cut_never_saw(store: Store) -> None:
    chosen = store.current("sequences", "h3")
    publish(store, "p2", [record(6, accession="EPI6", collection_date=parse_date("2020-01-01"))])
    pin = Pin(frozenset({("EPI_ISL_2", "EPI2")}), chosen)
    sel = S.select(store, rules(S.Rule("cut", "cut", "R4''")), cut=pin)
    # Known at the cut and not below it: 3, 4, 5 go. 6 arrived later (even with an old date): kept.
    assert sorted(sel.keys) == [("EPI_ISL_1", "EPI1"), ("EPI_ISL_2", "EPI2"), ("EPI_ISL_6", "EPI6")]
    assert sel.counts[0].removed == 3


def test_a_cut_rule_without_a_pin_is_refused(store: Store) -> None:
    with pytest.raises(S.SelectionError, match="needs the cycle's cut pin"):
        S.select(store, rules(S.Rule("cut", "cut", "R4''")))


def test_order_is_outgroup_then_date_then_key(store: Store) -> None:
    sel = S.select(store, rules(S.Rule("f", "date_floor", "x", floor="2000-01-01", optional=True)))
    assert sel.keys == [OUTGROUP.key, ("EPI_ISL_3", "EPI3"), ("EPI_ISL_5", "EPI5"),
                        ("EPI_ISL_2", "EPI2"), ("EPI_ISL_4", "EPI4")]  # fmt: skip


def overrides(path: Path, *rows: tuple[str, str, str]) -> Path:
    path.write_text("# invented\nepi_isl\taccession\tgisaid_host\treason\n"
                    + "".join(f"{e}\t{a}\tSwine\t{why}\n" for e, a, why in rows))  # fmt: skip
    return path


def test_a_host_override_lets_a_mislabelled_record_through(store: Store, tmp_path: Path) -> None:
    ovr = overrides(tmp_path / "ovr.tsv", ("EPI_ISL_4", "EPI4", "name and lab say human"))
    host = S.Rule("host", "host", "human only", allow=["Human"], file=ovr)
    sel = S.select(store, rules(host))
    assert ("EPI_ISL_4", "EPI4") in sel.keys
    assert [(c.rule, c.removed, c.added, c.remaining) for c in sel.counts] == [("host", 0, 1, 4)]
    assert sel.counts[0].source == str(ovr)


def test_a_host_override_that_changes_nothing_is_an_error_unless_optional(
    store: Store, tmp_path: Path
) -> None:
    ovr = overrides(tmp_path / "ovr.tsv",
                    ("EPI_ISL_4", "EPI4", "mislabelled"),
                    ("EPI_ISL_2", "EPI2", "already Human"),
                    ("EPI_ISL_8", "EPI8", "not in the store yet"))  # fmt: skip
    host = S.Rule("host", "host", "human only", allow=["Human"], file=ovr)
    with pytest.raises(S.SelectionError, match=r"change nothing.*EPI_ISL_2.*EPI_ISL_8"):
        S.select(store, rules(host))
    sel = S.select(store, rules(S.Rule("host", "host", "human only", allow=["Human"],
                                       file=ovr, optional=True)))  # fmt: skip
    assert [(c.removed, c.added) for c in sel.counts] == [(0, 1)]


def test_every_host_override_needs_a_reason(store: Store, tmp_path: Path) -> None:
    ovr = overrides(tmp_path / "ovr.tsv", ("EPI_ISL_4", "EPI4", " "))
    with pytest.raises(S.SelectionError, match="without a reason"):
        S.select(store, rules(S.Rule("host", "host", "h", allow=["Human"], file=ovr)))


def test_a_missing_host_override_file_is_an_error(store: Store, tmp_path: Path) -> None:
    host = S.Rule("host", "host", "h", allow=["Human"], file=tmp_path / "absent.tsv")
    with pytest.raises(S.SelectionError, match="override file"):
        S.select(store, rules(host))


def without_partial(store: Store, pull: str) -> None:
    """Rewrite one pull's sequences partition as a build from before aa_partial would have."""
    path = store.resolve(store.current("sequences", "h3")) / "sequences" / f"pull={pull}"
    part = path / processed.PART
    table = pq.read_table(part).drop_columns(["aa_partial"])
    part.unlink()  # read-only, and a hard link shared with the earlier version
    pq.write_table(table, part)


@pytest.mark.parametrize(
    ("stripped", "partial"),
    [
        ((), True),  # every pull re-stored since the column was added
        (("p2",), True),  # mid-sweep: p1 re-stored, p2 not yet
        (("p1", "p2"), False),  # nothing re-stored: no build then could store a partial protein
    ],
)
def test_select_reads_partitions_with_and_without_aa_partial(
    tmp_path: Path, stripped: tuple[str, ...], partial: bool
) -> None:
    st = Store.create(tmp_path / "s")
    p1 = [record(1, accession="EPI1"), record(2, accession="EPI2")]
    publish(st, "p1", p1, {"EPI_ISL_2": aligned(p1[1], amino_acids="TX", aa_partial=True)})
    publish(st, "p2", [record(3, accession="EPI3")])
    for pull in stripped:
        without_partial(st, pull)
    sel = S.select(st, rules(S.Rule("f", "date_floor", "x", floor="2000-01-01", optional=True)))
    assert sorted(sel.keys) == [("EPI_ISL_1", "EPI1"), ("EPI_ISL_2", "EPI2"), ("EPI_ISL_3", "EPI3")]
    loaded = S._load(st, st.current("sequences", "h3"))
    assert {k: r.aligned.aa_partial for k, r in loaded.items()} == {
        ("EPI_ISL_1", "EPI1"): False,
        ("EPI_ISL_2", "EPI2"): partial,
        ("EPI_ISL_3", "EPI3"): False,
    }
