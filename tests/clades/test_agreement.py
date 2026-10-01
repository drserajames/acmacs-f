"""The agreement guard: a tree whose calls disagree with the fallback's is not published."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from af.clades.agreement import (
    AgreementError,
    AgreementLimit,
    check_agreement,
    limit_for,
    load_agreement_limits,
)
from af.clades.assign import Assignment
from af.clades.from_tree import publish_clades
from af.clades.nomenclature import CladeSet
from af.clades.store import CladeStoreError, read_report
from af.store import Store, StoreRef

from .synthetic import build_clone, load_synthetic
from .test_fallback_store import standard
from .test_from_tree import (
    CONFLICTS,
    STARTED,
    SUBTYPE,
    clade_set,
    nomenclature_input,
    tree_version,
)

HEADER = "subtype\tmax_disagreement\treason\n"


def limits_file(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    path = tmp_path / "agreement.tsv"
    path.write_text(header + "".join(row + "\n" for row in rows))
    return path


def synthetic(tmp_path: Path) -> CladeSet:
    return load_synthetic(build_clone(tmp_path).parent)


def calls(**clades: str | None) -> dict[str, Assignment]:
    return {leaf: Assignment(leaf, clade) for leaf, clade in clades.items()}


# ------------------------------------------------------------------------- the limits file


def test_reads_a_limit_per_subtype_with_its_reason(tmp_path: Path) -> None:
    path = limits_file(tmp_path, f"{SUBTYPE}\t0.02\tcalibrated on four trees", "B/Vic\t0.05\tx")
    limits = load_agreement_limits(path)
    assert limit_for(limits, SUBTYPE) == AgreementLimit(SUBTYPE, 0.02, "calibrated on four trees")
    assert limits["B/Vic"].max_disagreement == 0.05


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (f"{SUBTYPE}\t0.02\t", "has no reason"),
        (f"{SUBTYPE}\tsmall\tx", "is not a number"),
        (f"{SUBTYPE}\t1.5\tx", "is not a fraction"),
        (f"{SUBTYPE}\t-0.1\tx", "is not a fraction"),
        ("\t0.02\tx", "no subtype"),
    ],
)
def test_a_bad_row_is_refused(tmp_path: Path, row: str, message: str) -> None:
    with pytest.raises(AgreementError, match=message):
        load_agreement_limits(limits_file(tmp_path, row))


def test_a_subtype_listed_twice_is_refused(tmp_path: Path) -> None:
    path = limits_file(tmp_path, f"{SUBTYPE}\t0.02\tx", f"{SUBTYPE}\t0.03\ty")
    with pytest.raises(AgreementError, match="listed twice"):
        load_agreement_limits(path)


def test_wrong_columns_a_missing_file_and_no_rows_are_refused(tmp_path: Path) -> None:
    with pytest.raises(AgreementError, match="columns must be"):
        load_agreement_limits(limits_file(tmp_path, "x\t1", header="subtype\tlimit\n"))
    with pytest.raises(AgreementError, match="not found"):
        load_agreement_limits(tmp_path / "absent.tsv")
    with pytest.raises(AgreementError, match="no limits"):
        load_agreement_limits(limits_file(tmp_path))


def test_a_subtype_without_a_limit_is_an_error_not_no_limit(tmp_path: Path) -> None:
    limits = load_agreement_limits(limits_file(tmp_path, f"{SUBTYPE}\t0.02\tx"))
    with pytest.raises(AgreementError, match="no agreement limit for 'B/Vic'"):
        limit_for(limits, "B/Vic")


# ------------------------------------------------------------------------- the comparison


def test_deeper_is_agreement_shallower_and_different_are_not(tmp_path: Path) -> None:
    """Being more specific is the tree engine's job; less specific or elsewhere is not."""
    clade_set = synthetic(tmp_path)
    limit = AgreementLimit(SUBTYPE, 0.9, "x")
    tree = calls(a="P.1.1", b="P", c="P.2", d="P.1")
    fallback = calls(a="P.1", b="P.1", c="P.1", d="P.1")
    check = check_agreement(tree, fallback, clade_set, limit)
    assert (check.compared, check.disagree) == (4, 2)
    assert set(check.transitions) == {("P.1", "P", 1), ("P.1", "P.2", 1)}


def test_only_leaves_both_engines_call_are_compared(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    check = check_agreement(
        calls(a="P.1", b="P.2"), calls(a="P.1"), clade_set, AgreementLimit(SUBTYPE, 0.0, "x")
    )
    assert (check.compared, check.disagree, check.exceeded) == (1, 0, False)


def test_over_the_limit_is_refused_naming_the_commonest_changes(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    tree = calls(a="P.2", b="P.2", c="P.1", d="P.1")
    fallback = calls(a="P.1", b="P.1", c="P.1", d="P.1")
    with pytest.raises(AgreementError, match=r"2 of 4 leaves \(50\.00%\).*P\.1 -> P\.2 2"):
        check_agreement(tree, fallback, clade_set, AgreementLimit(SUBTYPE, 0.02, "the limit"))


def test_a_named_reason_publishes_over_the_limit_and_is_kept(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    check = check_agreement(
        calls(a="P.2"),
        calls(a="P.1"),
        clade_set,
        AgreementLimit(SUBTYPE, 0.02, "x"),
        accept_disagreement="dataset lags the pin",
    )
    assert check.exceeded and check.accepted == "dataset lags the pin"
    assert check.provenance() == {
        "max_disagreement": 0.02,
        "reason": "x",
        "accepted": "dataset lags the pin",
    }


def test_an_empty_reason_and_nothing_to_compare_are_refused(tmp_path: Path) -> None:
    clade_set = synthetic(tmp_path)
    limit = AgreementLimit(SUBTYPE, 0.02, "x")
    with pytest.raises(AgreementError, match="needs a reason"):
        check_agreement(calls(a="P.2"), calls(a="P.1"), clade_set, limit, accept_disagreement=" ")
    with pytest.raises(AgreementError, match="no tree leaf has a fallback call"):
        check_agreement(calls(a="P.2"), calls(b="P.1"), clade_set, limit)


# ------------------------------------------------------------------------- publishing


def publish(tmp_path: Path, store: Store, **kwargs: Any) -> StoreRef:
    clades = clade_set(tmp_path)
    sequences, _ = standard(store)
    tree = tree_version(store, clades.version, sequences=sequences)
    return publish_clades(
        store,
        SUBTYPE,
        clades,
        tree=tree,
        sequences=sequences,
        nomenclature=[nomenclature_input(tmp_path)],
        started=STARTED,
        conflicts=CONFLICTS,
        **kwargs,
    )


def test_a_tree_with_a_fallback_needs_a_limit(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    with pytest.raises(CladeStoreError, match="needs an agreement limit"):
        publish(tmp_path, store)


def test_a_tree_over_the_limit_is_not_published(tmp_path: Path) -> None:
    """The synthetic tree disagrees on 1 of 5 leaves (20%)."""
    store = Store.create(tmp_path / "store")
    with pytest.raises(AgreementError, match=r"1 of 5 leaves \(20\.00%\)"):
        publish(tmp_path, store, agreement=AgreementLimit(SUBTYPE, 0.02, "the limit"))
    with pytest.raises(Exception, match="clades"):
        store.current("clades", "h3")


def test_an_accepted_disagreement_is_in_the_report_and_the_provenance(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    ref = publish(
        tmp_path,
        store,
        agreement=AgreementLimit(SUBTYPE, 0.02, "the limit"),
        accept_disagreement="known and understood",
    )
    report = read_report(store, ref)
    assert report["agreement"]["fraction"] == 0.2
    assert report["agreement"]["exceeded"] is True
    assert report["agreement"]["accepted"] == "known and understood"
    provenance = json.loads((store.resolve(ref) / "PROVENANCE.json").read_text())
    assert provenance["parameters"]["agreement"] == {
        "max_disagreement": 0.02,
        "reason": "the limit",
        "accepted": "known and understood",
    }


def test_an_override_without_a_comparison_is_refused(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    clades = clade_set(tmp_path)
    sequences, _ = standard(store)
    with pytest.raises(CladeStoreError, match="no tree and fallback to compare"):
        publish_clades(
            store,
            SUBTYPE,
            clades,
            tree=None,
            sequences=sequences,
            nomenclature=[nomenclature_input(tmp_path)],
            started=STARTED,
            accept_disagreement="anything",
        )
