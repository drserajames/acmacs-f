"""The retrospective legacy label: only for viruses the subclades do not name (task 4.10)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from af.clades.colours import ColourEntry, ColourSchemeError, scheme_from_rows, shadowed_entries
from af.clades.from_tree import publish_clades
from af.clades.legacy import (
    LegacyError,
    LegacyRule,
    legacy_label,
    legacy_labels,
    load_legacy_rules,
    rule_for,
)
from af.clades.nomenclature import CladeSet
from af.clades.sequence import AlignedSequence
from af.clades.store import ASSIGNMENTS_FILE, CladeStoreError, read_report
from af.store import ExternalInput, Provenance, Store, StoreRef

from .synthetic import add_legacy_clades, build_clone, load_synthetic
from .test_fallback_store import raw_dataset

SUBTYPE = "A(H3N2)"
STARTED = datetime.datetime(2026, 9, 30, tzinfo=datetime.UTC)
CODON = {"A": "GCT", "V": "GTT", "W": "TGG", "K": "AAA", "R": "CGT", "S": "TCT"}
HEADER = "subtype\tmax_ancestor_contradictions\ttolerate_root\treason\n"


def legacy_set(tmp_path: Path) -> CladeSet:
    add_legacy_clades(build_clone(tmp_path))
    return load_synthetic(tmp_path)


def nucleotides(**residues: str) -> str:
    """A mature HA of alanines, with ``p<position>=<residue>`` changed."""
    aa = ["A"] * 566
    for key, residue in residues.items():
        aa[int(key[1:]) - 1] = residue
    return "".join(CODON[a] for a in aa)


def virus(**residues: str) -> AlignedSequence:
    return AlignedSequence.from_nucleotides(nucleotides(**residues))


def rule(n: int = 1, root: bool = True) -> LegacyRule:
    return LegacyRule(SUBTYPE, n, root, "test")


# ------------------------------------------------------------------ the legacy hierarchy


def test_only_self_defining_legacy_clades_can_label(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    assert clade_set.legacy_defining() == ("L", "L.1", "L.1.1", "L.2", "L.3")  # not the pointer Q
    assert clade_set.legacy_ancestors("L.1.1") == ("L.1", "L")
    assert clade_set.legacy_parent("L") is None
    assert clade_set.legacy_is_within("L.1.1", "L") and not clade_set.legacy_is_within("L.2", "L.1")
    assert clade_set.legacy_common_ancestor(["L.1.1", "L.2"]) == "L"


# ------------------------------------------------------------------ the rules file


def rules_file(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    path = tmp_path / "legacy.tsv"
    path.write_text(header + "".join(row + "\n" for row in rows))
    return path


def test_reads_a_rule_per_subtype(tmp_path: Path) -> None:
    rules = load_legacy_rules(rules_file(tmp_path, f"{SUBTYPE}\t1\tfalse\ta later change"))
    assert rule_for(rules, SUBTYPE) == LegacyRule(SUBTYPE, 1, False, "a later change")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (f"{SUBTYPE}\t1\ttrue\t", "has no reason"),
        (f"{SUBTYPE}\tone\ttrue\tx", "is not a count"),
        (f"{SUBTYPE}\t-1\ttrue\tx", "is not a count"),
        (f"{SUBTYPE}\t1\tmaybe\tx", "is not true or false"),
        ("\t1\ttrue\tx", "no subtype"),
    ],
)
def test_a_bad_rule_is_refused(tmp_path: Path, row: str, message: str) -> None:
    with pytest.raises(LegacyError, match=message):
        load_legacy_rules(rules_file(tmp_path, row))


def test_duplicates_columns_absence_and_no_rule_are_refused(tmp_path: Path) -> None:
    with pytest.raises(LegacyError, match="listed twice"):
        load_legacy_rules(rules_file(tmp_path, f"{SUBTYPE}\t1\ttrue\tx", f"{SUBTYPE}\t0\ttrue\ty"))
    with pytest.raises(LegacyError, match="columns must be"):
        load_legacy_rules(rules_file(tmp_path, "x", header="subtype\n"))
    with pytest.raises(LegacyError, match="not found"):
        load_legacy_rules(tmp_path / "absent.tsv")
    with pytest.raises(LegacyError, match="no legacy rule for 'B/Vic'"):
        rule_for(load_legacy_rules(rules_file(tmp_path, f"{SUBTYPE}\t1\ttrue\tx")), "B/Vic")


# ------------------------------------------------------------------ labelling


def test_the_deepest_fully_carried_legacy_clade_wins(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    assert legacy_label(virus(p20="V", p21="W", p22="K"), clade_set, rule(0)).clade == "L.1.1"
    # without L.1.1's own marker, the label stops at L.1: a clade's own loci are never forgiven
    assert legacy_label(virus(p20="V", p21="W"), clade_set, rule(1)).clade == "L.1"
    assert legacy_label(virus(), clade_set, rule(1)).clade is None


def test_one_ancestral_locus_is_forgiven_and_recorded(tmp_path: Path) -> None:
    """The root's 20V has changed later in the lineage; everything below it is carried."""
    clade_set = legacy_set(tmp_path)
    changed = virus(p20="A", p21="W", p22="K")
    assert legacy_label(changed, clade_set, rule(0)).clade is None
    call = legacy_label(changed, clade_set, rule(1))
    assert (call.clade, call.tolerated) == ("L.1.1", ("L:20V",))


def test_a_root_locus_is_forgiven_only_when_the_rule_says_so(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    changed = virus(p20="A", p21="W", p22="K")
    assert legacy_label(changed, clade_set, rule(1, root=False)).clade is None


def test_the_budget_counts_every_forgiven_locus(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    # L.1.1 would need both 21W (L.1) and 20V (L) forgiven: over a budget of 1
    assert legacy_label(virus(p20="A", p21="A", p22="K"), clade_set, rule(1)).clade is None
    assert legacy_label(virus(p20="A", p21="A", p22="K"), clade_set, rule(2)).clade == "L.1.1"


def test_a_tie_is_labelled_with_the_common_ancestor(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    call = legacy_label(virus(p20="V", p21="W", p24="S"), clade_set, rule(0))
    assert (call.clade, call.ambiguous) == ("L", ("L.1", "L.3"))


def test_counts(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    sequences = {
        "strict": virus(p20="V", p21="W"),
        "forgiven": virus(p20="A", p21="W"),
        "tie": virus(p20="V", p21="W", p24="S"),
        "none": virus(),
        "no sequence": None,
    }
    calls, counts = legacy_labels(sequences, clade_set, rule(1))
    assert calls["forgiven"].clade == "L.1"
    assert counts.to_json() == {
        "labelled": 3,
        "strict": 2,
        "tolerated_loci": {"L:20V": 1},
        "ambiguous": {"L.1 / L.3": 1},
        "none": 1,
        "no_aligned_sequence": 1,
    }


# ------------------------------------------------------------------ colour schemes


def rows(*keys: str) -> list[tuple[str, dict[str, str]]]:
    return [
        (f"row {i}", {"order": str(i), "key": key, "legend": key, "colour": f"#0000{i:02d}"})
        for i, key in enumerate(keys, start=1)
    ]


def key_of(entry: ColourEntry | None) -> str | None:
    return entry.key if entry is not None else None


def test_a_scheme_row_may_name_a_legacy_clade(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    scheme = scheme_from_rows(rows("P", "L", "L.1"), SUBTYPE, "s", clade_set)
    assert [(e.key, e.is_legacy) for e in scheme] == [("P", False), ("L", True), ("L.1", True)]
    old = virus(p20="V", p21="W", p22="K")
    assert key_of(scheme.entry_for(None, old, clade_set, legacy_clade="L.1.1")) == "L.1"
    assert key_of(scheme.entry_for(None, old, clade_set, legacy_clade="L.2")) == "L"
    # a virus the subclades name is never coloured by a legacy row
    assert key_of(scheme.entry_for("P.1", old, clade_set)) == "P"
    with pytest.raises(ValueError, match="never both"):
        scheme.entry_for("P", old, clade_set, legacy_clade="L")


def test_a_short_or_pointer_legacy_name_is_not_a_legacy_key(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    with pytest.raises(ColourSchemeError, match="'l1' is neither"):
        scheme_from_rows(rows("l1"), SUBTYPE, "s", clade_set)


def test_a_legacy_row_is_shadowed_only_by_a_later_legacy_ancestor(tmp_path: Path) -> None:
    clade_set = legacy_set(tmp_path)
    scheme = scheme_from_rows(rows("L.1", "P", "L"), SUBTYPE, "s", clade_set)
    [shadowed] = shadowed_entries(scheme, clade_set)
    assert (shadowed.entry.key, shadowed.by.key) == ("L.1", "L")


def test_the_importer_resolves_a_full_legacy_name_and_refuses_a_short_one(tmp_path: Path) -> None:
    from af.clades.importer import import_semantic_clades

    clade_set = legacy_set(tmp_path)
    module = tmp_path / "semantic_clades.py"
    module.write_text(
        "from ae.utils.org import org_table_to_dict\n"
        'sData = {"A(H3N2)": {"clades-v1": org_table_to_dict("""\n'
        "| name  | legend | color   |\n|-------+--------+---------|\n"
        "| L.1.1 | old    | #112233 |\n| l1    | short  | #445566 |\n"
        '""")}}\n'
    )
    report = import_semantic_clades(module, {SUBTYPE: clade_set})
    assert [row["key"] for row in report.colour_rows[(SUBTYPE, "clades-v1")]] == ["L.1.1"]
    assert report.legacy == [(SUBTYPE, "clades-v1", 1, "L.1.1")]
    [dead] = report.dead
    assert "by its short name 'l1'" in dead.reason


# ------------------------------------------------------------------ publishing


def sequences_with_nucleotides(store: Store, dataset: StoreRef) -> StoreRef:
    rows = [
        ("EPI_ISL_910001", "EPI910001", "P", "good", nucleotides(p5="K")),
        ("EPI_ISL_910002", "EPI910002", "unassigned", "good", nucleotides(p20="V", p21="W")),
        ("EPI_ISL_910003", "EPI910003", "unassigned", "good", nucleotides(p20="A", p21="W")),
        ("EPI_ISL_910004", "EPI910004", "unassigned", "good", nucleotides()),
    ]
    names = ["epi_isl", "accession", "nextclade_subclade", "nextclade_qc_status", "nuc_aligned"]
    columns = {name: [row[i] for row in rows] for i, name in enumerate(names)}
    with store.build("sequences", "h3") as builder:
        part = builder.path / "sequences" / "pull=test"
        part.mkdir(parents=True)
        pq.write_table(pa.table(columns), part / "part-0.parquet")
        return builder.publish(Provenance("seq.build", (dataset,), {}, STARTED, STARTED))


def publish_with_legacy(tmp_path: Path, store: Store, *, with_clades_dir: bool = True) -> StoreRef:
    clade_set = legacy_set(tmp_path)
    dataset = raw_dataset(store, "good", "P", "P.1", "P.1.1", "P.2")
    sequences = sequences_with_nucleotides(store, dataset)
    clone = tmp_path / "synthetic_HA"
    nomenclature = [ExternalInput.of(clone / "subclades", version=clade_set.pin.commit)]
    if with_clades_dir:
        nomenclature.append(ExternalInput.of(clone / "clades", version=clade_set.pin.commit))
    return publish_clades(
        store,
        SUBTYPE,
        clade_set,
        tree=None,
        sequences=sequences,
        nomenclature=nomenclature,
        started=STARTED,
        legacy=rule(1),
    )


def test_unnamed_rows_get_a_legacy_label_and_named_rows_do_not(tmp_path: Path) -> None:
    import duckdb

    store = Store.create(tmp_path / "store")
    ref = publish_with_legacy(tmp_path, store)
    path = store.resolve(ref) / ASSIGNMENTS_FILE
    got = {
        r[0]: (r[1], r[2], r[3])
        for r in duckdb.sql(
            "select epi_isl, clade, legacy_clade, legacy_tolerated from read_parquet(?)",
            params=[str(path)],
        ).fetchall()
    }
    assert got == {
        "EPI_ISL_910001": ("P", None, []),
        "EPI_ISL_910002": (None, "L.1", []),
        "EPI_ISL_910003": (None, "L.1", ["L:20V"]),
        "EPI_ISL_910004": (None, None, []),
    }
    report = read_report(store, ref)["legacy"]
    assert (report["unnamed"], report["labelled"], report["none"]) == (3, 2, 1)
    provenance = json.loads((store.resolve(ref) / "PROVENANCE.json").read_text())
    assert provenance["parameters"]["legacy"]["max_ancestor_contradictions"] == 1


def test_a_legacy_label_needs_its_definitions_in_the_provenance(tmp_path: Path) -> None:
    store = Store.create(tmp_path / "store")
    with pytest.raises(CladeStoreError, match="clades/ directory"):
        publish_with_legacy(tmp_path, store, with_clades_dir=False)
