"""Read CDC's titre export (fludata CDC-Atlanta-WHO-CC, ``CDC_titers_sept_2019_onwards.tsv``).

The TSV has one row per titre. One af table is made per CDC ``test_id`` (CDC's own stable
identifier for a test). ae instead grouped rows by subtype + assay + date, which silently
fused nine pairs of separate tests into single tables.

Rows are dropped, and counted, when CDC marks them not for use (DECISIONS 24 Sep 2026):
``ag_do_not_report``, ``sr_do_not_report``, ``titer_reportable = FALSE``, ``titer_error``.
Sera are dropped or tagged by ``control_sera`` rules (human pools, mouse sera).

Within a test, an antigen is name + passage + harvest date + CDC id, and a serum is name +
lot + boosted, as in ae (whocc-cdc-tsv-ace:416-434). Repeated readings of one antigen/serum
pair are all kept in the cell; merging them is the chart model's job.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import aliases
from .model import Antigen, Serum, Table
from .passage import PassageParser
from .rules import Rules

LAB = "CDC"

# CDC's column set (their 2021-01-29 schema, still current). Unknown or missing columns are
# fatal: a format change must be looked at, not guessed at.
COLUMNS = [
    "test_id",
    "test_date",
    "test_file",
    "test_protocol",
    "test_subtype",
    "ag_isl_type",
    "ag_ha_type",
    "sr_isl_type",
    "sr_ha_type",
    "ag_position",
    "ag_entry_id",
    "ag_cdc_id",
    "ag_isolate_id",
    "ag_epi_isolate_id",
    "ag_strain_name",
    "ag_passage",
    "ag_seq_passage",
    "ag_collection_date",
    "ag_date_harvested",
    "ag_lab",
    "ag_type",
    "ag_is_homologous",
    "ag_do_not_report",
    "ag_back_titer",
    "ag_entry_error",
    "ag_pairing_status",
    "ag_test_for_fra",
    "sr_position",
    "sr_entry_id",
    "sr_cdc_id",
    "sr_isolate_id",
    "sr_epi_isolate_id",
    "sr_strain_name",
    "sr_passage",
    "sr_seq_passage",
    "sr_lot",
    "sr_ferret",
    "sr_collection_date",
    "sr_date_harvested",
    "sr_boosted",
    "sr_lab",
    "sr_pool",
    "sr_do_not_report",
    "sr_pairing_status",
    "titer_reportable",
    "titer_error",
    "titer_value",
    "titer_log",
    "titer_logfold",
]

PROTOCOLS = {  # test_protocol -> assay
    "hi_protocol": "HI",
    "hi_oseltamivir_protocol": "HI",  # the protocol is kept in Table.meta
    "hint_protocol": "HINT",
    "fra_protocol": "FRA",
}
SUBTYPES = {  # test_subtype -> (subtype, lineage, group prefix)
    "H1 swl": ("A(H1N1)", "", "h1pdm"),
    "H3": ("A(H3N2)", "", "h3"),
    "B": ("B", "", "b"),
    "B vic": ("B", "VICTORIA", "bvic"),
    "B yam": ("B", "YAMAGATA", "byam"),
}
# CDC's "not for use" flags: column -> the value that means "drop this row".
FLAGS = {
    "ag_do_not_report": "TRUE",
    "sr_do_not_report": "TRUE",
    "titer_reportable": "FALSE",
    "titer_error": "TRUE",
}
BOOLEAN = ("TRUE", "FALSE")
PAIRING = {"exact": "exact", "isolate proxy": "proxy", "": ""}  # ag/sr_pairing_status
EPI_ISL = re.compile(r"EPI_ISL_[0-9]+")
MERGED_ISOLATES = "antigen merges two CDC isolate ids"
HA_TYPES = {
    "VIC": "VICTORIA",
    "YAM": "YAMAGATA",
    "H1": "",
    "H3": "",
    "": "",
}  # ag_ha_type/sr_ha_type
PLAIN_TITRE = re.compile(r"[<>]?[1-9]\d*")

# Kept verbatim in Antigen.source / Serum.source: useful downstream (EPI_ISL for sequence
# matching, CDC's reference/homologous marks), not part of identity.
AG_SOURCE = (
    "ag_passage",
    "ag_isolate_id",
    "ag_epi_isolate_id",
    "ag_seq_passage",
    "ag_type",
    "ag_lab",
    "ag_pairing_status",
    "ag_test_for_fra",
)
SR_SOURCE = (
    "sr_passage",
    "sr_lot",
    "sr_cdc_id",
    "sr_isolate_id",
    "sr_epi_isolate_id",
    "sr_ferret",
    "sr_seq_passage",
    "sr_collection_date",
    "sr_lab",
    "sr_pool",
    "sr_pairing_status",
)


class CDCFormatError(ValueError):
    pass


@dataclass
class ReadResult:
    tables: list[Table]
    rows: int
    dropped: Counter[str] = field(default_factory=Counter)  # whole-file counts, by reason
    skipped_tests: list[str] = field(default_factory=list)  # tests with nothing left
    errors: list[str] = field(default_factory=list)


def read(path: Path, rules: Rules, *, drop_flagged: bool = True) -> ReadResult:
    """Read the TSV into af tables (without stable ids yet: see ``identity.assign``).

    ``drop_flagged=False`` keeps CDC's not-for-use rows. It exists only to reproduce ae for
    the comparison; production always drops them.
    """
    data = path.read_bytes()
    rows = _rows(path, data)
    by_test: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_test.setdefault(row["test_id"], []).append(row)
    result = ReadResult(tables=[], rows=len(rows))
    provenance = {
        "file": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "reader": "af.tables.cdc",
    }
    for test_id in sorted(by_test, key=int):
        try:
            table = _make_table(test_id, by_test[test_id], rules, drop_flagged, result)
        except CDCFormatError as err:
            result.errors.append(f"test_id {test_id}: {err}")
            continue
        if table is None:
            continue
        table.provenance = dict(provenance)
        if problems := table.check() or aliases.check_titres(table, rules):
            result.errors.extend(f"test_id {test_id}: {p}" for p in problems)
        result.tables.append(table)
    return result


def _rows(path: Path, data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8")
    reader = csv.DictReader(text.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)
    header = reader.fieldnames or []
    if unknown := [c for c in header if c not in COLUMNS]:
        raise CDCFormatError(f"{path}: unknown columns {unknown}")
    if missing := [c for c in COLUMNS if c not in header]:
        raise CDCFormatError(f"{path}: missing columns {missing}")
    rows = []
    for no, row in enumerate(reader, start=2):
        if None in row or any(v is None for v in row.values()):
            raise CDCFormatError(f"{path}:{no}: wrong number of cells")
        row = {k: v.strip() for k, v in row.items()}
        row["_line"] = str(no)
        rows.append(row)
    return rows


def _iso_date(value: str, what: str) -> str:
    """CDC exports ISO dates only; anything else is an error, not a guess (D-ingestion §2.4)."""
    try:
        return datetime.date.fromisoformat(value).isoformat()
    except ValueError:
        raise CDCFormatError(f"{what}: not an ISO date: {value!r}") from None


def _flag(row: dict[str, str], column: str) -> bool:
    value = row[column].upper()
    if value not in BOOLEAN:
        raise CDCFormatError(
            f"line {row['_line']}: {column} = {row[column]!r}, expected TRUE/FALSE"
        )
    return value == FLAGS[column]


def _single(rows: list[dict[str, str]], column: str) -> str:
    values = {r[column] for r in rows}
    if len(values) != 1:
        raise CDCFormatError(f"{column} differs within the test: {sorted(values)}")
    return values.pop()


def _make_table(
    test_id: str, rows: list[dict[str, str]], rules: Rules, drop_flagged: bool, result: ReadResult
) -> Table | None:
    protocol = _single(rows, "test_protocol")
    if protocol not in PROTOCOLS:
        raise CDCFormatError(f"unknown test_protocol {protocol!r}")
    test_subtype = _single(rows, "test_subtype")
    if test_subtype not in SUBTYPES:
        raise CDCFormatError(f"unknown test_subtype {test_subtype!r}")
    assay = PROTOCOLS[protocol]
    subtype, lineage, prefix = SUBTYPES[test_subtype]
    date = _iso_date(_single(rows, "test_date"), "test_date")
    rbc_rule = rules.table_defaults.lookup(lab=LAB, subtype=subtype, assay=assay)
    if rbc_rule is None:
        raise CDCFormatError(f"no table_defaults rule for {LAB} {subtype} {assay}")
    rbc = "" if rbc_rule["rbc"] == "-" else rbc_rule["rbc"]
    group = "-".join(p for p in (prefix, assay.lower(), rbc, LAB.lower()) if p)

    dropped: Counter[str] = Counter()
    warnings: list[str] = []
    kept = []
    for row in rows:
        flags = [c for c in FLAGS if _flag(row, c)]
        for c in flags:
            dropped[f"flag {c}"] += 1  # a row can carry several flags
        if flags and drop_flagged:
            dropped["rows: CDC not-for-use"] += 1
            continue
        kept.append(row)

    antigens: dict[tuple[str, ...], Antigen] = {}
    sera: dict[tuple[str, ...], Serum] = {}
    dropped_sera: set[tuple[str, ...]] = set()
    cells: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    ag_order: dict[tuple[str, ...], tuple[int, int]] = {}
    isolates: dict[tuple[str, ...], set[str]] = {}
    sr_order: dict[tuple[str, ...], tuple[int, str]] = {}
    passages = PassageParser(rules.passage_tokens, LAB)
    for row in kept:
        serum, sr_key, action = _serum(row, subtype, rules, passages, warnings)
        if action == "drop":
            dropped_sera.add(sr_key)
            dropped["rows: control serum"] += 1
            continue
        antigen, ag_key = _antigen(row, subtype, rules, passages, warnings)
        antigens.setdefault(ag_key, antigen)
        isolates.setdefault(ag_key, set()).add(row["ag_isolate_id"])
        if (old := sera.setdefault(sr_key, serum)) is not serum and old.passage != serum.passage:
            warnings.append(
                f"serum {serum.name} {serum.serum_id}: "
                f"passage {old.passage!r} and {serum.passage!r}"
            )
        ag_order[ag_key] = min(
            ag_order.get(ag_key, (10**9, 0)), (int(row["ag_position"]), int(row["_line"]))
        )
        sr_order[sr_key] = min(
            sr_order.get(sr_key, (10**9, "")), (len(row["sr_position"]), row["sr_position"])
        )
        titre = _titre(row, assay, rules)
        if titre is None:
            dropped["readings: discarded token"] += 1
            continue
        cells.setdefault((ag_key, sr_key), []).append(titre)
    dropped["sera: control"] = len(dropped_sera)
    for key, ids in isolates.items():
        if len(ids) > 1 and key in {a for a, _ in cells}:
            # Sarah 25 Sep (Q13): keep merged, flag and count. Same name, passage, harvest
            # date and CDC id, but CDC records two isolates.
            warnings.append(
                f"{MERGED_ISOLATES}: {antigens[key].name} {antigens[key].ae_passage()} "
                f"CDC isolate ids {sorted(ids)}"
            )

    # Keep an antigen or serum only if it has a reading left.
    live_ag = {a for a, _ in cells}
    live_sr = {s for _, s in cells}
    dropped["antigens: no readings left"] = len(set(antigens) - live_ag)
    dropped["sera: no readings left"] = len(set(sera) - live_sr)
    ag_keys = sorted(live_ag, key=lambda k: ag_order[k])
    sr_keys = sorted(live_sr, key=lambda k: sr_order[k])
    dropped = Counter({k: v for k, v in dropped.items() if v})
    result.dropped.update(dropped)
    if not ag_keys or not sr_keys:
        result.skipped_tests.append(
            f"test_id {test_id} ({group} {date}): nothing left; dropped {dict(dropped)}"
        )
        return None
    titres = [[sorted(cells.get((a, s), []), key=_titre_order) for s in sr_keys] for a in ag_keys]
    return Table(
        table_id="",  # assigned by identity.assign
        group=group,
        lab=LAB,
        subtype=subtype,
        lineage=lineage,
        assay=assay,
        rbc=rbc,
        date=date,
        date_suffix=0,
        source_key=f"CDC test_id {test_id}",
        antigens=[antigens[k] for k in ag_keys],
        sera=[sera[k] for k in sr_keys],
        titres=titres,
        meta={
            "test_id": int(test_id),
            "test_file": _single(rows, "test_file"),
            "test_protocol": protocol,
        },
        dropped=dict(sorted(dropped.items())),
        warnings=sorted(set(warnings)),
    )


def _antigen(
    row: dict[str, str], subtype: str, rules: Rules, passages: PassageParser, warnings: list[str]
) -> tuple[Antigen, tuple[str, ...]]:
    name, renamed = aliases.parse_name(
        rules,
        row["ag_strain_name"],
        lab=LAB,
        subtype=subtype,
        applies_to="antigen",
        warnings=warnings,
    )
    passage = passages.parse(row["ag_passage"])
    warnings.extend(passage.problems)
    harvest = (
        _iso_date(row["ag_date_harvested"], "ag_date_harvested")
        if row["ag_date_harvested"]
        else None
    )
    antigen = Antigen(
        name=name.name,
        raw_name=row["ag_strain_name"],
        passage=passage.text,
        passage_class=passages.passage_class(passage.text),
        passage_date=harvest,
        date=_iso_date(row["ag_collection_date"], "ag_collection_date")
        if row["ag_collection_date"]
        else None,
        lab_ids=[f"CDC#{row['ag_cdc_id']}"],
        reassortant=name.reassortant,
        annotations=name.annotations,
        lineage=_lineage(row, "ag_ha_type"),
        epi_isl=_epi_isl(row, "ag_epi_isolate_id"),
        sequence_pairing=_pairing(row, "ag_pairing_status"),
        reference=row["ag_type"] == "reference",
        source={c: row[c] for c in AG_SOURCE},
    )
    if renamed:
        antigen.source[aliases.SOURCE_KEY] = renamed
    return antigen, (row["ag_strain_name"], row["ag_passage"], harvest or "", row["ag_cdc_id"])


def _serum(
    row: dict[str, str], subtype: str, rules: Rules, passages: PassageParser, warnings: list[str]
) -> tuple[Serum, tuple[str, ...], str]:
    name, renamed = aliases.parse_name(
        rules,
        row["sr_strain_name"],
        lab=LAB,
        subtype=subtype,
        applies_to="serum",
        warnings=warnings,
    )
    passage = passages.parse(row["sr_passage"])
    warnings.extend(passage.problems)
    raw_lot, lot_rule = aliases.serum_lot(rules, row["sr_lot"], lab=LAB, ferret=row["sr_ferret"])
    if lot_rule is not None:
        warnings.append(f"serum lot {row['sr_lot']!r} restored as {raw_lot!r} by {lot_rule.where}")
    lot = raw_lot.replace(", ", ",")
    boosted = row["sr_boosted"].upper()
    if boosted not in BOOLEAN:
        raise CDCFormatError(f"line {row['_line']}: sr_boosted = {row['sr_boosted']!r}")
    harvest = (
        _iso_date(row["sr_date_harvested"], "sr_date_harvested")
        if row["sr_date_harvested"]
        else None
    )
    serum = Serum(
        name=name.name,
        raw_name=row["sr_strain_name"],
        serum_id=f"CDC {lot}",
        passage=passage.text,
        passage_class=passages.passage_class(passage.text),
        passage_date=harvest,
        reassortant=name.reassortant,
        annotations=name.annotations + (["BOOSTED"] if boosted == "TRUE" else []),
        lineage=_lineage(row, "sr_ha_type"),
        epi_isl=_epi_isl(row, "sr_epi_isolate_id"),
        sequence_pairing=_pairing(row, "sr_pairing_status"),
        source={c: row[c] for c in SR_SOURCE},
    )
    if renamed:
        serum.source[aliases.SOURCE_KEY] = renamed
    if lot_rule is not None:
        serum.source[aliases.SERUM_ID_KEY] = lot_rule.where
    action = ""
    if (rule := rules.control_sera.find(row["sr_lot"], lab=LAB)) is not None and rule[
        "field"
    ] == "lot":
        action = rule["action"]
        if action == "species":
            serum.species = rule["value"]
        elif action != "drop":
            raise ValueError(f"{rule.where}: unknown action {action!r}")
    return serum, (row["sr_strain_name"], serum.serum_id, boosted), action


def _epi_isl(row: dict[str, str], column: str) -> str:
    value = row[column]
    if value and not EPI_ISL.fullmatch(value):
        raise CDCFormatError(f"line {row['_line']}: {column} = {value!r} is not an EPI_ISL id")
    return value


def _pairing(row: dict[str, str], column: str) -> str:
    if row[column] not in PAIRING:
        raise CDCFormatError(f"line {row['_line']}: {column} = {row[column]!r}")
    return PAIRING[row[column]]


def _lineage(row: dict[str, str], column: str) -> str:
    if row[column] not in HA_TYPES:
        raise CDCFormatError(f"line {row['_line']}: {column} = {row[column]!r}")
    return HA_TYPES[row[column]]


def _titre(row: dict[str, str], assay: str, rules: Rules) -> str | None:
    """The reading as a titre string, or None when a rule discards it."""
    raw = row["titer_value"]
    if (rule := rules.titre_tokens.find(raw, lab=LAB, assay=assay)) is not None:
        return None if rule["titre"] == "*" else rule["titre"]
    if PLAIN_TITRE.fullmatch(raw) and (raw[0] in "<>" or int(raw) >= 10):
        return raw
    raise CDCFormatError(f"line {row['_line']}: titre {raw!r} matches no titre_tokens rule")


def _titre_order(titre: str) -> tuple[int, int]:
    """Readings in a cell are sorted so the hash doesn't depend on the TSV's row order."""
    kind = {"<": 0, ">": 2}.get(titre[0], 1)
    return int(titre.lstrip("<>")), kind
