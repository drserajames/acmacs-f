"""The af table: one lab's assay table as read from its raw source (interface I2).

A table holds what the lab reported, lightly normalised, and nothing inferred from other
tables. Cross-table identity, titre merging and column bases belong to the chart model
(workstream 7), so a cell keeps every reading the lab gave for it rather than a merged value.

Two hashes. ``content_hash`` covers everything except provenance (where and when the table
was read): it names the stored object. ``map_hash`` covers only what a map is built from
(table identity, antigen and serum identity and dates, titres): a table whose ``map_hash``
changes restarts its chains there. CDC fills in sequencing metadata (EPI_ISL, sequenced
passage, pairing status) long after a test; that changes ``content_hash`` only.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

FORMAT = "af-table-2"
OLD_FORMATS = ("af-table-1",)  # read, never written; see Table.from_json
# map_hash's own format marker: it changes only when what a map is built from changes, so a
# new metadata field (af-table-2 added passage_class) does not restart every chain
MAP_FORMAT = "af-table-1"


@dataclass
class Antigen:
    name: str  # normalised for identity, e.g. "A(H3N2)/EXAMPLETOWN/2/2018"
    raw_name: str  # exactly as the lab wrote it
    passage: str = ""
    passage_date: str | None = None  # CDC harvest date (ISO); kept apart from the passage
    passage_class: str = ""  # egg/cell/original/unknown (af.tables.passage.class_of); "" = not set
    date: str | None = None  # collection date (ISO)
    lab_ids: list[str] = field(default_factory=list)  # e.g. ["CDC#3000415788"]
    reassortant: str = ""
    annotations: list[str] = field(default_factory=list)
    lineage: str = ""  # B only: "VICTORIA"/"YAMAGATA"; per antigen, since old B tables mix them
    reference: bool = False  # the lab marked it a reference antigen
    epi_isl: str = ""  # GISAID isolate the lab paired this antigen with ("" = none given)
    sequence_pairing: str = ""  # "exact" (sequenced this passage), "proxy" (another isolate), ""
    source: dict[str, Any] = field(default_factory=dict)  # lab-specific fields, verbatim

    def ae_passage(self) -> str:
        """ae's passage string, harvest date included (whocc-cdc-tsv-ace make_passage):
        "<passage> (<date>)" when there is a date, else the passage. Cross-table identity in
        ae compares this whole string."""
        return f"{self.passage} ({self.passage_date})" if self.passage_date else self.passage


@dataclass
class Serum:
    name: str
    raw_name: str
    serum_id: str = ""  # "CDC <lot>" for CDC
    passage: str = ""
    passage_date: str | None = None
    passage_class: str = ""  # egg/cell/original/unknown (af.tables.passage.class_of); "" = not set
    species: str = ""  # "" = the lab's default (ferret); "MOUSE" etc. when a rule says so
    lineage: str = ""
    reassortant: str = ""
    annotations: list[str] = field(default_factory=list)  # e.g. ["BOOSTED"]
    epi_isl: str = ""  # as for Antigen: the serum strain's paired GISAID isolate
    sequence_pairing: str = ""
    source: dict[str, Any] = field(default_factory=dict)

    def ae_passage(self) -> str:
        """ae's passage string, harvest date included (whocc-cdc-tsv-ace make_passage):
        "<passage> (<date>)" when there is a date, else the passage. Cross-table identity in
        ae compares this whole string."""
        return f"{self.passage} ({self.passage_date})" if self.passage_date else self.passage


@dataclass
class Table:
    table_id: str  # stable, e.g. "h3-hint-cdc-20260901" or "h3-hint-cdc-20260901.2"
    group: str  # "h3-hint-cdc": the tables a chain is built from
    lab: str
    subtype: str  # "A(H1N1)", "A(H3N2)", "B"
    lineage: str  # "VICTORIA", "YAMAGATA" or ""
    assay: str  # "HI", "HINT", "FRA", ...
    rbc: str  # "turkey", "guinea-pig", ... ; "" for neutralisation assays
    date: str  # ISO test date
    date_suffix: int  # 1 for the first table of the day, 2 for ".2", ...
    source_key: str  # stable identifier of the table in its source, e.g. "CDC test_id 3401"
    antigens: list[Antigen]
    sera: list[Serum]
    titres: list[list[list[str]]]  # [antigen][serum] -> readings; [] = not tested
    meta: dict[str, Any] = field(default_factory=dict)  # lab-specific table fields
    dropped: dict[str, int] = field(default_factory=dict)  # counts, by reason
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)  # excluded from the hash

    @property
    def test_date(self) -> dt.date:
        """The test date, parsed. Consumers use this (or ``order_key``), never the id string."""
        return dt.date.fromisoformat(self.date)

    @property
    def order_key(self) -> tuple[dt.date, int]:
        """Chronological order of tables in a group: date, then the same-day number (1, 2, ...)."""
        return self.test_date, self.date_suffix

    def content(self) -> dict[str, Any]:
        """Everything that the hash covers: the table minus its provenance."""
        d = asdict(self)
        d.pop("provenance")
        d["format"] = FORMAT
        return d

    def content_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.content()).encode()).hexdigest()

    def content_hash_as(self, fmt: str) -> str:
        """The hash this table would have had written in ``fmt`` (an older format: the
        fields added since left out). Lets a golden record made before a format change
        still check everything the older format covered."""
        return self.content_hash() if fmt == FORMAT else _legacy_hash(self, fmt)

    def map_content(self) -> dict[str, Any]:
        """The part of the table a map depends on (see the module docstring)."""
        d = self.content()
        d["format"] = MAP_FORMAT
        for key in ("meta", "warnings", "dropped", "source_key"):
            d.pop(key)
        for item in (*d["antigens"], *d["sera"]):
            for key in ("source", "raw_name", "epi_isl", "sequence_pairing", "passage_class"):
                item.pop(key)  # sequence links (filled in later) and what is derived from passage
        return d

    def map_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.map_content()).encode()).hexdigest()

    def to_json(self) -> dict[str, Any]:
        d = self.content()
        d["content_hash"] = self.content_hash()
        d["provenance"] = self.provenance
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Table:
        """Reads the current format and the older ones. An older table is checked against
        its own hash (computed as it was written) and comes back with the fields added since
        at their defaults, so ``content_hash`` of the result is the current format's."""
        fmt = d.get("format")
        if fmt != FORMAT and fmt not in OLD_FORMATS:
            raise ValueError(f"not an {FORMAT} table: format={fmt!r}")
        kw = {k: v for k, v in d.items() if k not in ("format", "content_hash")}
        kw["antigens"] = [Antigen(**a) for a in d["antigens"]]
        kw["sera"] = [Serum(**s) for s in d["sera"]]
        table = cls(**kw)
        stored = table.content_hash_as(fmt)
        if stored != d["content_hash"]:
            raise ValueError(
                f"{table.table_id}: stored hash {d['content_hash']} != content {stored}"
            )
        return table

    def check(self) -> list[str]:
        """Structural invariants; an empty list means the table is sound."""
        errors = []
        if not self.antigens:
            errors.append("no antigens")
        if not self.sera:
            errors.append("no sera")
        if len(self.titres) != len(self.antigens):
            errors.append(f"{len(self.titres)} titre rows for {len(self.antigens)} antigens")
        for no, row in enumerate(self.titres):
            if len(row) != len(self.sera):
                errors.append(f"titre row {no}: {len(row)} cells for {len(self.sera)} sera")
        if errors:
            return errors  # the per-antigen/serum checks below assume the shape is right
        for no, ag in enumerate(self.antigens):
            if all(not cell for cell in self.titres[no]):
                errors.append(f"antigen {no} {ag.name} has no titres")
        for no, sr in enumerate(self.sera):
            if all(not row[no] for row in self.titres):
                errors.append(f"serum {no} {sr.name} has no titres")
        return errors


def _legacy_hash(table: Table, fmt: str) -> str:
    """content_hash as an ``fmt`` table was hashed: af-table-1 had no passage_class."""
    if fmt not in OLD_FORMATS:
        raise ValueError(f"unknown table format {fmt!r}")
    d = table.content()
    d["format"] = fmt
    for item in (*d["antigens"], *d["sera"]):
        item.pop("passage_class")
    return hashlib.sha256(canonical_json(d).encode()).hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
