"""The report's geo figures for a window: ``python -m af.geo.build``.

Reads the stores under one guarded read (:meth:`af.store.Store.reading`), colours every
preparation by its clade through the same join and user tables as the maps, and writes per
subtype and month ``<figures>/geo/<slot>/<YYYY-MM>/af/figure.{pdf,i7.json}``, the clade key
``<figures>/geo/clade-key.json`` for the report's geo intro page, and the run record
``<figures>/geo/af-build.json``. Every input is an explicit argument (design rule 4): the
stores, the data checkouts, the coastline, the window, the scheme per subtype row and the
report slot per table subtype.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from af.geo.figures import clade_key, write_geo_figures
from af.geo.records import DotRule, Month
from af.map.colouring import StoreColours
from af.map.config import ColouringConfig
from af.seq.matching_rules import matching_rules
from af.serology.outputs import make_geo_and_stat
from af.store import Store
from af.util.artefacts import sha256_path
from af.util.subtypes import subtypes

# what a geo build reads: tables (the serology staleness guard), serology, sequences, clades
KINDS_READ = frozenset({"tables", "serology", "sequences", "clades"})


def build_geo(
    store: Store,
    af_data: Path,
    clones: Path,
    acmacs_data: Path,
    coastline: Path,
    first: Month,
    last: Month,
    figures: Path,
    work: Path,
    schemes: Mapping[str, str],
    slots: Mapping[str, str],
    *,
    ignore_busy: bool = False,
    created: dt.datetime | None = None,
) -> dict[str, Any]:
    """Write the geo figures, key and run record; returns the run record.

    Every figure's provenance carries what the report requires of each figure
    (:mod:`af.report.figures`, :mod:`af.report.provenance`): ``created`` (time-zoned; the
    report's "latest" figure per slot is chosen by it) and ``store_refs``, the store versions
    the slot's dots and colours were READ from: serology, and the sequences and clades its
    colours can come from (:meth:`StoreColours.store_refs`), so a stale geo input is refused by
    the report rather than unchecked.

    ``schemes``: subtype row -> scheme name (``{"h3": "clades-v10", ...}``). ``slots``: table
    subtype -> report slot (``{"A(H3N2)": "h3", "B": "bvic", ...}``). A table subtype drawn
    without a slot is an error, as is a slot for one that has no dots (design rule 1).
    """
    with store.reading("geo-build", kinds=KINDS_READ, override=ignore_busy) as guard:
        # the maps' one read of the join and the user's tables (design rule 6): a virus is the
        # colour on a geo page that it is on a map
        colours = StoreColours(
            store, ColouringConfig("store", acmacs_data, clones), matching_rules(af_data),
            ignore_busy=ignore_busy,
        )  # fmt: skip
        report = make_geo_and_stat(
            store, af_data / "rules" / "locations", coastline, first, last, work,
            dot_rule=DotRule.VIRUS, store_colours=colours, schemes=schemes,
        )  # fmt: skip
    created = created or dt.datetime.now(dt.UTC)
    if created.tzinfo is None:
        raise ValueError(f"created {created.isoformat()} has no time zone")
    drawn = set(report.geo_docs)
    if missing := sorted(drawn - set(slots)):
        raise ValueError(f"no report slot given for {missing}")
    if unused := sorted(set(slots) - drawn):
        raise ValueError(f"slots given for subtypes with no geo dots: {unused}")
    assert report.coordinates is not None  # set whenever a document was drawn
    links = report.links
    provenance = {
        "store_read": guard.to_json(),
        "colours_store_read": colours.store_read,
        "serology": report.serology.to_json(),
        "refs": links.refs if links is not None else {},
        "clades_behind": {d: list(v) for d, v in links.clades_behind.items()} if links else {},
        "clades_same_content": (
            {d: list(v) for d, v in links.clades_same_content.items()} if links else {}
        ),
        "schemes": dict(schemes),
        "colour_inputs": report.colour_inputs,
        "matching_inputs": report.matching_inputs,
        "location_tables": report.location_tables,
        "coastline": {"path": str(coastline), "sha256": sha256_path(coastline)},
        "dot_rule": report.geo_rule,
        "merges": report.geo_merges,
        "merged_preparations": report.geo_merged_preparations,
    }
    files: list[Path] = []
    key: dict[str, Any] = {}
    record: dict[str, Any] = {"provenance": provenance, "created": created.isoformat(), "slots": {}}
    for subtype, doc in sorted(report.geo_docs.items()):
        slot = slots[subtype]
        all_rows = [r.key for r in subtypes().for_table(subtype, "")]
        rows = [r for r in all_rows if r in report.colour_schemes]
        slot_schemes = [report.colour_schemes[r] for r in rows]
        # the sequence datasets this slot read: every row of its table subtype (a B dot of
        # unknown lineage is placed in a row by the dataset it matched in, and B/Yam dots are
        # drawn uncoloured from their own), plus any dataset its colours came from
        datasets = sorted(
            {d for r in rows for d in report.colour_provenance[r]["datasets"]} | set(all_rows)
        )
        slot_provenance = {
            **provenance,
            "created": created.isoformat(),
            "store_refs": colours.store_refs(datasets),
        }
        written = write_geo_figures(
            doc, slot, report.coordinates, coastline, figures, slot_provenance,
            scheme=", ".join(s.name for s in slot_schemes),
        )  # fmt: skip
        files += written.files
        key[slot] = clade_key(doc, slot_schemes)
        record["slots"][slot] = {
            "subtype": subtype,
            "store_refs": slot_provenance["store_refs"],
            "drawn": written.drawn,
            "no_coordinates": dict(sorted(written.no_coordinates.items())),
            "undated": doc["undated"],
            "no_location": len(doc["no_location"]),
        }
    record["colours"] = {
        row: {"coloured": dict(c.coloured), "uncoloured": dict(c.uncoloured)}
        for row, c in sorted(report.colours.items())
    }
    record["colour_provenance"] = report.colour_provenance
    record["unknown_lineage"] = report.unknown_lineage
    record["coloured_from_partial"] = report.coloured_from_partial
    out = figures / "geo"
    (out / "clade-key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n")
    record["files"] = sorted(str(f.relative_to(figures)) for f in files)
    (out / "af-build.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    return record


def _pairs(values: Sequence[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        name, sep, target = value.partition("=")
        if not sep or not name or not target:
            raise SystemExit(f"--{what} {value!r}: expected NAME=VALUE")
        if name in out:
            raise SystemExit(f"--{what} {name!r} given twice")
        out[name] = target
    return out


def _month(text: str) -> Month:
    year, sep, month = text.partition("-")
    if not sep or not (year.isdigit() and month.isdigit()):
        raise argparse.ArgumentTypeError(f"{text!r}: expected YYYY-MM")
    return Month(int(year), int(month))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--af-data", type=Path, required=True, help="acmacs-f-data checkout")
    parser.add_argument("--clones", type=Path, required=True, help="clade nomenclature clones")
    parser.add_argument("--acmacs-data", type=Path, required=True, help="the user's clade tables")
    parser.add_argument("--coastline", type=Path, required=True, help="Natural Earth .shp")
    parser.add_argument("--first", type=_month, required=True, help="YYYY-MM")
    parser.add_argument("--last", type=_month, required=True, help="YYYY-MM")
    parser.add_argument("--figures", type=Path, required=True, help="the report's figure root")
    parser.add_argument("--work", type=Path, required=True, help="scratch for records and stat")
    parser.add_argument("--scheme", action="append", default=[], help="ROW=SCHEME, per row")
    parser.add_argument("--slot", action="append", default=[], help="TABLE_SUBTYPE=SLOT")
    parser.add_argument("--ignore-busy", action="store_true", help="diagnosis only")
    args = parser.parse_args(argv)
    record = build_geo(
        Store.open(args.store), args.af_data, args.clones, args.acmacs_data, args.coastline,
        args.first, args.last, args.figures, args.work, _pairs(args.scheme, "scheme"),
        _pairs(args.slot, "slot"), ignore_busy=args.ignore_busy,
    )  # fmt: skip
    for slot, info in record["slots"].items():
        print(f"{slot:6s} {sum(info['drawn'].values()):6d} dots  {info['drawn']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
