"""When is an antigen or serum in one table the same as one in another?

One copy of the rule, used by the chain merge, the serology store (workstream 10) and map
matching (workstream 8). It is ae's strict matching (`ae/cc/chart/v3/common.cc:203-262`),
which reproduces today's chains exactly (notes/chains/STATUS.md).

A key of None means "never the same as anything": a DISTINCT point, an antigen with no
passage, a serum with no id. Such points become new points in every table that has them.
"""

from __future__ import annotations

AntigenKey = tuple[str, str, tuple[str, ...], str]
SerumKey = tuple[str, str, tuple[str, ...], str]


def antigen_identity(
    name: str, reassortant: str, annotations: tuple[str, ...] | list[str], passage: str
) -> AntigenKey | None:
    """(name, reassortant, annotations, passage). The passage includes its date, as ae writes it
    ("MDCK2/SIAT1 (2016-05-12)"), so two harvests of one isolate are different antigens."""
    annotations = tuple(annotations)
    if "DISTINCT" in annotations or not passage:
        return None
    return (name, reassortant, annotations, passage)


def serum_identity(
    name: str, reassortant: str, annotations: tuple[str, ...] | list[str], serum_id: str
) -> SerumKey | None:
    """(name, reassortant, annotations, serum id). The serum's passage is not part of it."""
    annotations = tuple(annotations)
    if "DISTINCT" in annotations or not serum_id:
        return None
    return (name, reassortant, annotations, serum_id)
