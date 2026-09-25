"""The per-round map configuration: which maps, and every named choice made about them.

Why a schema rather than a dict: each field here is a decision someone took about a published
figure, so it is declared, validated and carries its reason. A key the file sets that no field
declares is an error (a typo would otherwise silently leave a setting at its default), and a
rule that matches nothing is an error unless marked optional — the failure modes the per-folder
scripts this replaces were full of.

Only a round's own choices belong in the round's file. Rules live in code, and facts shared
between rounds (the curated vaccine list, colour schemes) live in the shared data repo.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class WindowConfig:
    """A time window of a map. ``since`` absent means "every antigen"."""

    name: str
    since: dt.date | None = None


@dataclass(frozen=True)
class FrameConfig:
    """Frame size for a subtype, optionally narrowed to one assay.

    A size is raised for the whole subtype (or subtype+assay), never for one map and never by
    hiding a recent antigen (Sarah, 25 Sep 2026). Omit ``assay`` to cover every assay of the
    subtype, which is what "keep all h3 maps at same scale" means.
    """

    subtype: str
    size: float
    assay: str | None = None


@dataclass(frozen=True)
class MoveConfig:
    """Move named antigens to the median of a colour-scheme row's points, then relax."""

    name: str
    reason: str
    decided: dt.date
    movers: tuple[str, ...]
    target_legend: str
    max_stress_rise: float
    max_from_target: float
    min_target_points: int = 5


@dataclass(frozen=True)
class HideConfig:
    """Hide named points. The list must match exactly, so it cannot rot unnoticed."""

    name: str
    reason: str
    decided: dt.date
    designations: tuple[str, ...] = ()
    designations_file: Path | None = None

    def load(self) -> tuple[str, ...]:
        """The designations, from the inline list or the file beside the config."""
        if self.designations and self.designations_file:
            raise ValueError(
                f"hide {self.name!r}: give designations or designations_file, not both"
            )
        if self.designations_file is not None:
            lines = self.designations_file.read_text().splitlines()
            return tuple(ln.strip() for ln in lines if ln.strip() and not ln.startswith("#"))
        if not self.designations:
            raise ValueError(f"hide {self.name!r}: no designations")
        return self.designations


@dataclass(frozen=True)
class RotationConfig:
    """A one-off turn applied after the automatic orientation."""

    name: str
    reason: str
    decided: dt.date
    degrees: float
    reflect: bool = False


@dataclass(frozen=True)
class VaccineDisableConfig:
    name: str
    reason: str
    passage: str = "any"
    optional: bool = False


@dataclass(frozen=True)
class VaccineChooseConfig:
    name: str
    reason: str
    passage_class: str
    passage: str
    optional: bool = False


@dataclass(frozen=True)
class MapConfig:
    """One map folder: where its chart comes from, and its named choices."""

    folder: str
    clade_scheme: str
    chain: str | None = None  # store dataset, resolved to CURRENT at run time
    layout_stand_in: Path | None = None  # bring-up: a chart to take the layout from
    # bring-up: where to read the colour rows, when the layout chart does not carry them all.
    # A chain's chart is unstyled, and a pre-curation chart can be missing rows that the round's
    # styling step added, so the scheme often comes from a different chart than the layout.
    scheme_stand_in: Path | None = None
    title: str | None = None  # default: built from the chart's own lab/subtype/assay
    moves: tuple[MoveConfig, ...] = ()
    hides: tuple[HideConfig, ...] = ()
    rotations: tuple[RotationConfig, ...] = ()
    vaccine_disable: tuple[VaccineDisableConfig, ...] = ()
    vaccine_choose: tuple[VaccineChooseConfig, ...] = ()

    def __post_init__(self) -> None:
        if not self.chain and not self.layout_stand_in:
            raise ValueError(f"map {self.folder!r}: needs a chain or a layout_stand_in")


@dataclass(frozen=True)
class Defaults:
    must_show_since: dt.date
    windows: tuple[WindowConfig, ...]
    min_common_points: int = 50
    previous_round: Path | None = None
    orientation_reference: str = "previous-round"


@dataclass(frozen=True)
class MapsConfig:
    defaults: Defaults
    frames: tuple[FrameConfig, ...]
    maps: tuple[MapConfig, ...]
    vaccine_defaults: Path | None = None  # shared subtype defaults (acmacs-f-data)

    def frame_size(self, subtype: str, assay: str) -> float:
        """Most specific frame rule wins: subtype+assay, then subtype alone."""
        for f in self.frames:
            if f.subtype == subtype and f.assay == assay:
                return f.size
        for f in self.frames:
            if f.subtype == subtype and f.assay is None:
                return f.size
        raise ValueError(f"no frame size configured for subtype {subtype!r} assay {assay!r}")


@dataclass
class VaccineDefaults:
    """Subtype-wide vaccine rules shared by every round (one editable copy, design rule 6)."""

    disable: dict[str, tuple[VaccineDisableConfig, ...]] = field(default_factory=dict)

    def for_subtype(self, subtype: str) -> tuple[VaccineDisableConfig, ...]:
        return self.disable.get(subtype, ())
