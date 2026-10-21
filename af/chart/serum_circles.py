"""Serum circles and serum coverage: one record per serum, for the circle and coverage addenda.

A serum circle is drawn around a serum on a map to show which antigens the serum still
"protects": those within ``fold`` log2 units (fold 2.0 = 4-fold, 3.0 = 8-fold) of its titre
against its homologous antigen, the antigen of the serum's own strain. Its radius is
*theoretical* (from the titres: fold + column basis - homologous titre) or *empirical* (from the
map: the radius that best separates the protected antigens from the others). Serum coverage
marks every antigen titrated against the serum as within or outside that fold, by titre alone.

This module computes; it draws nothing (08's map looks draw circles and coverage from these
records). Everything follows ae as the Sep 2026 addenda were made, so that af's pages can be
compared with them, except one thing: ae took the homologous titre one way for the circle and
another for the coverage colouring, so a shipped map could draw a circle and colour its antigens
from two different titres. Two rules choose the homologous antigen:

- ``best-match``: the best-ranked antigen of the serum's name with a titre, passage matched
  (ae ``homologous_titer_for_serum``, which coloured the shipped coverage);
- ``ae-circle``: of the antigens ae's ``Antigens::homologous`` finds, the one giving the
  smallest radius (ae ``serum_circles_for_serum_t``, which drew the shipped circles).

Sarah, Q228 (21 Oct 2026): "keep both rules, use the best match for the figure but declare the
maps which disagree". So a figure's circle and colouring both come from ``best-match`` by
default, every record keeps the other rule's answer too (``other_circle``,
``other_coverage``), and :meth:`SerumCircle.disagreement` says how a map's two answers differ.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np

from af.chart.model import Antigen, Chart, Projection, Serum
from af.chart.titre import Titre, TitreType
from af.map.vaccines import passage_class

MIN_RADIUS = 2.0  # ae serum_circle_min_radius: applies to every fold, and to both radii
RULES = ("best-match", "ae-circle")

# ae Passage::is_egg reads the LAST step only (cc/virus/passage.hh), unlike af's passage_class,
# which reads every step; the homologous match is ae's, so it reads the last step too.
_AE_EGG_STEPS = frozenset({"E", "SPFCE", "SPE", "SPF", "D"})
_NOTE = re.compile(r"\([^()]*\)")  # a lab's note after a step, "E2(Am1Al1)", or a date
_STEP = re.compile(r"[A-Z]+")


class SerumCircleError(ValueError):
    """A rule name af does not know, or a layout that does not fit the chart."""


@dataclass(frozen=True)
class HomologousAntigen:
    """One homologous antigen of a serum and the circle it gives. ``status``: good, or why
    a radius is missing (ae's statuses: non-regular homologous titre, titre too low, serum or
    antigen disconnected)."""

    antigen: int  # chart index: internal, never a selector
    designation: str
    titre: str
    theoretical: float | None
    empirical: float | None
    status: str


@dataclass(frozen=True)
class Circle:
    """The serum's circle under ``rule``: each radius floored at MIN_RADIUS, None when no
    antigen gives one (ae then draws a dashed circle of radius fold). The two radii can come
    from different antigens (ae takes each minimum separately)."""

    rule: str
    empirical: float | None
    empirical_antigen: int | None
    titre: str | None  # the homologous titre of the empirical radius's antigen
    theoretical: float | None
    theoretical_antigen: int | None
    theoretical_titre: str | None  # the homologous titre of the theoretical radius's antigen
    floored: bool  # either radius was raised to MIN_RADIUS
    status: str  # good, no_homologous, no_radius


@dataclass(frozen=True)
class Coverage:
    """Antigens titrated against the serum, within or outside ``fold`` of the homologous
    titre, under ``rule``. Missing titres are in neither (``dont_care`` counts them)."""

    rule: str
    antigen: int | None
    titre: str | None
    threshold: float | None
    within: tuple[int, ...]
    outside: tuple[int, ...]
    dont_care: int
    status: str  # good, no_homologous_titre, non_regular, too_low


@dataclass(frozen=True)
class SerumCircle:
    """Everything the circle and coverage pages need about one serum on one map.

    ``homologous`` is in chart index order (guaranteed): ae colours a circle in the circle
    addenda by the first homologous antigen in that order with a theoretical radius, which is
    not necessarily the one giving the minimum radius.
    """

    serum: int  # chart index: internal, never a selector
    name: str
    designation: str
    serum_id: str
    passage: str
    reassortant: str
    annotations: tuple[str, ...]
    passage_class: str | None  # cell, egg, reassortant; None for a passage that is not one
    xy: tuple[float, float] | None
    fold: float
    column_basis: float
    minimum_column_basis: str
    column_bases_from: str  # projection-forced, chart-forced or computed
    homologous: tuple[HomologousAntigen, ...]
    circle: Circle  # drawn
    coverage: Coverage  # drawn
    other_circle: Circle  # the other rule's answer, kept to declare where they differ (Q228)
    other_coverage: Coverage

    def to_json(self) -> dict:
        return asdict(self)

    def disagreement(self, drawn: str = "empirical") -> str | None:
        """How the two rules' answers for this serum's map differ, or None when they draw the
        same map. ``drawn`` is the radius the figure draws: ``empirical`` (the coverage pages)
        or ``theoretical`` (the circle panels); each radius is its own minimum, so the two can
        come from different antigens and differ apart.

        In order: ``neither`` - no circle and no colouring under the drawn rule;
        ``no-homologous`` - one rule finds no homologous antigen; ``visible`` - the drawn radius
        or the within/outside sets differ; ``titre-only`` - the titres differ and the map does
        not."""
        if drawn not in ("empirical", "theoretical"):
            raise SerumCircleError(f"drawn {drawn!r}: empirical or theoretical")
        radius = getattr(self.circle, drawn)
        other = getattr(self.other_circle, drawn)
        titre = "titre" if drawn == "empirical" else "theoretical_titre"
        if radius is None and self.coverage.status != "good":
            return "neither"
        circles = {self.circle.status, self.other_circle.status}
        if "no_homologous" in circles and len(circles) > 1:
            return "no-homologous"
        if _radius_differs(radius, other) or (self.coverage.within, self.coverage.outside) != (
            self.other_coverage.within,
            self.other_coverage.outside,
        ):
            return "visible"
        if (getattr(self.circle, titre), self.coverage.titre) != (
            getattr(self.other_circle, titre),
            self.other_coverage.titre,
        ):
            return "titre-only"
        return None


def serum_circles(
    chart: Chart,
    layout: np.ndarray,
    fold: float = 2.0,
    *,
    minimum_column_basis: str = "none",
    projection: Projection | None = None,
    circle_rule: str = "best-match",
    coverage_rule: str = "best-match",
) -> list[SerumCircle]:
    """One record per serum of ``chart``, on ``layout`` (the drawn map: n_points x 2, NaN rows
    for points without coordinates). ``fold`` is in log2 units: 2.0 is 4-fold, 3.0 is 8-fold.

    The column bases are the ones the map was drawn with: ``projection``'s forced bases when it
    has them (a round's column-base override goes there), else the chart's at
    ``minimum_column_basis``. Lowering a serum's basis by 1 shrinks its theoretical radius by
    exactly 1, as the Sep 2026 bvic-vidrl override did in ae.
    """
    for what, rule in (("circle", circle_rule), ("coverage", coverage_rule)):
        if rule not in RULES:
            raise SerumCircleError(f"{what} rule {rule!r}: not one of {RULES}")
    layout = np.asarray(layout, dtype=float)
    if layout.ndim != 2 or layout.shape[0] != chart.n_points:
        raise SerumCircleError(
            f"layout has {layout.shape[0] if layout.ndim else 0} points, the chart {chart.n_points}"
        )
    bases = chart.column_bases(minimum_column_basis, projection)
    if projection is not None and projection.forced_column_bases is not None:
        bases_from = "projection-forced"
    elif chart.forced_column_bases is not None:
        bases_from = "chart-forced"
    else:
        bases_from = "computed"
    out = []
    for sr, serum in enumerate(chart.sera):
        cb = float(bases[sr])
        titres = chart.titres.serum_titres(sr)
        xy = layout[chart.n_antigens + sr]
        has_xy = not np.isnan(xy).any()
        distances = _distances(layout, chart.n_antigens + sr, chart.n_antigens)

        def radii(ag: int) -> HomologousAntigen:
            antigen = chart.antigens[ag]
            return _homologous(antigen, ag, titres, distances, has_xy, cb, fold)  # noqa: B023

        found = [radii(ag) for ag in homologous_antigens(chart.antigens, serum)]
        best = coverage_antigen(chart.antigens, serum, titres)
        best_entry = [radii(best)] if best is not None else []
        circles = {
            "best-match": _circle("best-match", best_entry, best is not None),
            "ae-circle": _circle("ae-circle", found, bool(found)),
        }
        smallest = _first_min(found, "empirical")
        coverages = {
            "best-match": _coverage("best-match", best, titres, fold),
            "ae-circle": _coverage(
                "ae-circle", smallest.antigen if smallest else None, titres, fold
            ),
        }
        other = {"best-match": "ae-circle", "ae-circle": "best-match"}
        out.append(
            SerumCircle(
                serum=sr,
                name=serum.name,
                designation=serum.designation(),
                serum_id=serum.serum_id,
                passage=serum.passage,
                reassortant=serum.reassortant,
                annotations=serum.annotations,
                passage_class=passage_class(serum.passage, serum.reassortant),
                xy=(float(xy[0]), float(xy[1])) if has_xy else None,
                fold=fold,
                column_basis=cb,
                minimum_column_basis=minimum_column_basis,
                column_bases_from=bases_from,
                homologous=tuple(found),
                circle=circles[circle_rule],
                coverage=coverages[coverage_rule],
                other_circle=circles[other[circle_rule]],
                other_coverage=coverages[other[coverage_rule]],
            )
        )
    return out


def homologous_antigens(antigens: Sequence[Antigen], serum: Serum) -> list[int]:
    """ae ``Antigens::homologous``: antigens of the serum's name and reassortant whose last
    passage step is egg exactly when the serum's is (or, for a serum with no passage, when its
    id says EGG: NIID writes the passage type there). Annotations must match too, unless no
    antigen matches with them. In chart index order."""

    def matching(respect_annotations: bool) -> list[int]:
        return [
            ag
            for ag, a in enumerate(antigens)
            if a.name == serum.name
            and a.reassortant == serum.reassortant
            and (not respect_annotations or a.annotations == serum.annotations)
            and last_step_is_egg(a.passage) == _serum_is_egg(serum)
        ]

    return matching(True) or matching(False)


def coverage_antigen(
    antigens: Sequence[Antigen], serum: Serum, titres: Sequence[Titre]
) -> int | None:
    """ae ``homologous_titer_for_serum`` (cc/map-draw/styled-draw.cc): of the antigens of the
    serum's name with a titre, the best ranked; a mismatch costs annotations 16, reassortant 8,
    passage 4, and 2 more when the passage type differs too. The first in index order wins a
    tie."""
    serum_type = _passage_type(serum.passage, serum.reassortant)
    best: tuple[int, int] | None = None
    for ag, a in enumerate(antigens):
        if a.name != serum.name or titres[ag].is_missing:
            continue
        rank = 16 * (a.annotations != serum.annotations) + 8 * (a.reassortant != serum.reassortant)
        if a.passage != serum.passage:
            rank += 4 + 2 * (_passage_type(a.passage, a.reassortant) != serum_type)
        if best is None or rank < best[0]:
            best = (rank, ag)
    return None if best is None else best[1]


def last_step_is_egg(passage: str) -> bool:
    """ae ``Passage::is_egg``: is the LAST step egg (or, unparsed, does it say EGG or EMBRYON)."""
    steps = _STEP.findall(_NOTE.sub("", passage.upper()))
    if not steps:
        return False
    last = steps[-1]
    # ae: an unparsed passage saying EGG or EMBRYON anywhere is egg
    return last in _AE_EGG_STEPS or "EGG" in passage.upper() or "EMBRYON" in passage.upper()


def _serum_is_egg(serum: Serum) -> bool:
    if not serum.passage.strip():
        return "EGG" in serum.serum_id
    return last_step_is_egg(serum.passage)


def _passage_type(passage: str, reassortant: str) -> str | None:
    """ae's "p" attribute (ae.semantic.passage): r for a reassortant, else e/c by the last step,
    else none."""
    if reassortant.strip():
        return "r"
    if last_step_is_egg(passage):
        return "e"
    return "c" if passage_class(passage, "") == "cell" else None


def _distances(layout: np.ndarray, serum_point: int, n_antigens: int) -> np.ndarray:
    """Distance from the serum to each antigen; NaN where either has no coordinates.

    Summed as ae's compiled distance2 sums (each square added by a fused multiply-add), not as
    numpy would: antigens at one position give tied radii, and a last-bit difference in their
    distances changes which radii tie, so the empirical radius moved by 1.5e-5 on one serum of
    the Sep 2026 round until the sums matched.
    """
    sx, sy = layout[serum_point][:2]
    out = np.full(n_antigens, np.nan)
    for a in range(n_antigens):
        dx, dy = layout[a, 0] - sx, layout[a, 1] - sy
        out[a] = math.sqrt(_fma(dy, dy, _fma(dx, dx, 0.0)))
    return out


def _fma(a: float, b: float, c: float) -> float:
    """a * b + c rounded once (math.fma from Python 3.13; exact rationals before it)."""
    if math.isnan(a) or math.isnan(b) or math.isnan(c):
        return math.nan
    fma = getattr(math, "fma", None)
    if fma is not None:
        return float(fma(a, b, c))
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _similarity(t: Titre) -> float:
    """ae ``Titer::logged_for_column_bases``: regular and ``<`` at their log, ``>`` one dilution
    up, and -1 for a dodgy or missing titre (so such a titre protects nothing)."""
    v = t.logged_for_column_bases()
    return -1.0 if v is None else v


def _homologous(
    antigen: Antigen,
    ag: int,
    titres: Sequence[Titre],
    distances: np.ndarray,
    serum_has_xy: bool,
    column_basis: float,
    fold: float,
) -> HomologousAntigen:
    """ae ``set_theoretical`` then ``set_empirical``: the status is the empirical search's,
    so a ``<`` homologous titre is "good" with an empirical radius and no theoretical one."""
    titre = titres[ag]
    theoretical = None
    if titre.type == TitreType.REGULAR:
        theoretical = fold + column_basis - _similarity(titre)
    if not serum_has_xy:
        empirical, status = None, "serum disconnected"
    elif math.isnan(distances[ag]):
        empirical, status = None, "antigen disconnected"
    elif titre.is_missing:
        empirical, status = None, "non-regular homologous titre"
    else:
        empirical = _empirical(_similarity(titre), titres, distances, column_basis, fold)
        status = "good" if empirical is not None else "titre too low"
    return HomologousAntigen(ag, antigen.designation(), str(titre), theoretical, empirical, status)


def _empirical(
    homologous: float,
    titres: Sequence[Titre],
    distances: np.ndarray,
    column_basis: float,
    fold: float,
) -> float | None:
    """ae ``set_empirical``: try a radius at the nearest titrated antigen, then midway between
    each next pair outwards, and keep the radius that leaves fewest antigens on the wrong side
    (protected by titre but outside, or unprotected but inside); ties are averaged. None when
    the protection boundary is below 1 (ae: titre too low)."""
    boundary = min(column_basis, homologous) - fold
    if boundary < 1.0:
        return None
    points = sorted(
        (float(distances[a]), _protected(t, column_basis, boundary))
        for a, t in enumerate(titres)
        if not t.is_missing and not math.isnan(distances[a])
    )
    best: int | None = None
    total, count = 0.0, 0
    previous: float | None = None
    for d, _ in points:
        radius = d if previous is None else (d + previous) / 2.0
        wrong = sum(1 for e, prot in points if prot != (e <= radius))
        if best is None or wrong < best:
            best, total, count = wrong, radius, 1
        elif wrong == best:
            total, count = total + radius, count + 1
        previous = d
    return total / count if count else None


def _protected(t: Titre, column_basis: float, boundary: float) -> bool:
    final = min(column_basis, _similarity(t))
    return final >= boundary if t.type == TitreType.REGULAR else final > boundary


def _first_min(entries: Sequence[HomologousAntigen], field: str) -> HomologousAntigen | None:
    """The first entry (index order) with the smallest ``field``, as ae's min_element with
    a missing radius ordered last; None if no entry has one."""
    found: HomologousAntigen | None = None
    for e in entries:
        v = getattr(e, field)
        if v is not None and (found is None or v < getattr(found, field)):
            found = e
    return found


def _circle(rule: str, entries: Sequence[HomologousAntigen], has_homologous: bool) -> Circle:
    if not has_homologous:
        return Circle(rule, None, None, None, None, None, None, False, "no_homologous")
    e = _first_min(entries, "empirical")
    t = _first_min(entries, "theoretical")
    e_radius = None if e is None or e.empirical is None else max(e.empirical, MIN_RADIUS)
    t_radius = None if t is None or t.theoretical is None else max(t.theoretical, MIN_RADIUS)
    floored = (e is not None and e.empirical is not None and e.empirical < MIN_RADIUS) or (
        t is not None and t.theoretical is not None and t.theoretical < MIN_RADIUS
    )
    status = "good" if e_radius is not None or t_radius is not None else "no_radius"
    return Circle(
        rule,
        e_radius,
        None if e is None else e.antigen,
        None if e is None else e.titre,
        t_radius,
        None if t is None else t.antigen,
        None if t is None else t.titre,
        floored,
        status,
    )


def _radius_differs(a: float | None, b: float | None) -> bool:
    return (a is None) != (b is None) or (a is not None and b is not None and abs(a - b) > 1e-9)


def _coverage(rule: str, ag: int | None, titres: Sequence[Titre], fold: float) -> Coverage:
    """ae ``serum_coverage``: within if the titre (as it counts for a column basis) is at least
    the homologous titre's log less fold, outside if below; a missing or dodgy titre is in
    neither and counted in ``dont_care``. ae refuses a non-regular homologous titre and a
    threshold at or below 0."""
    dont_care = sum(1 for t in titres if _similarity(t) < 0)
    if ag is None:
        return Coverage(rule, None, None, None, (), (), dont_care, "no_homologous_titre")
    homologous = titres[ag]
    if homologous.type != TitreType.REGULAR:
        return Coverage(rule, ag, str(homologous), None, (), (), dont_care, "non_regular")
    threshold = homologous.logged() - fold
    if threshold <= 0:
        return Coverage(rule, ag, str(homologous), threshold, (), (), dont_care, "too_low")
    within = tuple(a for a, t in enumerate(titres) if _similarity(t) >= threshold)
    outside = tuple(a for a, t in enumerate(titres) if 0 <= _similarity(t) < threshold)
    return Coverage(rule, ag, str(homologous), threshold, within, outside, dont_care, "good")
