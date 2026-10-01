"""Chart.select: removing points keeps the maps, and nothing ends up on another point."""

import numpy as np
import pytest

from af.chart.ace import read_chart, write_chart
from af.chart.model import Antigen, Chart, Projection, Serum, Titres, empty_table
from af.chart.titre import Titre

N_AG, N_SR = 4, 3
N = N_AG + N_SR


def tagged_chart() -> Chart:
    """Every per-point field carries the point's own number, so a misattached value shows."""
    table = empty_table(N_AG, N_SR)
    for i in range(N_AG):
        for j in range(N_SR):
            table[i][j] = Titre.parse(str(10 * 2 ** (i + j)))
    layers = [
        {(i, j): table[i][j] for i in range(N_AG) for j in range(N_SR) if (i + j) % 2 == 0},
        {(i, j): table[i][j] for i in range(N_AG) for j in range(N_SR) if (i + j) % 2 == 1},
    ]

    def projection(k: int) -> Projection:
        return Projection(
            layout=np.array([[p, 10.0 * p + k] for p in range(N)], dtype=float),
            stress=123.4,
            forced_column_bases=np.array([7.0 + j for j in range(N_SR)]),
            transformation=np.array([[0.0, 1.0], [-1.0, 0.0]]),
            disconnected=(1, N_AG + 2),
            unmovable=(0, 3, N_AG + 1),
            avidity_adjusts=np.array([100.0 + p for p in range(N)]),
            gradient_multipliers=np.array([200.0 + p for p in range(N)]),
            extra={"u": [2, N_AG], "e": 0.0001},
        )

    return Chart(
        info={"D": "20210115"},
        antigens=[Antigen(f"TEST-A{i}", passage="MDCK1") for i in range(N_AG)],
        sera=[Serum(f"TEST-S{j}", serum_id=f"S{j}", extra={"h": [j, j + 1]}) for j in range(N_SR)],
        titres=Titres(table, layers),
        forced_column_bases=np.array([5.0 + j for j in range(N_SR)]),
        projections=[projection(0), projection(1)],
        extra={
            "p": {
                "p": [p % 3 for p in range(N)],
                "d": list(range(N))[::-1],
                "s": [0, N_AG],
                "P": [{}, {}, {}],
            },
            "R": {"style": {"A": [{"T": {"!i": 3}, "A": 1}, {"T": {"!i": 2}, "A": 0}]}},
        },
    )


KEEP_AG, KEEP_SR = [0, 2, 3], [0, 2]
POINTS = KEEP_AG + [N_AG + j for j in KEEP_SR]  # old point index of each new point


def arr(a: np.ndarray | None) -> np.ndarray:
    assert a is not None
    return a


def test_every_per_point_value_follows_its_point():
    old, new = tagged_chart(), tagged_chart().select(KEEP_AG, KEEP_SR)
    assert [a.name for a in new.antigens] == ["TEST-A0", "TEST-A2", "TEST-A3"]
    assert [s.name for s in new.sera] == ["TEST-S0", "TEST-S2"]
    for p_old, p_new in zip(old.projections, new.projections, strict=True):
        assert np.array_equal(p_new.layout, p_old.layout[POINTS])
        assert np.array_equal(arr(p_new.avidity_adjusts), arr(p_old.avidity_adjusts)[POINTS])
        assert np.array_equal(
            arr(p_new.gradient_multipliers), arr(p_old.gradient_multipliers)[POINTS]
        )
        assert np.array_equal(
            arr(p_new.forced_column_bases), arr(p_old.forced_column_bases)[KEEP_SR]
        )
        assert np.array_equal(arr(p_new.transformation), arr(p_old.transformation))
        assert p_new.stress is None  # stale for the new point set
        # index lists name the same points as before, renumbered; removed ones dropped
        for field in ("disconnected", "unmovable"):
            kept_old = [x for x in getattr(p_old, field) if x in POINTS]
            assert [POINTS[x] for x in getattr(p_new, field)] == kept_old
        assert [POINTS[x] for x in p_new.extra["u"]] == [x for x in p_old.extra["u"] if x in POINTS]
        assert p_new.extra["e"] == p_old.extra["e"]
    assert np.array_equal(arr(new.forced_column_bases), arr(old.forced_column_bases)[KEEP_SR])
    # titres: table and layers, cell by cell
    for a, i in enumerate(KEEP_AG):
        for b, j in enumerate(KEEP_SR):
            assert new.titres.table[a][b] == old.titres.table[i][j]
            for L_new, L_old in zip(new.titres.layers, old.titres.layers, strict=True):
                assert L_new.get((a, b)) == L_old.get((i, j))
    # homologous antigens: antigen numbers, renumbered, removed ones dropped
    # (S0: [0, 1] -> [0]; S2: [2, 3] -> [1, 2])
    assert [s.extra["h"] for s in new.sera] == [[0], [1, 2]]
    # plot spec: each point keeps its style; drawing order and shown-on-all renumbered
    plot_old, plot_new = old.extra["p"], new.extra["p"]
    assert plot_new["p"] == [plot_old["p"][x] for x in POINTS]
    assert [POINTS[x] for x in plot_new["d"]] == [x for x in plot_old["d"] if x in POINTS]
    assert [POINTS[x] for x in plot_new["s"]] == [0, N_AG]
    assert plot_new["P"] == plot_old["P"]
    # semantic selectors: antigen 3 -> 2, serum 2 -> 1
    sel = new.extra["R"]["style"]["A"]
    assert (sel[0]["T"]["!i"], sel[1]["T"]["!i"]) == (2, 1)


def test_the_original_chart_is_untouched():
    chart = tagged_chart()
    chart.select(KEEP_AG, KEEP_SR)
    assert chart.n_points == N and chart.projections[0].stress == 123.4
    assert chart.sera[0].extra["h"] == [0, 1] and chart.extra["R"]["style"]["A"][0]["T"]["!i"] == 3


def test_selected_chart_is_a_valid_ace(tmp_path):
    new = tagged_chart().select(KEEP_AG, KEEP_SR)
    back = read_chart(write_chart(new, tmp_path / "s.ace"))
    assert back.n_points == len(POINTS)
    assert np.array_equal(back.projections[1].layout, new.projections[1].layout)


@pytest.mark.parametrize(
    ("keep_ag", "keep_sr", "match"),
    [
        ([2, 0], [0], "original order"),
        ([0, 0], [0], "repeated"),
        ([0, N_AG], [0], "out of range"),
        ([0], [N_SR], "out of range"),
    ],
)
def test_bad_selections_are_refused(keep_ag, keep_sr, match):
    with pytest.raises(ValueError, match=match):
        tagged_chart().select(keep_ag, keep_sr)


def test_a_selector_naming_a_removed_point_is_refused():
    with pytest.raises(ValueError, match="removed antigen 3"):
        tagged_chart().select([0, 1, 2], [0, 1, 2])


def test_a_selector_without_antigen_or_serum_is_refused():
    chart = tagged_chart()
    chart.extra["R"]["style"]["A"][0].pop("A")
    with pytest.raises(ValueError, match="without A"):
        chart.select(KEEP_AG, KEEP_SR)
