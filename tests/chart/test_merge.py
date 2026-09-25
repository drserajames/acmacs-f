"""Identity across tables and the layer merge (ae strict matching; lispmds merge)."""

import numpy as np
import pytest

from af.chart.identity import antigen_identity, serum_identity
from af.chart.merge import ColumnBasisConvention, MergeError, MergeOptions, merge
from af.chart.model import Antigen, Chart, Projection, Serum, Titres, empty_table
from af.chart.titre import Titre


def table(antigens, sera, cells, date="20210101", projection=False):
    t = empty_table(len(antigens), len(sera))
    for (i, j), v in cells.items():
        t[i][j] = Titre.parse(v)
    c = Chart({"D": date, "V": "A(H9N2)", "l": "TEST", "A": "HI"}, antigens, sera, Titres(t))
    if projection:
        c.projections = [
            Projection(layout=np.arange(2 * c.n_points, dtype=float).reshape(-1, 2), stress=1.0)
        ]
    return c


def AG(n, p="E3", **k):
    return Antigen(f"TEST-{n}", passage=p, **k)


def SR(n, i):
    return Serum(f"TEST-{n}", serum_id=i)


def test_identity_rules():
    assert antigen_identity("X", "", (), "E3") == ("X", "", (), "E3")
    assert antigen_identity("X", "", (), "") is None  # empty passage never matches
    assert antigen_identity("X", "", ("DISTINCT",), "E3") is None
    assert serum_identity("X", "", (), "") is None
    assert serum_identity("X", "", (), "F1") == ("X", "", (), "F1")


def test_common_points_and_new_points_appended():
    a = table([AG(1), AG(2)], [SR(1, "F1")], {(0, 0): "40", (1, 0): "80"}, projection=True)
    b = table(
        [AG(2), AG(3), AG(4, p="")],
        [SR(1, "F1"), SR(2, "F2")],
        {(0, 0): "160", (1, 1): "20", (2, 0): "10"},
        date="20210201",
    )
    m, rep = merge(a, b)
    assert [x.name for x in m.antigens] == [AG(n).name for n in (1, 2, 3, 4)]
    assert m.n_sera == 2 and rep.common_antigens == 1 and rep.common_sera == 1
    assert str(m.titres.table[1][0]) == "113"  # 80 and 160
    assert len(m.titres.layers) == 2
    lay = m.projections[
        0
    ].layout  # incremental: old points keep their coordinates, new ones are NaN
    assert lay[0].tolist() == [0.0, 1.0] and np.isnan(lay[2]).all()
    assert lay[4].tolist() == [4.0, 5.0] and np.isnan(lay[5]).all()
    # An empty-passage antigen never matches, so merging it again makes a duplicate, which
    # ae rejects (throw_if_duplicates on the merge) and so do we.
    with pytest.raises(MergeError):
        merge(m, b)


def test_distinct_duplicates_allowed_plain_duplicates_rejected():
    ok = table([AG(1), AG(1, annotations=("DISTINCT",))], [SR(1, "F1")], {})
    merge(ok, ok)
    bad = table([AG(1), AG(1)], [SR(1, "F1")], {})
    with pytest.raises(MergeError):
        merge(bad, ok)


def test_forced_column_bases_from_discarded_more_than():
    a = table([AG(1), AG(2)], [SR(1, "F1")], {(0, 0): "40", (1, 0): ">5120"})
    b = table([AG(1), AG(3)], [SR(1, "F1")], {(0, 0): "80", (1, 0): "40"}, date="20210201")
    m, rep = merge(a, b)
    assert str(m.titres.table[1][0]) == "*"
    assert m.forced_column_bases is not None and m.forced_column_bases.tolist() == [10.0]
    assert rep.column_basis_slack == {0: pytest.approx(10.0 - np.log2(5.7))}
    m2, _ = merge(a, b, MergeOptions(column_bases=ColumnBasisConvention.TABLE_ONLY))
    assert m2.forced_column_bases is None


def test_cheating_assay_merges_only_test_antigens():
    ref = [AG(1), AG(2)]
    sera = [SR(1, "F1"), SR(2, "F2")]
    cells = {(0, 0): "640", (0, 1): "80", (1, 0): "40", (1, 1): "320"}
    a = table(ref + [AG(5)], sera, {**cells, (2, 0): "20"})
    b = table(ref + [AG(6)], sera, {**cells, (2, 1): "160"}, date="20210201")
    m, rep = merge(a, b, MergeOptions(combine_cheating_assays=True))
    assert rep.cheating_assay and rep.skipped_reference_antigens == 2
    assert not any(ag < 2 for ag, _ in m.titres.layers[1])  # no reference titres in the new layer
    assert str(m.titres.table[0][0]) == "640"
    _, plain = merge(a, b)
    assert not plain.cheating_assay
