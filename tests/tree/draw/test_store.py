"""Reading a tree-store version (I6) for the figure."""

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it")

from af.tree.draw.store import StoreInputError, StoreTree  # noqa: E402

from .synthetic import PARENTS, standard_tree  # noqa: E402


def test_centre_leaves_and_unknown_centre():
    titrated = {"EPI_ISL_0000001": ["LAB1"], "EPI_ISL_0000002": ["LAB1", "LAB2"]}
    st = StoreTree(standard_tree(), dict(PARENTS), "v1", titrated, {}, {})
    assert st.centre_leaves("LAB2") == frozenset({"EPI_ISL_0000002"})
    with pytest.raises(StoreInputError, match="LAB9"):
        st.centre_leaves("LAB9")
