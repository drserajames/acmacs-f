"""Skipping far-apart label pairs in placement changes nothing: same boxes, same costs."""

import random

import pytest

pytest.importorskip("numpy", reason="numpy not installed: af.tree.draw needs it (pyproject, WS1)")

from af.tree.draw import place  # noqa: E402


def crowded(seed, n=60):
    """Invented labels on a small page, so many pairs are near and many are not."""
    rng = random.Random(seed)
    grid = place.Grid(300, 400)
    for _ in range(80):  # invented tree ink: branches and stems
        x, y = rng.uniform(0, 300), rng.uniform(0, 400)
        grid.add_hline(x, x + rng.uniform(5, 60), y)
        grid.add_vline(x, y, y + rng.uniform(5, 40))
    items = [
        (k, f"{'ABCD'[k % 4]}{k}{'EFGH'[k % 4]}", (rng.uniform(80, 300), rng.uniform(10, 390)))
        for k in range(n)
    ]
    corners = [(rng.uniform(0, 250), y) for y in (50.0, 200.0)]
    obstacles = [(x, y, x + 40, y + 10) for x, y in corners]
    return items, grid, obstacles


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_pruned_placement_equals_checking_every_pair(seed, monkeypatch):
    items, grid, obstacles = crowded(seed)

    def run():
        return place.place_labels(
            items, grid, lambda t: 5.0 * len(t), 7.0, obstacles, search=place.Search(top=0)
        )

    pruned = run()
    reaches = [place.reach(p.box, p.anchor) for p in pruned]
    pairs = [place.near(a, b) for i, a in enumerate(reaches) for b in reaches[i + 1 :]]
    assert any(pairs) and not all(pairs)  # both branches taken: near pairs and skipped ones
    monkeypatch.setattr(place, "near", lambda *a, **k: True)
    every_pair = run()
    assert [(p.key, p.box, p.cost) for p in pruned] == [(p.key, p.box, p.cost) for p in every_pair]
