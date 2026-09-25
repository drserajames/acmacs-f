"""Report tree figures: layout, automatic clade sections, aa-transition labels, rendering.

Entry point: :func:`af.tree.draw.figure.make_figure`. The layout (:mod:`.layout`) is a separate
result so signature pages can reuse it. Curation that used to be hand-edited in `.tal` files is
derived here; overrides are named data and counted in the draw report.

Rule thresholds (which clades are "very small", how many aa labels) live in ``defaults.toml``
with their reasons. To change one, copy that file, edit it, and load it with
``af.tree.draw.defaults.load_defaults(path)``; to show or hide a single clade, use
``Overrides.show_clades`` / ``Overrides.hide_clades``.
"""
