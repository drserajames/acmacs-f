"""ae's automatic aa-label placer, ported (Geometry.placer = "ae"; "grid" is af's own, place.py).

ae (cc/tal/draw-tree.cc) samples each label's positions around its branch by leader angle and
length, then searches all labels together: a seeded descent from many random starts, pair and
window repairs, and a tidy pass that never adds a conflict. Labels are kept apart glyph line to
glyph line (ae's ink rule), and the same predicates the search avoids are the ones reported.

Modules: scene (inputs, options, candidate arrays), geometry (ae's segment and glyph predicates),
candidates (occupancy grid, exact tree ink, the candidate tiers and their shape cost), pairs (the
pairwise conflict and penetration matrices), search (the seeded schedule), measure (residual
conflicts and metrics). Units are page points, y down; a box is (x0, y0, x1, y1), the text's own
box (ae's: monospace advance wide, lines x 1.18 sizes tall), not af's padded one.
"""
