// Grid test: is any point stuck in a worse place than it could be?
//
// For each point, scan a grid over the area around its titrated partners and evaluate
// that point's own stress contribution at every cell. If a cell is better, move the point
// there and re-minimise the whole map roughly:
//   - trapped:      the map's stress drops by more than 0.25;
//   - hemisphering: the point has another position more than 1 unit away where the map's
//                   stress is within 0.25 of the current one (the position is ambiguous).
// Follows ae grid-test.cc; one deliberate difference is noted in grid_test.cc.

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "problem.hh"

namespace af::map
{
    enum class Diagnosis { excluded, normal, trapped, hemisphering };

    struct GridResult
    {
        std::size_t point;
        Diagnosis diagnosis{Diagnosis::excluded};
        std::vector<double> position; // better position found (empty if none)
        double distance{0.0};         // from the current position to `position`
        double stress_diff{0.0};      // map stress after moving minus before (negative = better)
    };

    // One result per point. Disconnected and unmovable points, and points with no table
    // distances, are `excluded`.
    std::vector<GridResult> grid_test(const Problem& problem, std::span<const double> layout, std::size_t dimensions, double step, int threads);

} // namespace af::map
