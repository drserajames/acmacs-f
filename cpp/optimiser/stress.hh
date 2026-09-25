// Map stress and its analytic gradient.
//
// For a regular titre with target distance t and map distance D the error is
//     w * (t - D)^2
// For a "<" titre the target is only a lower bound: the map distance should be at least
// t + 1 (one dilution further than the threshold implies). The error is
//     w * (t + 1 - D)^2 * sigmoid(10 * (t + 1 - D))
// which is ~0 when D is comfortably above t + 1 and ~squared error when D is too short.
// This is the lispmds/AD/ae formula (ae stress.cc, SigmoidMutiplier = 10).

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "problem.hh"

namespace af::map
{
    class Stress
    {
      public:
        Stress(TableDistances distances, std::size_t n_points, std::vector<char> unmovable = {});

        double value(std::span<const double> layout, std::size_t dimensions) const;

        // Writes d(stress)/d(layout) into `gradient` (same shape as `layout`) and returns the
        // stress. Unmovable points get a zero gradient, so no minimiser moves them.
        double value_and_gradient(std::span<const double> layout, std::size_t dimensions, std::span<double> gradient) const;

        // The part of the stress that involves `point`: used by the grid test.
        double point_contribution(std::size_t point, std::span<const double> layout, std::size_t dimensions) const;

        const TableDistances& distances() const { return distances_; }
        std::size_t n_points() const { return n_points_; }
        bool has_entries(std::size_t point) const { return !by_point_[point].empty(); }

        // The table distances that involve `point`, as (other point, distance, weight, is "<").
        struct PointEntry
        {
            std::size_t other;
            double distance;
            double weight;
            bool less_than;
        };
        const std::vector<PointEntry>& entries_for(std::size_t point) const { return by_point_[point]; }

      private:
        TableDistances distances_;
        std::size_t n_points_;
        std::vector<char> unmovable_;
        std::vector<std::vector<PointEntry>> by_point_;
    };

    double map_distance(std::span<const double> layout, std::size_t dimensions, std::size_t p1, std::size_t p2);

} // namespace af::map
