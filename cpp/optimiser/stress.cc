#include "stress.hh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace af::map
{
    namespace
    {
        using terms::less_than_error;
        using terms::less_than_error_slope;
        using terms::nonzero;

        // Adds `scale * (x1 - x2)` to the gradient of p2 and subtracts it from p1.
        void push_apart(std::span<const double> layout, std::span<double> gradient, std::size_t dimensions, std::size_t p1, std::size_t p2, double scale)
        {
            const double* x1 = layout.data() + p1 * dimensions;
            const double* x2 = layout.data() + p2 * dimensions;
            double* g1 = gradient.data() + p1 * dimensions;
            double* g2 = gradient.data() + p2 * dimensions;
            for (std::size_t dim = 0; dim < dimensions; ++dim) {
                const double inc = scale * (x1[dim] - x2[dim]);
                g1[dim] -= inc;
                g2[dim] += inc;
            }
        }
    } // namespace

    double map_distance(std::span<const double> layout, std::size_t dimensions, std::size_t p1, std::size_t p2)
    {
        const double* x1 = layout.data() + p1 * dimensions;
        const double* x2 = layout.data() + p2 * dimensions;
        double sum = 0.0;
        for (std::size_t dim = 0; dim < dimensions; ++dim) {
            const double d = x1[dim] - x2[dim];
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    Stress::Stress(TableDistances distances, std::size_t n_points, std::vector<char> unmovable)
        : distances_{std::move(distances)}, n_points_{n_points}, unmovable_{std::move(unmovable)}, by_point_(n_points)
    {
        for (const auto& entry : distances_.regular) {
            by_point_[entry.p1].push_back({entry.p2, entry.distance, entry.weight, false});
            by_point_[entry.p2].push_back({entry.p1, entry.distance, entry.weight, false});
        }
        for (const auto& entry : distances_.less_than) {
            by_point_[entry.p1].push_back({entry.p2, entry.distance, entry.weight, true});
            by_point_[entry.p2].push_back({entry.p1, entry.distance, entry.weight, true});
        }
    }

    double Stress::value(std::span<const double> layout, std::size_t dimensions) const
    {
        double result = 0.0;
        for (const auto& entry : distances_.regular) {
            const double diff = entry.distance - map_distance(layout, dimensions, entry.p1, entry.p2);
            result += entry.weight * diff * diff;
        }
        for (const auto& entry : distances_.less_than) {
            const double diff = entry.distance - map_distance(layout, dimensions, entry.p1, entry.p2) + 1.0;
            result += entry.weight * less_than_error(diff);
        }
        return result;
    }

    double Stress::value_and_gradient(std::span<const double> layout, std::size_t dimensions, std::span<double> gradient) const
    {
        std::fill(gradient.begin(), gradient.end(), 0.0);
        double result = 0.0;
        for (const auto& entry : distances_.regular) {
            const double dist = map_distance(layout, dimensions, entry.p1, entry.p2);
            const double diff = entry.distance - dist;
            result += entry.weight * diff * diff;
            push_apart(layout, gradient, dimensions, entry.p1, entry.p2, entry.weight * 2.0 * diff / nonzero(dist));
        }
        for (const auto& entry : distances_.less_than) {
            const double dist = map_distance(layout, dimensions, entry.p1, entry.p2);
            const double diff = entry.distance - dist + 1.0;
            result += entry.weight * less_than_error(diff);
            push_apart(layout, gradient, dimensions, entry.p1, entry.p2, entry.weight * less_than_error_slope(diff) / nonzero(dist));
        }
        if (!unmovable_.empty()) {
            for (std::size_t point = 0; point < n_points_; ++point) {
                if (unmovable_[point])
                    std::fill_n(gradient.begin() + static_cast<std::ptrdiff_t>(point * dimensions), dimensions, 0.0);
            }
        }
        return result;
    }

    std::vector<double> Stress::titre_terms(std::span<const double> layout, std::size_t dimensions, std::size_t n_antigens, std::size_t n_sera) const
    {
        std::vector<double> result(n_antigens * n_sera, std::numeric_limits<double>::quiet_NaN());
        // an entry's p1 is the antigen, p2 is n_antigens + serum (table_distances())
        const auto cell = [n_antigens, n_sera](const TableDistance& entry) { return entry.p1 * n_sera + (entry.p2 - n_antigens); };
        for (const auto& entry : distances_.regular) {
            const double diff = entry.distance - map_distance(layout, dimensions, entry.p1, entry.p2);
            result[cell(entry)] = entry.weight * diff * diff;
        }
        for (const auto& entry : distances_.less_than)
            result[cell(entry)] = entry.weight * less_than_error(entry.distance - map_distance(layout, dimensions, entry.p1, entry.p2) + 1.0);
        return result;
    }

    double Stress::point_contribution(std::size_t point, std::span<const double> layout, std::size_t dimensions) const
    {
        double result = 0.0;
        for (const auto& entry : by_point_[point]) {
            const double dist = map_distance(layout, dimensions, point, entry.other);
            if (entry.less_than)
                result += entry.weight * less_than_error(entry.distance - dist + 1.0);
            else
                result += entry.weight * (entry.distance - dist) * (entry.distance - dist);
        }
        return result;
    }

} // namespace af::map
