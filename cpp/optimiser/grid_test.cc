#include "grid_test.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "minimise.hh"
#include "stress.hh"

namespace af::map
{
    namespace
    {
        constexpr double hemisphering_distance = 1.0;  // acmacs-c2 hemi-local test
        constexpr double stress_threshold = 0.25;      // acmacs-c2 hemi-local test
        constexpr double hemisphering_search_margin = 2.0 * stress_threshold;

        double distance(std::span<const double> a, std::span<const double> b)
        {
            double sum = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i)
                sum += (a[i] - b[i]) * (a[i] - b[i]);
            return std::sqrt(sum);
        }

        // Bounding box of every partner's position +- its table distance.
        void search_area(const Stress& stress, std::span<const double> layout, std::size_t dimensions, std::size_t point, std::vector<double>& low, std::vector<double>& high)
        {
            low.assign(dimensions, std::numeric_limits<double>::infinity());
            high.assign(dimensions, -std::numeric_limits<double>::infinity());
            for (const auto& entry : stress.entries_for(point)) {
                for (std::size_t dim = 0; dim < dimensions; ++dim) {
                    const double centre = layout[entry.other * dimensions + dim];
                    low[dim] = std::min(low[dim], centre - entry.distance);
                    high[dim] = std::max(high[dim], centre + entry.distance);
                }
            }
        }

        // Visits every grid cell low + k * step (k >= 0) that is <= high in each dimension.
        template <typename Visit> void for_each_cell(const std::vector<double>& low, const std::vector<double>& high, double step, std::vector<double>& cell, Visit visit)
        {
            const std::size_t dimensions = low.size();
            std::vector<std::size_t> index(dimensions, 0), count(dimensions);
            for (std::size_t dim = 0; dim < dimensions; ++dim)
                count[dim] = static_cast<std::size_t>(std::floor((high[dim] - low[dim]) / step)) + 1;
            cell.resize(dimensions);
            while (true) {
                for (std::size_t dim = 0; dim < dimensions; ++dim)
                    cell[dim] = low[dim] + static_cast<double>(index[dim]) * step;
                visit(cell);
                std::size_t dim = 0;
                for (; dim < dimensions; ++dim) {
                    if (++index[dim] < count[dim])
                        break;
                    index[dim] = 0;
                }
                if (dim == dimensions)
                    return;
            }
        }

        GridResult test_point(const Problem& problem, const Stress& stress, std::span<const double> layout, std::size_t dimensions, double layout_stress, std::size_t point, double step)
        {
            GridResult result{point, Diagnosis::excluded, {}, 0.0, 0.0};
            if (problem.disconnected[point] || problem.is_unmovable(point) || !stress.has_entries(point))
                return result;
            result.diagnosis = Diagnosis::normal;

            std::vector<double> trial(layout.begin(), layout.end());
            const auto row = [&trial, dimensions, point]() { return std::span<double>{trial.data() + point * dimensions, dimensions}; };
            const std::vector<double> original(layout.begin() + static_cast<std::ptrdiff_t>(point * dimensions), layout.begin() + static_cast<std::ptrdiff_t>((point + 1) * dimensions));

            const double current = stress.point_contribution(point, trial, dimensions);
            double best = current;
            double hemisphering = current + hemisphering_search_margin;
            std::vector<double> best_cell, hemisphering_cell, low, high, cell;
            search_area(stress, layout, dimensions, point, low, high);
            for_each_cell(low, high, step, cell, [&](const std::vector<double>& at) {
                std::copy(at.begin(), at.end(), row().begin());
                const double contribution = stress.point_contribution(point, trial, dimensions);
                if (contribution < best) {
                    best = contribution;
                    best_cell = at;
                }
                else if (best_cell.empty() && contribution < hemisphering && distance(original, at) > hemisphering_distance) {
                    hemisphering = contribution;
                    hemisphering_cell = at;
                }
            });

            const auto move_and_minimise = [&](const std::vector<double>& to, Precision precision) {
                std::copy(original.begin(), original.end(), row().begin());
                std::copy(to.begin(), to.end(), row().begin());
                return minimise(stress, trial, dimensions, problem.disconnected, Method::cg, precision);
            };

            if (!best_cell.empty()) {
                const auto minimised = move_and_minimise(best_cell, Precision::rough);
                result.position.assign(row().begin(), row().end());
                result.distance = distance(original, result.position);
                result.stress_diff = minimised.stress - layout_stress;
                // As ae: trapped when the stress changes by more than 0.25 either way. A point
                // whose better cell ends in a *worse* map can still plausibly sit in either
                // place, so it is reported (Sarah, 25 Sep 2026); only moves that lower the
                // stress are applied (resolve_trapped).
                result.diagnosis = std::abs(result.stress_diff) > stress_threshold ? Diagnosis::trapped : Diagnosis::hemisphering;
            }
            else if (!hemisphering_cell.empty()) {
                auto minimised = move_and_minimise(hemisphering_cell, Precision::rough);
                result.position.assign(row().begin(), row().end());
                result.distance = distance(original, result.position);
                if (result.distance > hemisphering_distance && result.distance < hemisphering_distance * 1.2) {
                    minimised = minimise(stress, trial, dimensions, problem.disconnected, Method::cg, Precision::fine);
                    result.position.assign(row().begin(), row().end());
                    result.distance = distance(original, result.position);
                }
                result.stress_diff = minimised.stress - layout_stress;
                if (result.distance > hemisphering_distance && std::abs(result.stress_diff) < stress_threshold)
                    result.diagnosis = Diagnosis::hemisphering;
            }
            return result;
        }
    } // namespace

    std::vector<GridResult> grid_test(const Problem& problem, std::span<const double> layout, std::size_t dimensions, double step, int threads)
    {
        problem.validate();
        if (layout.size() != problem.n_points() * dimensions)
            throw std::invalid_argument{"grid_test: layout must have n_points rows of `dimensions` columns"};
        if (!(step > 0.0))
            throw std::invalid_argument{"grid_test: step must be positive"};

        const Stress stress{table_distances(problem), problem.n_points(), problem.unmovable};
        std::vector<double> zeroed(layout.begin(), layout.end());
        for (std::size_t point = 0; point < problem.n_points(); ++point) {
            if (problem.disconnected[point])
                std::fill_n(zeroed.begin() + static_cast<std::ptrdiff_t>(point * dimensions), dimensions, 0.0);
        }
        for (const double v : zeroed) {
            if (!std::isfinite(v))
                throw std::invalid_argument{"grid_test: a connected point has no coordinates"};
        }
        const double layout_stress = stress.value(zeroed, dimensions);

        std::vector<GridResult> results(problem.n_points());
        std::vector<std::string> errors(problem.n_points());
#ifdef _OPENMP
        const int n_threads = threads > 0 ? threads : omp_get_max_threads();
#else
        (void)threads;
#endif
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1)
        for (std::size_t point = 0; point < problem.n_points(); ++point) {
            try {
                results[point] = test_point(problem, stress, zeroed, dimensions, layout_stress, point, step);
            }
            catch (std::exception& err) {
                errors[point] = err.what();
            }
        }
        for (std::size_t point = 0; point < problem.n_points(); ++point) {
            if (!errors[point].empty())
                throw std::runtime_error{"grid_test point " + std::to_string(point) + ": " + errors[point]};
        }
        return results;
    }

} // namespace af::map
