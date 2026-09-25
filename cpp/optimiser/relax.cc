#include "relax.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "random.hh"
#include "stress.hh"

namespace af::map
{
    namespace
    {
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();
        constexpr std::size_t annealing_start_dimensions = 5;
        constexpr std::size_t incremental_refined = 5; // ae relax_incremental re-relaxes the best five

        bool row_is_nan(std::span<const double> layout, std::size_t dimensions, std::size_t point)
        {
            return std::all_of(layout.begin() + static_cast<std::ptrdiff_t>(point * dimensions), layout.begin() + static_cast<std::ptrdiff_t>((point + 1) * dimensions),
                               [](double v) { return std::isnan(v); });
        }

        void randomise_point(std::span<double> layout, std::size_t dimensions, std::size_t point, SplitMix64& rng, double diameter)
        {
            for (std::size_t dim = 0; dim < dimensions; ++dim)
                layout[point * dimensions + dim] = rng.uniform(-diameter / 2.0, diameter / 2.0);
        }

        double bounding_box_diagonal(std::span<const double> layout, std::size_t dimensions, const std::vector<char>& disconnected)
        {
            double sum = 0.0;
            for (std::size_t dim = 0; dim < dimensions; ++dim) {
                double low = std::numeric_limits<double>::infinity(), high = -low;
                for (std::size_t point = 0; point < disconnected.size(); ++point) {
                    if (!disconnected[point]) {
                        low = std::min(low, layout[point * dimensions + dim]);
                        high = std::max(high, layout[point * dimensions + dim]);
                    }
                }
                sum += (high - low) * (high - low);
            }
            return std::sqrt(sum);
        }

        void require_enough_connected(const Problem& problem)
        {
            if (problem.n_connected() < 3)
                throw std::invalid_argument{"cannot relax: fewer than 3 connected points"};
        }

        int thread_count(int requested)
        {
#ifdef _OPENMP
            return requested > 0 ? requested : omp_get_max_threads();
#else
            (void)requested;
            return 1;
#endif
        }

        // Stable order: stress ascending, NaN last, ties by start index, so the result does
        // not depend on which thread finished first.
        void sort_by_stress(std::vector<Projection>& projections)
        {
            std::sort(projections.begin(), projections.end(), [](const Projection& a, const Projection& b) {
                const bool a_nan = std::isnan(a.stress), b_nan = std::isnan(b.stress);
                if (a_nan != b_nan)
                    return b_nan;
                if (!a_nan && a.stress != b.stress)
                    return a.stress < b.stress;
                return a.start_index < b.start_index;
            });
        }

        void keep_best(std::vector<Projection>& projections, std::size_t keep)
        {
            if (keep > 0 && projections.size() > keep)
                projections.resize(keep);
        }

        // Runs `body(i)` for every start in parallel. An exception in any start is re-thrown
        // after the loop (the first by start index), because an exception may not leave an
        // OpenMP region.
        template <typename Body> void for_each_start(std::size_t n, int threads, Body body)
        {
            std::vector<std::string> errors(n);
            [[maybe_unused]] const int n_threads = thread_count(threads);
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1)
            for (std::size_t i = 0; i < n; ++i) {
                try {
                    body(i);
                }
                catch (std::exception& err) {
                    errors[i] = err.what();
                }
            }
            for (std::size_t i = 0; i < n; ++i) {
                if (!errors[i].empty())
                    throw std::runtime_error{"start " + std::to_string(i) + ": " + errors[i]};
            }
        }
    } // namespace

    double randomisation_diameter(const Problem& problem, std::size_t dimensions, std::uint64_t seed, double multiplier)
    {
        const double max_distance = max_table_distance(problem);
        if (!(max_distance > 0.0) || !std::isfinite(max_distance))
            throw std::invalid_argument{"cannot size random layouts: the largest table distance is " + std::to_string(max_distance)};

        // The sizing map ignores unmovable points: it only measures how big a map is.
        const Stress stress{table_distances(problem), problem.n_points()};
        SplitMix64 rng{seed};
        std::vector<double> layout(problem.n_points() * dimensions);
        for (std::size_t point = 0; point < problem.n_points(); ++point)
            randomise_point(layout, dimensions, point, rng, max_distance);
        minimise(stress, layout, dimensions, problem.disconnected, Method::cg, Precision::very_rough);

        const double diameter = bounding_box_diagonal(layout, dimensions, problem.disconnected);
        if (!(diameter > 0.0) || !std::isfinite(diameter))
            throw std::runtime_error{"sample optimisation gave a map of diameter " + std::to_string(diameter)};
        return diameter * multiplier;
    }

    std::vector<Projection> relax(const Problem& problem, const RelaxOptions& options)
    {
        problem.validate();
        require_enough_connected(problem);
        if (problem.has_unmovable())
            throw std::invalid_argument{"relax from scratch cannot have unmovable points; give a start layout"};
        if (options.dimensions < 1 || options.n_starts < 1)
            throw std::invalid_argument{"relax needs at least 1 dimension and 1 start"};

        const bool anneal = options.dimension_annealing && options.dimensions < annealing_start_dimensions;
        const std::size_t start_dimensions = anneal ? annealing_start_dimensions : options.dimensions;
        const Stress stress{table_distances(problem), problem.n_points()};
        const double diameter = randomisation_diameter(problem, start_dimensions, sample_seed(options.seed), options.randomisation_diameter_multiplier);

        std::vector<Projection> projections(options.n_starts);
        for_each_start(options.n_starts, options.threads, [&](std::size_t i) {
            const auto seed = start_seed(options.seed, i);
            SplitMix64 rng{seed};
            std::vector<double> layout(problem.n_points() * start_dimensions);
            for (std::size_t point = 0; point < problem.n_points(); ++point)
                randomise_point(layout, start_dimensions, point, rng, diameter);

            Minimised result;
            std::size_t iterations = 0;
            if (anneal) {
                iterations += minimise(stress, layout, start_dimensions, problem.disconnected, options.method, Precision::rough).iterations;
                layout = reduce_dimensions(layout, problem.n_points(), start_dimensions, options.dimensions, problem.disconnected);
            }
            result = minimise(stress, layout, options.dimensions, problem.disconnected, options.method, options.precision);
            iterations += result.iterations;
            projections[i] = Projection{std::move(layout), options.dimensions, result.stress, iterations, seed, i, result.termination};
        });

        sort_by_stress(projections);
        keep_best(projections, options.keep);
        return projections;
    }

    std::vector<Projection> relax_incremental(const Problem& problem, std::span<const double> start_layout, const RelaxOptions& options)
    {
        problem.validate();
        require_enough_connected(problem);
        const std::size_t dimensions = options.dimensions;
        if (start_layout.size() != problem.n_points() * dimensions)
            throw std::invalid_argument{"start_layout must have n_points rows of `dimensions` columns"};
        if (options.dimension_annealing)
            throw std::invalid_argument{"dimension annealing is only for relax from scratch"};

        std::vector<std::size_t> to_randomise;
        for (std::size_t point = 0; point < problem.n_points(); ++point) {
            const bool nan_row = row_is_nan(start_layout, dimensions, point);
            if (!nan_row) {
                for (std::size_t dim = 0; dim < dimensions; ++dim) {
                    if (!std::isfinite(start_layout[point * dimensions + dim]))
                        throw std::invalid_argument{"start_layout: point " + std::to_string(point) + " has some but not all coordinates"};
                }
            }
            if (problem.disconnected[point])
                continue;
            if (nan_row) {
                if (problem.is_unmovable(point))
                    throw std::invalid_argument{"start_layout: unmovable point " + std::to_string(point) + " has no coordinates"};
                to_randomise.push_back(point);
            }
        }

        const Stress stress{table_distances(problem), problem.n_points(), problem.unmovable};
        const double diameter = randomisation_diameter(problem, dimensions, sample_seed(options.seed), options.randomisation_diameter_multiplier);

        std::vector<Projection> projections(options.n_starts);
        for_each_start(options.n_starts, options.threads, [&](std::size_t i) {
            const auto seed = start_seed(options.seed, i);
            SplitMix64 rng{seed};
            std::vector<double> layout(start_layout.begin(), start_layout.end());
            for (std::size_t point = 0; point < problem.n_points(); ++point) {
                if (problem.disconnected[point])
                    std::fill_n(layout.begin() + static_cast<std::ptrdiff_t>(point * dimensions), dimensions, nan);
            }
            for (const auto point : to_randomise)
                randomise_point(layout, dimensions, point, rng, diameter);
            const auto result = minimise(stress, layout, dimensions, problem.disconnected, options.method, Precision::rough);
            projections[i] = Projection{std::move(layout), dimensions, result.stress, result.iterations, seed, i, result.termination};
        });
        sort_by_stress(projections);

        if (options.precision == Precision::fine) {
            const std::size_t n_refined = std::min(incremental_refined, projections.size());
            for_each_start(n_refined, options.threads, [&](std::size_t i) {
                const auto result = minimise(stress, projections[i].layout, dimensions, problem.disconnected, options.method, Precision::fine);
                projections[i].stress = result.stress;
                projections[i].iterations += result.iterations;
                projections[i].termination = result.termination;
            });
            sort_by_stress(projections);
        }
        keep_best(projections, options.keep);
        return projections;
    }

    Projection optimise(const Problem& problem, std::span<const double> layout, std::size_t dimensions, Method method, Precision precision)
    {
        problem.validate();
        require_enough_connected(problem);
        if (layout.size() != problem.n_points() * dimensions)
            throw std::invalid_argument{"layout must have n_points rows of `dimensions` columns"};
        const Stress stress{table_distances(problem), problem.n_points(), problem.unmovable};
        std::vector<double> result_layout(layout.begin(), layout.end());
        const auto result = minimise(stress, result_layout, dimensions, problem.disconnected, method, precision);
        return Projection{std::move(result_layout), dimensions, result.stress, result.iterations, 0, 0, result.termination};
    }

} // namespace af::map
