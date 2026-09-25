// Making maps: many random starts minimised in parallel, sorted by stress.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "minimise.hh"
#include "problem.hh"

namespace af::map
{
    struct RelaxOptions
    {
        std::size_t dimensions{2};
        std::size_t n_starts{1};
        std::uint64_t seed{0};
        Method method{Method::cg};
        Precision precision{Precision::fine};
        bool dimension_annealing{false};              // start in 5-D, reduce by PCA (scratch only)
        double randomisation_diameter_multiplier{2.0}; // random layouts span this x the sample map
        std::size_t keep{0};                          // projections to return, 0 = all
        int threads{0};                               // 0 = OpenMP default
    };

    struct Projection
    {
        std::vector<double> layout; // [n_points * dimensions], NaN rows for disconnected points
        std::size_t dimensions;
        double stress;
        std::size_t iterations;
        std::uint64_t start_seed;
        std::size_t start_index;
        int termination;
    };

    // Maps from scratch: every point placed at random. Sorted by stress (then start index).
    std::vector<Projection> relax(const Problem& problem, const RelaxOptions& options);

    // Maps from an existing layout (the chain's incremental step): points whose row is NaN
    // (and that are not disconnected) are placed at random, every movable point is then
    // minimised. As ae relax_incremental: all starts at rough precision, then the best five
    // again at `options.precision` if that is fine.
    std::vector<Projection> relax_incremental(const Problem& problem, std::span<const double> start_layout, const RelaxOptions& options);

    // Minimises one given layout (no randomisation): ae Projection::relax.
    Projection optimise(const Problem& problem, std::span<const double> layout, std::size_t dimensions, Method method, Precision precision);

    // The diameter of the random cube used for starts: a very rough map from a random layout
    // within the table's largest distance, its bounding-box diagonal times the multiplier
    // (ae randomizer_plain_from_sample_optimization).
    double randomisation_diameter(const Problem& problem, std::size_t dimensions, std::uint64_t seed, double multiplier);

} // namespace af::map
