// Antigen reactivity: one log2-titre offset per antigen, fitted together with the map.
//
// An antigen that reacts more (or less) strongly than its position explains, with every
// serum, shows up as titres all shifted by the same amount. The adjustment r_i is ADDED to
// antigen i's log titres to correct for that, so each target distance drops by r_i (Racmacs's
// agReactivityAdjustments; the same convention as avidity_adjust, r_i = log2(avidity factor)).
// An antigen whose titres all read too high for its position gets a negative r_i.
//
// The fit minimises   stress(layout, r) + sum_i (penalty * r_i)^2   over the layout and the
// reactivities together, in one minimisation with an analytic gradient (Racmacs's objective,
// optimizeAgReactivity, without its finite differences). Two choices differ between the
// references and are explicit here:
//   - column bases: `fixed` (af, interface I1, ae) keeps the caller's; `recompute`
//     (Racmacs) recomputes them from the adjusted titres, so raising the antigen that sets a
//     serum's column basis raises the basis too;
//   - clipping: ae and af clip target distances at 0; Racmacs does not.

#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "minimise.hh"
#include "problem.hh"

namespace af::map
{
    enum class ColumnBasesMode { fixed, recompute };

    struct ReactivityOptions
    {
        double penalty{1.0};
        ColumnBasesMode column_bases{ColumnBasesMode::fixed};
        bool clip{true};
        double minimum_column_basis{0.0}; // recompute only (log2(titre/10); 0 = "none")
        Method method{Method::cg};
        Precision precision{Precision::fine};
    };

    struct ReactivityFit
    {
        std::vector<double> reactivity;   // [n_antigens], log2 units
        std::vector<double> layout;       // [n_points * dimensions], NaN rows for disconnected
        std::size_t dimensions;
        double stress;                    // the map stress with these reactivities (no penalty)
        double objective;                 // stress + the penalty term
        std::vector<double> column_bases; // the column bases the result uses
        std::size_t iterations;
        int termination;
    };

    // `fixed`: [n_antigens]; NaN marks an antigen to fit, a finite value holds it there.
    // `start`: [n_antigens] starting reactivities (ignored for held antigens).
    ReactivityFit fit_reactivity(const Problem& problem, std::span<const double> layout, std::size_t dimensions, std::span<const double> start, std::span<const double> fixed,
                                 const ReactivityOptions& options);

    // The objective at a given layout and reactivities, every antigen counted as free; writes
    // d/d(layout) and d/d(reactivity) when the spans are not empty. For tests.
    double reactivity_objective(const Problem& problem, std::span<const double> layout, std::size_t dimensions, std::span<const double> reactivity, const ReactivityOptions& options,
                                std::span<double> layout_gradient, std::span<double> reactivity_gradient);

    // The column bases for given reactivities under `options` (the caller's when fixed).
    std::vector<double> reactivity_column_bases(const Problem& problem, std::span<const double> reactivity, const ReactivityOptions& options);

    ColumnBasesMode column_bases_mode_from_string(const std::string& name);

} // namespace af::map
