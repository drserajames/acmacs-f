#include "reactivity.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "stress.hh"

namespace af::map
{
    namespace
    {
        constexpr double nan = std::numeric_limits<double>::quiet_NaN();

        // One fitted titre (regular, "<", or dodgy when it counts as regular).
        struct Titre
        {
            std::size_t antigen;
            std::size_t serum;
            double logged;
            double weight;
            bool less_than;
        };

        // A titre that can set a serum's column basis when bases are recomputed: regular and
        // "<" at their value, ">" one step above (af's and ae's column-basis rule).
        struct BasisCandidate
        {
            std::size_t antigen;
            double logged;
        };

        class Objective
        {
          public:
            Objective(const Problem& problem, std::size_t dimensions, const ReactivityOptions& options)
                : problem_{problem}, dimensions_{dimensions}, options_{options}, candidates_(problem.n_sera), logged_adjust_(problem.n_points(), 0.0)
            {
                const auto n_ag = problem.n_antigens, n_sr = problem.n_sera;
                for (std::size_t point = 0; point < problem.n_points() && !problem.avidity_adjust.empty(); ++point)
                    logged_adjust_[point] = std::log2(problem.avidity_adjust[point]);
                for (std::size_t ag = 0; ag < n_ag; ++ag) {
                    for (std::size_t sr = 0; sr < n_sr; ++sr) {
                        const auto cell = ag * n_sr + sr;
                        const auto type = static_cast<TitreType>(problem.titre_type[cell]);
                        const double value = problem.titre_value[cell];
                        if (type == TitreType::regular || type == TitreType::less_than)
                            candidates_[sr].push_back({ag, value});
                        else if (type == TitreType::more_than)
                            candidates_[sr].push_back({ag, value + 1.0});
                        const bool fitted = type == TitreType::regular || type == TitreType::less_than || (type == TitreType::dodgy && problem.dodgy_is_regular);
                        if (fitted && !problem.disconnected[ag] && !problem.disconnected[n_ag + sr])
                            titres_.push_back({ag, sr, value, problem.weights.empty() ? 1.0 : problem.weights[cell], type == TitreType::less_than});
                    }
                }
            }

            // Column bases for reactivities `r`, and for each serum the antigen setting it
            // (n_antigens when the caller's basis or the minimum applies).
            void column_bases(std::span<const double> r, std::vector<double>& bases, std::vector<std::size_t>& setter) const
            {
                const auto n_ag = problem_.n_antigens, n_sr = problem_.n_sera;
                bases.assign(problem_.column_bases.begin(), problem_.column_bases.end());
                setter.assign(n_sr, n_ag);
                if (options_.column_bases == ColumnBasesMode::fixed)
                    return;
                for (std::size_t sr = 0; sr < n_sr; ++sr) {
                    bases[sr] = options_.minimum_column_basis;
                    for (const auto& candidate : candidates_[sr]) {
                        const double adjusted = candidate.logged + r[candidate.antigen];
                        if (adjusted > bases[sr]) {
                            bases[sr] = adjusted;
                            setter[sr] = candidate.antigen;
                        }
                    }
                }
            }

            // Value; gradients when the spans are not empty. Disconnected rows must be finite.
            double evaluate(std::span<const double> layout, std::span<const double> r, std::span<double> layout_gradient, std::span<double> r_gradient) const
            {
                const auto n_ag = problem_.n_antigens;
                const bool want_gradient = !layout_gradient.empty();
                if (want_gradient) {
                    std::fill(layout_gradient.begin(), layout_gradient.end(), 0.0);
                    std::fill(r_gradient.begin(), r_gradient.end(), 0.0);
                }
                std::vector<double> bases;
                std::vector<std::size_t> setter;
                column_bases(r, bases, setter);

                double value = 0.0;
                for (const auto& titre : titres_) {
                    const auto serum_point = n_ag + titre.serum;
                    double target = bases[titre.serum] - titre.logged - r[titre.antigen] - logged_adjust_[titre.antigen] - logged_adjust_[serum_point];
                    const bool clipped = options_.clip && target < 0.0;
                    if (clipped)
                        target = 0.0;
                    const double dist = map_distance(layout, dimensions_, titre.antigen, serum_point);
                    double d_target; // d(term)/d(target); d(term)/d(dist) is its negative
                    if (titre.less_than) {
                        const double diff = target + 1.0 - dist;
                        value += titre.weight * terms::less_than_error(diff);
                        d_target = titre.weight * terms::less_than_error_slope(diff);
                    }
                    else {
                        const double diff = target - dist;
                        value += titre.weight * diff * diff;
                        d_target = titre.weight * 2.0 * diff;
                    }
                    if (!want_gradient)
                        continue;
                    // map distance: d(term)/d(x_antigen) = -d_target * (x_ag - x_sr) / dist
                    const double scale = d_target / terms::nonzero(dist);
                    const double* x1 = layout.data() + titre.antigen * dimensions_;
                    const double* x2 = layout.data() + serum_point * dimensions_;
                    double* g1 = layout_gradient.data() + titre.antigen * dimensions_;
                    double* g2 = layout_gradient.data() + serum_point * dimensions_;
                    for (std::size_t dim = 0; dim < dimensions_; ++dim) {
                        const double inc = scale * (x1[dim] - x2[dim]);
                        g1[dim] -= inc;
                        g2[dim] += inc;
                    }
                    if (clipped)
                        continue; // the target is held at 0: no dependence on r
                    r_gradient[titre.antigen] -= d_target;
                    if (setter[titre.serum] < n_ag)
                        r_gradient[setter[titre.serum]] += d_target;
                }
                // Racmacs adds the penalty for every antigen's adjustment, held ones included
                for (std::size_t ag = 0; ag < n_ag; ++ag) {
                    value += (options_.penalty * r[ag]) * (options_.penalty * r[ag]);
                    if (want_gradient)
                        r_gradient[ag] += 2.0 * options_.penalty * options_.penalty * r[ag];
                }
                if (want_gradient) {
                    for (std::size_t point = 0; point < problem_.n_points(); ++point) {
                        if (problem_.is_unmovable(point) || problem_.disconnected[point])
                            std::fill_n(layout_gradient.begin() + static_cast<std::ptrdiff_t>(point * dimensions_), dimensions_, 0.0);
                    }
                }
                return value;
            }

            double stress_only(std::span<const double> layout, std::span<const double> r) const
            {
                ReactivityOptions no_penalty = options_;
                no_penalty.penalty = 0.0;
                return Objective{problem_, dimensions_, no_penalty}.evaluate(layout, r, {}, {});
            }

          private:
            const Problem& problem_;
            std::size_t dimensions_;
            ReactivityOptions options_;
            std::vector<Titre> titres_;
            std::vector<std::vector<BasisCandidate>> candidates_;
            std::vector<double> logged_adjust_;
        };

        void check(const Problem& problem, std::span<const double> layout, std::size_t dimensions, const ReactivityOptions& options)
        {
            problem.validate();
            if (problem.n_connected() < 3)
                throw std::invalid_argument{"cannot fit reactivity: fewer than 3 connected points"};
            if (layout.size() != problem.n_points() * dimensions)
                throw std::invalid_argument{"layout must have n_points rows of `dimensions` columns"};
            if (!(options.penalty >= 0.0) || !std::isfinite(options.penalty))
                throw std::invalid_argument{"penalty must be finite and non-negative"};
            if (!std::isfinite(options.minimum_column_basis))
                throw std::invalid_argument{"minimum_column_basis must be finite"};
        }

        std::vector<double> zero_disconnected(const Problem& problem, std::span<const double> layout, std::size_t dimensions)
        {
            std::vector<double> result(layout.begin(), layout.end());
            for (std::size_t point = 0; point < problem.n_points(); ++point) {
                if (problem.disconnected[point])
                    std::fill_n(result.begin() + static_cast<std::ptrdiff_t>(point * dimensions), dimensions, 0.0);
            }
            for (std::size_t i = 0; i < result.size(); ++i) {
                if (!std::isfinite(result[i]))
                    throw std::invalid_argument{"layout: point " + std::to_string(i / dimensions) + " has no coordinates but is not disconnected"};
            }
            return result;
        }
    } // namespace

    ColumnBasesMode column_bases_mode_from_string(const std::string& name)
    {
        if (name == "fixed")
            return ColumnBasesMode::fixed;
        if (name == "recompute")
            return ColumnBasesMode::recompute;
        throw std::invalid_argument{"unknown column_bases \"" + name + "\": expected fixed or recompute"};
    }

    double reactivity_objective(const Problem& problem, std::span<const double> layout, std::size_t dimensions, std::span<const double> reactivity, const ReactivityOptions& options,
                                std::span<double> layout_gradient, std::span<double> reactivity_gradient)
    {
        check(problem, layout, dimensions, options);
        if (reactivity.size() != problem.n_antigens)
            throw std::invalid_argument{"reactivity must have one entry per antigen"};
        if (layout_gradient.size() != (layout_gradient.empty() ? 0 : layout.size()) || reactivity_gradient.size() != (reactivity_gradient.empty() ? 0 : problem.n_antigens))
            throw std::invalid_argument{"gradient spans have the wrong size"};
        const auto args = zero_disconnected(problem, layout, dimensions);
        return Objective{problem, dimensions, options}.evaluate(args, reactivity, layout_gradient, reactivity_gradient);
    }

    std::vector<double> reactivity_column_bases(const Problem& problem, std::span<const double> reactivity, const ReactivityOptions& options)
    {
        problem.validate();
        if (reactivity.size() != problem.n_antigens)
            throw std::invalid_argument{"reactivity must have one entry per antigen"};
        std::vector<double> bases;
        std::vector<std::size_t> setter;
        Objective{problem, 1, options}.column_bases(reactivity, bases, setter);
        return bases;
    }

    ReactivityFit fit_reactivity(const Problem& problem, std::span<const double> layout, std::size_t dimensions, std::span<const double> start, std::span<const double> fixed,
                                 const ReactivityOptions& options)
    {
        check(problem, layout, dimensions, options);
        const auto n_ag = problem.n_antigens;
        if (start.size() != n_ag || fixed.size() != n_ag)
            throw std::invalid_argument{"start and fixed must have one entry per antigen"};

        std::vector<std::size_t> free;
        std::vector<double> r(n_ag);
        for (std::size_t ag = 0; ag < n_ag; ++ag) {
            if (std::isnan(fixed[ag])) {
                if (!std::isfinite(start[ag]))
                    throw std::invalid_argument{"start must be finite for antigens being fitted"};
                free.push_back(ag);
                r[ag] = start[ag];
            }
            else {
                if (!std::isfinite(fixed[ag]))
                    throw std::invalid_argument{"fixed reactivities must be finite or NaN"};
                r[ag] = fixed[ag];
            }
        }

        // x = [layout | reactivities of the free antigens]
        const auto coords = zero_disconnected(problem, layout, dimensions);
        std::vector<double> x(coords);
        for (const auto ag : free)
            x.push_back(r[ag]);
        const Objective objective{problem, dimensions, options};
        const std::size_t n_coords = coords.size();
        std::vector<double> r_work(r), r_gradient(n_ag);
        const ValueAndGradient function = [&](std::span<const double> args, std::span<double> gradient) {
            for (std::size_t k = 0; k < free.size(); ++k)
                r_work[free[k]] = args[n_coords + k];
            const double value = objective.evaluate(args.first(n_coords), r_work, gradient.first(n_coords), r_gradient);
            for (std::size_t k = 0; k < free.size(); ++k)
                gradient[n_coords + k] = r_gradient[free[k]];
            return value;
        };
        const auto result = minimise_function(function, x, options.method, options.precision);

        for (std::size_t k = 0; k < free.size(); ++k)
            r[free[k]] = x[n_coords + k];
        std::vector<double> fitted_layout(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n_coords));
        const double objective_value = objective.evaluate(fitted_layout, r, {}, {});
        const double stress_value = objective.stress_only(fitted_layout, r);
        for (std::size_t point = 0; point < problem.n_points(); ++point) {
            if (problem.disconnected[point])
                std::fill_n(fitted_layout.begin() + static_cast<std::ptrdiff_t>(point * dimensions), dimensions, nan);
        }
        return ReactivityFit{r, std::move(fitted_layout), dimensions, stress_value, objective_value, reactivity_column_bases(problem, r, options), result.iterations, result.termination};
    }

} // namespace af::map
