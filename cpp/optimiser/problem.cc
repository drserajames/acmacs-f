#include "problem.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace af::map
{
    namespace
    {
        void require(bool condition, const std::string& message)
        {
            if (!condition)
                throw std::invalid_argument{"af.map optimiser input: " + message};
        }

        bool is_valid_type(std::int8_t type) { return type >= 0 && type <= 4; }
    } // namespace

    std::size_t Problem::n_connected() const { return static_cast<std::size_t>(std::count(disconnected.begin(), disconnected.end(), char{0})); }

    bool Problem::has_unmovable() const { return std::any_of(unmovable.begin(), unmovable.end(), [](char flag) { return flag != 0; }); }

    void Problem::validate() const
    {
        const auto n_titres = n_antigens * n_sera;
        require(n_antigens > 0 && n_sera > 0, "the table has no antigens or no sera");
        require(titre_value.size() == n_titres, "titre_value must have n_antigens * n_sera entries");
        require(titre_type.size() == n_titres, "titre_type must have n_antigens * n_sera entries");
        require(column_bases.size() == n_sera, "column_bases must have one entry per serum");
        require(disconnected.size() == n_points(), "disconnected must have one entry per point");
        require(weights.empty() || weights.size() == n_titres, "weights must be absent or have n_antigens * n_sera entries");
        require(avidity_adjust.empty() || avidity_adjust.size() == n_points(), "avidity_adjust must be absent or have one entry per point");
        require(unmovable.empty() || unmovable.size() == n_points(), "unmovable must be absent or have one entry per point");

        for (std::size_t i = 0; i < n_titres; ++i) {
            require(is_valid_type(titre_type[i]), "titre_type " + std::to_string(titre_type[i]) + " at cell " + std::to_string(i) + " is not 0..4");
            if (static_cast<TitreType>(titre_type[i]) != TitreType::missing)
                require(std::isfinite(titre_value[i]), "titre_value at cell " + std::to_string(i) + " is not finite but its type is not missing");
        }
        for (std::size_t sr = 0; sr < n_sera; ++sr)
            require(std::isfinite(column_bases[sr]), "column basis of serum " + std::to_string(sr) + " is not finite");
        for (const double weight : weights)
            require(std::isfinite(weight) && weight >= 0.0, "weights must be finite and non-negative");
        for (const double adjust : avidity_adjust)
            require(std::isfinite(adjust) && adjust > 0.0, "avidity_adjust factors must be finite and positive");
        // ae silently takes an unmovable point out of the disconnected set. Here the caller
        // has to decide which one it means.
        if (!unmovable.empty()) {
            for (std::size_t p = 0; p < n_points(); ++p)
                require(!(unmovable[p] && disconnected[p]), "point " + std::to_string(p) + " is both unmovable and disconnected");
        }
    }

    TableDistances table_distances(const Problem& problem)
    {
        TableDistances result;
        const auto n_ag = problem.n_antigens;
        const auto n_sr = problem.n_sera;
        const auto logged_adjust = [&problem](std::size_t point) { return problem.avidity_adjust.empty() ? 0.0 : std::log2(problem.avidity_adjust[point]); };

        for (std::size_t ag = 0; ag < n_ag; ++ag) {
            for (std::size_t sr = 0; sr < n_sr; ++sr) {
                const auto cell = ag * n_sr + sr;
                const auto type = static_cast<TitreType>(problem.titre_type[cell]);
                const auto serum_point = n_ag + sr;
                if (type == TitreType::missing || type == TitreType::more_than || (type == TitreType::dodgy && !problem.dodgy_is_regular))
                    continue;
                if (problem.disconnected[ag] || problem.disconnected[serum_point])
                    continue;
                // A titre above the column basis would give a negative distance; ae clips it
                // to 0 (multiply_antigen_titer_until_column_adjust::yes, always on in the chains).
                const double distance = std::max(0.0, problem.column_bases[sr] - problem.titre_value[cell] - (logged_adjust(ag) + logged_adjust(serum_point)));
                const double weight = problem.weights.empty() ? 1.0 : problem.weights[cell];
                const TableDistance entry{static_cast<std::uint32_t>(ag), static_cast<std::uint32_t>(serum_point), distance, weight};
                if (type == TitreType::less_than)
                    result.less_than.push_back(entry);
                else
                    result.regular.push_back(entry);
            }
        }
        return result;
    }

    double max_table_distance(const Problem& problem)
    {
        double result = 0.0;
        for (std::size_t ag = 0; ag < problem.n_antigens; ++ag) {
            for (std::size_t sr = 0; sr < problem.n_sera; ++sr) {
                const auto cell = ag * problem.n_sera + sr;
                double logged = problem.titre_value[cell];
                switch (static_cast<TitreType>(problem.titre_type[cell])) {
                    case TitreType::missing:
                        continue;
                    case TitreType::less_than:
                        logged -= 1.0;
                        break;
                    case TitreType::more_than:
                        logged += 1.0;
                        break;
                    case TitreType::regular:
                    case TitreType::dodgy:
                        break;
                }
                result = std::max(result, problem.column_bases[sr] - logged);
            }
        }
        return result;
    }

    std::vector<double> column_bases(const std::vector<double>& titre_value, const std::vector<std::int8_t>& titre_type, std::size_t n_antigens, std::size_t n_sera, double minimum)
    {
        if (titre_value.size() != n_antigens * n_sera || titre_type.size() != n_antigens * n_sera)
            throw std::invalid_argument{"column_bases: titre arrays must have n_antigens * n_sera entries"};
        std::vector<double> result(n_sera, minimum);
        for (std::size_t ag = 0; ag < n_antigens; ++ag) {
            for (std::size_t sr = 0; sr < n_sera; ++sr) {
                const auto cell = ag * n_sera + sr;
                switch (static_cast<TitreType>(titre_type[cell])) {
                    case TitreType::regular:
                    case TitreType::less_than:
                        result[sr] = std::max(result[sr], titre_value[cell]);
                        break;
                    case TitreType::more_than:
                        result[sr] = std::max(result[sr], titre_value[cell] + 1.0);
                        break;
                    case TitreType::missing:
                    case TitreType::dodgy:
                        break;
                }
            }
        }
        return result;
    }

} // namespace af::map
