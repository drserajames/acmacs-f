// Inputs to the map optimiser (interface I1) and the table distances derived from them.
//
// Python resolves everything that needs chart knowledge (titre parsing, column bases,
// disconnection) and hands the core plain arrays. The core never extends or overrides
// those decisions; it only checks that they are consistent.
//
// Points are antigens then sera: point p < n_antigens is antigen p, otherwise serum
// p - n_antigens. Layouts are row-major [n_points * dimensions].

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace af::map
{
    enum class TitreType : std::int8_t { missing = 0, regular = 1, less_than = 2, more_than = 3, dodgy = 4 };

    struct Problem
    {
        std::size_t n_antigens{0};
        std::size_t n_sera{0};
        std::vector<double> titre_value;       // [n_antigens * n_sera], log2(titre/10); ignored where missing
        std::vector<std::int8_t> titre_type;   // [n_antigens * n_sera], TitreType values
        std::vector<double> column_bases;      // [n_sera], resolved by the caller
        std::vector<char> disconnected;        // [n_points]
        bool dodgy_is_regular{false};
        std::vector<double> weights;           // empty (all 1) or [n_antigens * n_sera]
        std::vector<double> avidity_adjust;    // empty (all 1) or [n_points]; raw factors as in .ace "f"
        std::vector<char> unmovable;           // empty (none) or [n_points]

        std::size_t n_points() const { return n_antigens + n_sera; }
        std::size_t n_connected() const;
        bool is_unmovable(std::size_t point) const { return !unmovable.empty() && unmovable[point]; }
        bool has_unmovable() const;

        // Throws std::invalid_argument naming the first inconsistency found.
        void validate() const;
    };

    // One titre turned into a target distance between an antigen (p1) and a serum (p2).
    struct TableDistance
    {
        std::uint32_t p1;
        std::uint32_t p2;
        double distance;
        double weight;
    };

    // Regular titres are fitted exactly; "<" titres only penalise map distances that are
    // too short (see stress.cc). ">" titres and, unless dodgy_is_regular, "~" titres do not
    // take part. Titres touching a disconnected point are left out.
    struct TableDistances
    {
        std::vector<TableDistance> regular;
        std::vector<TableDistance> less_than;

        bool empty() const { return regular.empty() && less_than.empty(); }
    };

    TableDistances table_distances(const Problem& problem);

    // Largest target distance in the table, counting "<X" one step below X and ">X" one
    // step above; sizes the first random layout (ae Titers::max_distance).
    double max_table_distance(const Problem& problem);

    // Column bases as ae computes them without forcing: per serum the largest log titre
    // ("<X" counts as X, ">X" as one step above, "~" and missing are ignored), raised to
    // `minimum`. For tests and comparisons only: production column bases come from Python.
    std::vector<double> column_bases(const std::vector<double>& titre_value, const std::vector<std::int8_t>& titre_type, std::size_t n_antigens, std::size_t n_sera, double minimum);

} // namespace af::map
