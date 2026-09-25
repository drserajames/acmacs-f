// Local minimisation of the stress with alglib, and the PCA used by dimension annealing.

#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "stress.hh"

namespace af::map
{
    // alglib nonlinear conjugate gradient is ae's (and AD's) default; L-BFGS is an option.
    enum class Method { cg, lbfgs };

    // Stopping conditions, as in ae alglib.cc: `fine` stops only when the gradient is
    // (nearly) zero; the rough settings stop early to find a basin quickly.
    enum class Precision { fine, rough, very_rough };

    struct Minimised
    {
        double stress;
        std::size_t iterations;
        std::size_t evaluations;
        int termination; // alglib termination type, always > 0 (failures throw)
    };

    // Minimises the stress starting from `layout`, in place. Rows of disconnected points
    // must be NaN; they are held at 0 during minimisation (alglib rejects NaN) and are NaN
    // again afterwards. Any other NaN is an error.
    Minimised minimise(const Stress& stress, std::span<double> layout, std::size_t dimensions, const std::vector<char>& disconnected, Method method, Precision precision);

    // Projects a layout onto its first `to` principal components (alglib
    // pcatruncatedsubspace, as ae). Disconnected rows stay NaN.
    std::vector<double> reduce_dimensions(std::span<const double> layout, std::size_t n_points, std::size_t from, std::size_t to, const std::vector<char>& disconnected);

    Method method_from_string(const std::string& name);
    Precision precision_from_string(const std::string& name);

} // namespace af::map
