#include "minimise.hh"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "dataanalysis.h"
#include "linalg.h"
#include "optimization.h"

namespace af::map
{
    namespace
    {
        struct Tolerances
        {
            double epsg; // stop when the scaled gradient norm is below this
            double epsx; // stop when the step is below this
        };

        Tolerances tolerances(Precision precision)
        {
            switch (precision) {
                case Precision::fine:
                    return {1e-10, 0.0};
                case Precision::rough:
                    return {0.5, 1e-3};
                case Precision::very_rough:
                    return {1.0, 0.1};
            }
            throw std::invalid_argument{"unknown precision"};
        }

        struct Objective
        {
            const Stress* stress;
            std::size_t dimensions;
        };

        void evaluate(const alglib::real_1d_array& x, double& value, alglib::real_1d_array& gradient, void* data)
        {
            const auto* objective = static_cast<const Objective*>(data);
            const auto n = static_cast<std::size_t>(x.length());
            value = objective->stress->value_and_gradient({x.getcontent(), n}, objective->dimensions, {gradient.getcontent(), n});
        }

        void set_disconnected_rows(std::span<double> layout, std::size_t dimensions, const std::vector<char>& disconnected, double value)
        {
            for (std::size_t point = 0; point < disconnected.size(); ++point) {
                if (disconnected[point])
                    std::fill_n(layout.begin() + static_cast<std::ptrdiff_t>(point * dimensions), dimensions, value);
            }
        }

        void require_no_nan(std::span<const double> layout, std::size_t dimensions)
        {
            for (std::size_t i = 0; i < layout.size(); ++i) {
                if (!std::isfinite(layout[i]))
                    throw std::invalid_argument{"minimise: point " + std::to_string(i / dimensions) + " has no coordinates but is not disconnected"};
            }
        }

        // Puts disconnected rows at 0 for alglib and back to NaN on every exit path.
        class DisconnectedAtZero
        {
          public:
            DisconnectedAtZero(std::span<double> layout, std::size_t dimensions, const std::vector<char>& disconnected)
                : layout_{layout}, dimensions_{dimensions}, disconnected_{disconnected}
            {
                set_disconnected_rows(layout_, dimensions_, disconnected_, 0.0);
            }
            ~DisconnectedAtZero() { set_disconnected_rows(layout_, dimensions_, disconnected_, std::numeric_limits<double>::quiet_NaN()); }
            DisconnectedAtZero(const DisconnectedAtZero&) = delete;
            DisconnectedAtZero& operator=(const DisconnectedAtZero&) = delete;

          private:
            std::span<double> layout_;
            std::size_t dimensions_;
            const std::vector<char>& disconnected_;
        };

        std::string termination_error(int type)
        {
            switch (type) {
                case -8:
                    return "alglib: infinite or NaN values in the stress or gradient";
                case -7:
                    return "alglib: gradient verification failed";
                case -1:
                    return "alglib: incorrect parameters";
                default:
                    return "alglib: termination type " + std::to_string(type);
            }
        }
    } // namespace

    Minimised minimise(const Stress& stress, std::span<double> layout, std::size_t dimensions, const std::vector<char>& disconnected, Method method, Precision precision)
    {
        DisconnectedAtZero guard{layout, dimensions, disconnected};
        require_no_nan(layout, dimensions);

        const auto [epsg, epsx] = tolerances(precision);
        const double epsf = 0.0;
        const alglib::ae_int_t unlimited_iterations = 0;
        Objective objective{&stress, dimensions};

        alglib::real_1d_array x;
        x.attach_to_ptr(static_cast<alglib::ae_int_t>(layout.size()), layout.data());

        int termination = 0;
        std::size_t iterations = 0, evaluations = 0;
        try {
            if (method == Method::cg) {
                alglib::mincgstate state;
                alglib::mincgreport report;
                alglib::mincgcreate(x, state);
                alglib::mincgsetcond(state, epsg, epsf, epsx, unlimited_iterations);
                alglib::mincgoptimize(state, evaluate, nullptr, &objective);
                alglib::mincgresultsbuf(state, x, report);
                termination = static_cast<int>(report.terminationtype);
                iterations = static_cast<std::size_t>(report.iterationscount);
                evaluations = static_cast<std::size_t>(report.nfev);
            }
            else {
                // ae uses a single correction pair and a maximum step of 0.1.
                const alglib::ae_int_t corrections = 1;
                const double max_step = 0.1;
                alglib::minlbfgsstate state;
                alglib::minlbfgsreport report;
                alglib::minlbfgscreate(corrections, x, state);
                alglib::minlbfgssetcond(state, epsg, epsf, epsx, unlimited_iterations);
                alglib::minlbfgssetstpmax(state, max_step);
                alglib::minlbfgsoptimize(state, evaluate, nullptr, &objective);
                alglib::minlbfgsresultsbuf(state, x, report);
                termination = static_cast<int>(report.terminationtype);
                iterations = static_cast<std::size_t>(report.iterationscount);
                evaluations = static_cast<std::size_t>(report.nfev);
            }
        }
        catch (alglib::ap_error& err) {
            throw std::runtime_error{"alglib: " + err.msg};
        }
        if (termination < 0)
            throw std::runtime_error{termination_error(termination)};

        return {stress.value(layout, dimensions), iterations, evaluations, termination};
    }

    std::vector<double> reduce_dimensions(std::span<const double> layout, std::size_t n_points, std::size_t from, std::size_t to, const std::vector<char>& disconnected)
    {
        if (to >= from)
            throw std::invalid_argument{"reduce_dimensions: target dimensions must be fewer than the source"};
        std::vector<double> source(layout.begin(), layout.end());
        set_disconnected_rows(source, from, disconnected, 0.0);

        const auto n = static_cast<alglib::ae_int_t>(n_points);
        const auto n_from = static_cast<alglib::ae_int_t>(from);
        const auto n_to = static_cast<alglib::ae_int_t>(to);
        std::vector<double> result(n_points * to);
        try {
            alglib::real_2d_array x;
            x.attach_to_ptr(n, n_from, source.data());
            alglib::real_1d_array variances;
            alglib::real_2d_array basis;
            const double eps = 0.0;
            const alglib::ae_int_t max_iterations = 0;
            alglib::pcatruncatedsubspace(x, n, n_from, n_to, eps, max_iterations, variances, basis);
            // result = x * basis
            for (std::size_t p = 0; p < n_points; ++p) {
                for (std::size_t j = 0; j < to; ++j) {
                    double sum = 0.0;
                    for (std::size_t k = 0; k < from; ++k)
                        sum += source[p * from + k] * basis(static_cast<alglib::ae_int_t>(k), static_cast<alglib::ae_int_t>(j));
                    result[p * to + j] = sum;
                }
            }
        }
        catch (alglib::ap_error& err) {
            throw std::runtime_error{"alglib pca: " + err.msg};
        }
        set_disconnected_rows(result, to, disconnected, std::numeric_limits<double>::quiet_NaN());
        return result;
    }

    Method method_from_string(const std::string& name)
    {
        if (name == "cg")
            return Method::cg;
        if (name == "lbfgs")
            return Method::lbfgs;
        throw std::invalid_argument{"unknown method \"" + name + "\": expected cg or lbfgs"};
    }

    Precision precision_from_string(const std::string& name)
    {
        if (name == "fine")
            return Precision::fine;
        if (name == "rough")
            return Precision::rough;
        if (name == "very_rough")
            return Precision::very_rough;
        throw std::invalid_argument{"unknown precision \"" + name + "\": expected fine, rough or very_rough"};
    }

} // namespace af::map
