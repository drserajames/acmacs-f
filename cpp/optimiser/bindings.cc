// Python module af.map._core: numpy arrays in, numpy arrays out (interface I1).
// The friendly API with argument checking and dataclasses is af/map/optimise.py.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "grid_test.hh"
#include "problem.hh"
#include "random.hh"
#include "relax.hh"
#include "stress.hh"

namespace py = pybind11;
using namespace af::map;

namespace
{
    template <typename T> using carray = py::array_t<T, py::array::c_style | py::array::forcecast>;

    template <typename T> std::vector<T> to_vector(const carray<T>& array) { return std::vector<T>(array.data(), array.data() + array.size()); }

    std::vector<char> to_flags(const carray<bool>& array)
    {
        std::vector<char> result(static_cast<std::size_t>(array.size()));
        for (py::ssize_t i = 0; i < array.size(); ++i)
            result[static_cast<std::size_t>(i)] = array.data()[i] ? 1 : 0;
        return result;
    }

    Problem make_problem(const carray<double>& titre_value, const carray<std::int8_t>& titre_type, const carray<double>& column_bases, const carray<bool>& disconnected, bool dodgy_is_regular,
                         const std::optional<carray<double>>& weights, const std::optional<carray<double>>& avidity_adjust, const std::optional<carray<double>>& gradient_multipliers,
                         const std::optional<carray<bool>>& unmovable)
    {
        if (titre_value.ndim() != 2)
            throw std::invalid_argument{"titre_value must be a 2-D array [n_antigens, n_sera]"};
        if (titre_type.ndim() != 2 || titre_type.shape(0) != titre_value.shape(0) || titre_type.shape(1) != titre_value.shape(1))
            throw std::invalid_argument{"titre_type must have the same shape as titre_value"};
        if (weights && (weights->ndim() != 2 || weights->shape(0) != titre_value.shape(0) || weights->shape(1) != titre_value.shape(1)))
            throw std::invalid_argument{"weights must have the same shape as titre_value"};
        if (gradient_multipliers) {
            for (py::ssize_t i = 0; i < gradient_multipliers->size(); ++i) {
                if (gradient_multipliers->data()[i] != 1.0)
                    throw std::invalid_argument{"gradient_multipliers other than 1.0 are not implemented"};
            }
        }
        Problem problem;
        problem.n_antigens = static_cast<std::size_t>(titre_value.shape(0));
        problem.n_sera = static_cast<std::size_t>(titre_value.shape(1));
        problem.titre_value = to_vector(titre_value);
        problem.titre_type = to_vector(titre_type);
        problem.column_bases = to_vector(column_bases);
        problem.disconnected = to_flags(disconnected);
        problem.dodgy_is_regular = dodgy_is_regular;
        if (weights)
            problem.weights = to_vector(*weights);
        if (avidity_adjust)
            problem.avidity_adjust = to_vector(*avidity_adjust);
        if (unmovable)
            problem.unmovable = to_flags(*unmovable);
        problem.validate();
        return problem;
    }

    std::vector<double> layout_vector(const Problem& problem, const carray<double>& layout, std::size_t& dimensions)
    {
        if (layout.ndim() != 2 || static_cast<std::size_t>(layout.shape(0)) != problem.n_points())
            throw std::invalid_argument{"layout must be a 2-D array [n_points, dimensions]"};
        dimensions = static_cast<std::size_t>(layout.shape(1));
        return to_vector(layout);
    }

    py::array_t<double> layout_array(const std::vector<double>& layout, std::size_t dimensions)
    {
        py::array_t<double> result({static_cast<py::ssize_t>(layout.size() / dimensions), static_cast<py::ssize_t>(dimensions)});
        std::copy(layout.begin(), layout.end(), result.mutable_data());
        return result;
    }

    py::dict projection_dict(const Projection& projection)
    {
        py::dict result;
        result["layout"] = layout_array(projection.layout, projection.dimensions);
        result["stress"] = projection.stress;
        result["dimensions"] = projection.dimensions;
        result["n_iterations"] = projection.iterations;
        result["start_seed"] = projection.start_seed;
        result["start_index"] = projection.start_index;
        result["termination"] = projection.termination;
        return result;
    }

    py::list projection_list(const std::vector<Projection>& projections)
    {
        py::list result;
        for (const auto& projection : projections)
            result.append(projection_dict(projection));
        return result;
    }

    const char* diagnosis_name(Diagnosis diagnosis)
    {
        switch (diagnosis) {
            case Diagnosis::excluded:
                return "excluded";
            case Diagnosis::normal:
                return "normal";
            case Diagnosis::trapped:
                return "trapped";
            case Diagnosis::hemisphering:
                return "hemisphering";
        }
        return "unknown";
    }
} // namespace

PYBIND11_MODULE(_core, m)
{
    m.doc() = "af map optimiser core (C++, alglib). Use af.map.optimise rather than this module directly.";

    // Translate our own exceptions here, in this module. pybind11's built-in translation
    // goes through a registry shared by every pybind11 module in the process; when one of
    // matplotlib's modules had been imported first, std::invalid_argument from this module
    // was not recognised there and reached Python as "RuntimeError: Caught an unknown
    // exception!". A local translator runs first for this module's functions only.
    py::register_local_exception_translator([](std::exception_ptr error) {
        try {
            if (error)
                std::rethrow_exception(error);
        }
        catch (const py::builtin_exception& err) { // pybind11's own (type_error, index_error, ...)
            err.set_error();
        }
        catch (const std::invalid_argument& err) {
            PyErr_SetString(PyExc_ValueError, err.what());
        }
        catch (const std::exception& err) {
            PyErr_SetString(PyExc_RuntimeError, err.what());
        }
    });

    py::class_<Problem>(m, "Problem")
        .def(py::init(&make_problem), py::arg("titre_value"), py::arg("titre_type"), py::arg("column_bases"), py::arg("disconnected"), py::kw_only(), py::arg("dodgy_is_regular"),
             py::arg("weights") = py::none(), py::arg("avidity_adjust") = py::none(), py::arg("gradient_multipliers") = py::none(), py::arg("unmovable") = py::none())
        .def_property_readonly("n_antigens", [](const Problem& p) { return p.n_antigens; })
        .def_property_readonly("n_sera", [](const Problem& p) { return p.n_sera; })
        .def_property_readonly("n_points", &Problem::n_points)
        .def_property_readonly("column_bases", [](const Problem& p) { return py::array_t<double>(static_cast<py::ssize_t>(p.column_bases.size()), p.column_bases.data()); })
        .def(
            "stress",
            [](const Problem& p, const carray<double>& layout) {
                std::size_t dimensions = 0;
                auto args = layout_vector(p, layout, dimensions);
                for (std::size_t point = 0; point < p.n_points(); ++point) {
                    if (p.disconnected[point])
                        std::fill_n(args.begin() + static_cast<std::ptrdiff_t>(point * dimensions), dimensions, 0.0);
                }
                return Stress{table_distances(p), p.n_points()}.value(args, dimensions);
            },
            py::arg("layout"))
        .def(
            "gradient",
            [](const Problem& p, const carray<double>& layout) {
                std::size_t dimensions = 0;
                auto args = layout_vector(p, layout, dimensions);
                for (std::size_t point = 0; point < p.n_points(); ++point) {
                    if (p.disconnected[point])
                        std::fill_n(args.begin() + static_cast<std::ptrdiff_t>(point * dimensions), dimensions, 0.0);
                }
                std::vector<double> gradient(args.size());
                Stress{table_distances(p), p.n_points(), p.unmovable}.value_and_gradient(args, dimensions, gradient);
                return layout_array(gradient, dimensions);
            },
            py::arg("layout"))
        .def("n_table_distances",
             [](const Problem& p) {
                 const auto td = table_distances(p);
                 return py::make_tuple(td.regular.size(), td.less_than.size());
             })
        .def("max_table_distance", &max_table_distance);

    m.def(
        "relax",
        [](const Problem& problem, std::size_t dimensions, std::size_t n_starts, std::size_t first_start, std::uint64_t seed, const std::string& method, const std::string& precision,
           bool dimension_annealing, std::size_t keep, int threads) {
            RelaxOptions options{dimensions, n_starts, first_start, seed, method_from_string(method), precision_from_string(precision), dimension_annealing, 2.0, keep, threads};
            std::vector<Projection> projections;
            {
                py::gil_scoped_release release;
                projections = relax(problem, options);
            }
            return projection_list(projections);
        },
        py::arg("problem"), py::kw_only(), py::arg("dimensions"), py::arg("n_starts"), py::arg("first_start") = 0, py::arg("seed"), py::arg("method") = "cg", py::arg("precision") = "fine",
        py::arg("dimension_annealing") = false, py::arg("keep") = 0, py::arg("threads") = 0, "Maps from random starts, sorted by stress.");

    m.def(
        "relax_incremental",
        [](const Problem& problem, const carray<double>& start_layout, std::size_t n_starts, std::size_t first_start, std::uint64_t seed, const std::string& method,
           const std::string& precision, std::size_t keep, int threads) {
            std::size_t dimensions = 0;
            const auto layout = layout_vector(problem, start_layout, dimensions);
            RelaxOptions options{dimensions, n_starts, first_start, seed, method_from_string(method), precision_from_string(precision), false, 2.0, keep, threads};
            std::vector<Projection> projections;
            {
                py::gil_scoped_release release;
                projections = relax_incremental(problem, layout, options);
            }
            return projection_list(projections);
        },
        py::arg("problem"), py::arg("start_layout"), py::kw_only(), py::arg("n_starts"), py::arg("first_start") = 0, py::arg("seed"), py::arg("method") = "cg",
        py::arg("precision") = "rough", py::arg("keep") = 0, py::arg("threads") = 0, "Maps from a layout whose NaN rows are placed at random (one precision for every start).");

    m.def(
        "refine",
        [](const Problem& problem, const std::vector<carray<double>>& layouts, const std::string& method, const std::string& precision, int threads) {
            std::vector<Projection> projections;
            for (const auto& layout : layouts) {
                std::size_t dimensions = 0;
                auto args = layout_vector(problem, layout, dimensions);
                projections.push_back(Projection{std::move(args), dimensions, 0.0, 0, 0, 0, 0});
            }
            {
                py::gil_scoped_release release;
                projections = refine(problem, std::move(projections), method_from_string(method), precision_from_string(precision), threads);
            }
            return projection_list(projections);
        },
        py::arg("problem"), py::arg("layouts"), py::kw_only(), py::arg("method") = "cg", py::arg("precision") = "fine", py::arg("threads") = 0,
        "Minimises each layout further, in parallel; results in the order given.");

    m.def(
        "optimise",
        [](const Problem& problem, const carray<double>& layout, const std::string& method, const std::string& precision) {
            std::size_t dimensions = 0;
            const auto args = layout_vector(problem, layout, dimensions);
            Projection projection;
            {
                py::gil_scoped_release release;
                projection = optimise(problem, args, dimensions, method_from_string(method), precision_from_string(precision));
            }
            return projection_dict(projection);
        },
        py::arg("problem"), py::arg("layout"), py::kw_only(), py::arg("method") = "cg", py::arg("precision") = "fine", "Minimises one layout in place of randomising.");

    m.def(
        "grid_test",
        [](const Problem& problem, const carray<double>& layout, double step, int threads) {
            std::size_t dimensions = 0;
            const auto args = layout_vector(problem, layout, dimensions);
            std::vector<GridResult> results;
            {
                py::gil_scoped_release release;
                results = grid_test(problem, args, dimensions, step, threads);
            }
            py::list out;
            for (const auto& result : results) {
                py::dict entry;
                entry["point"] = result.point;
                entry["diagnosis"] = diagnosis_name(result.diagnosis);
                entry["position"] = result.position.empty() ? py::object(py::none()) : py::object(py::array_t<double>(static_cast<py::ssize_t>(result.position.size()), result.position.data()));
                entry["distance"] = result.distance;
                entry["stress_diff"] = result.stress_diff;
                out.append(entry);
            }
            return out;
        },
        py::arg("problem"), py::arg("layout"), py::kw_only(), py::arg("step") = 0.1, py::arg("threads") = 0);

    m.def(
        "column_bases",
        [](const carray<double>& titre_value, const carray<std::int8_t>& titre_type, double minimum) {
            if (titre_value.ndim() != 2)
                throw std::invalid_argument{"titre_value must be 2-D"};
            const auto result = column_bases(to_vector(titre_value), to_vector(titre_type), static_cast<std::size_t>(titre_value.shape(0)), static_cast<std::size_t>(titre_value.shape(1)), minimum);
            return py::array_t<double>(static_cast<py::ssize_t>(result.size()), result.data());
        },
        py::arg("titre_value"), py::arg("titre_type"), py::arg("minimum") = 0.0, "Column bases as ae computes them unforced (tests only).");

    m.def(
        "resolve_threads", [](int requested) { return resolve_threads(requested); }, py::arg("requested"),
        "Threads a run with `threads=requested` uses (0: OpenMP's default, which honours OMP_NUM_THREADS).");

    m.def(
        "start_seed", [](std::uint64_t seed, std::uint64_t index) { return start_seed(seed, index); }, py::arg("seed"), py::arg("index"));

#ifdef _OPENMP
    m.attr("openmp") = true;
#else
    m.attr("openmp") = false;
#endif
}
