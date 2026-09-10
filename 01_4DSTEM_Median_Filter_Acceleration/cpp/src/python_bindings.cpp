#include "fixed_median.hpp"

#ifdef PHASE_A_PYTHON_CUDA_ENABLED
#include "fixed_median_cuda.hpp"
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct CopiedInput {
    phase_a::Dimensions4D dimensions;
    std::array<py::ssize_t, 4> shape;
    std::vector<double> values;
};

CopiedInput validate_and_copy_input(const py::handle input_object)
{
    if (!py::isinstance<py::array>(input_object)) {
        throw py::type_error("input must be a NumPy ndarray");
    }

    const py::array input = py::reinterpret_borrow<py::array>(input_object);
    if (input.ndim() != 4) {
        throw py::value_error("input must have exactly four dimensions");
    }
    if (!input.dtype().is(py::dtype::of<double>())) {
        throw py::type_error("input dtype must be exactly NumPy float64");
    }
    if ((input.flags() & py::array::c_style) == 0) {
        throw py::value_error("input must be C-contiguous");
    }

    CopiedInput copied{};
    for (py::ssize_t axis = 0; axis < 4; ++axis) {
        const py::ssize_t extent = input.shape(axis);
        if (extent <= 0) {
            throw py::value_error("all four input dimensions must be nonempty");
        }
        copied.shape[static_cast<std::size_t>(axis)] = extent;
    }
    copied.dimensions = {
        static_cast<std::size_t>(copied.shape[0]),
        static_cast<std::size_t>(copied.shape[1]),
        static_cast<std::size_t>(copied.shape[2]),
        static_cast<std::size_t>(copied.shape[3]),
    };

    // Baseline copy 1: NumPy-owned C-order storage to an independent vector.
    // Validation stays under the GIL and rejects nonfinite Phase A input.
    const auto* source = static_cast<const double*>(input.data());
    copied.values.resize(static_cast<std::size_t>(input.size()));
    for (std::size_t index = 0; index < copied.values.size(); ++index) {
        if (!std::isfinite(source[index])) {
            throw py::value_error("input values must all be finite");
        }
        copied.values[index] = source[index];
    }
    return copied;
}

py::array_t<double> copy_output_to_numpy(
    const std::array<py::ssize_t, 4>& shape,
    const std::vector<double>& output)
{
    py::array_t<double> result(shape);
    if (static_cast<std::size_t>(result.size()) != output.size()) {
        throw std::runtime_error("native output size does not match the input shape");
    }

    // Baseline copy 2: native vector storage to a new NumPy-owned C-order array.
    std::memcpy(
        result.mutable_data(),
        output.data(),
        output.size() * sizeof(double));
    return result;
}

template <typename Filter>
py::array_t<double> run_filter(const py::handle input_object, Filter&& filter)
{
    CopiedInput input = validate_and_copy_input(input_object);
    std::vector<double> output;
    {
        py::gil_scoped_release release;
        output = filter(input.values, input.dimensions);
    }
    return copy_output_to_numpy(input.shape, output);
}

} // namespace

PYBIND11_MODULE(fourdstem_median, module)
{
    module.doc() =
        "Correctness-first Python bindings for the Phase A 4D-STEM fixed median filter.";

    module.def(
        "fixed_median_serial",
        [](const py::object& input) {
            return run_filter(
                input,
                [](const std::vector<double>& values, const phase_a::Dimensions4D& dimensions) {
                    return phase_a::fixed_median_3x3_median9_direct_addressing(
                        values,
                        dimensions);
                });
        },
        py::arg("array"),
        "Apply the optimized serial fixed 3x3 scan-space median filter.");

    module.def(
        "fixed_median_openmp",
        [](const py::object& input, const int thread_count) {
            return run_filter(
                input,
                [thread_count](
                    const std::vector<double>& values,
                    const phase_a::Dimensions4D& dimensions) {
                    return phase_a::fixed_median_3x3_median9_direct_addressing_openmp(
                        values,
                        dimensions,
                        thread_count);
                });
        },
        py::arg("array"),
        py::arg("thread_count"),
        "Apply the optimized OpenMP fixed 3x3 scan-space median filter.");

#ifdef PHASE_A_PYTHON_CUDA_ENABLED
    module.def(
        "fixed_median_cuda",
        [](const py::object& input) {
            return run_filter(
                input,
                [](const std::vector<double>& values, const phase_a::Dimensions4D& dimensions) {
                    phase_a::CudaFilterResult result =
                        phase_a::fixed_median_3x3_cuda_baseline(values, dimensions);
                    return std::move(result.output);
                });
        },
        py::arg("array"),
        "Apply the one-shot correctness-first CUDA fixed 3x3 scan-space median filter.");

    module.def(
        "cuda_device_info",
        []() {
            phase_a::CudaDeviceInfo info;
            {
                py::gil_scoped_release release;
                info = phase_a::cuda_device_info();
            }
            py::dict result;
            result["name"] = info.name;
            result["compute_capability"] =
                py::make_tuple(info.compute_major, info.compute_minor);
            result["multiprocessor_count"] = info.multiprocessor_count;
            result["global_memory_bytes"] = info.global_memory_bytes;
            return result;
        },
        "Return basic information for the CUDA device used by the binding.");
#endif
}
