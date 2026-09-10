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

struct ValidatedInput {
    phase_a::Dimensions4D dimensions;
    std::array<py::ssize_t, 4> shape;
    const double* data = nullptr;
    std::size_t size = 0;
};

struct CopiedInput {
    phase_a::Dimensions4D dimensions;
    std::array<py::ssize_t, 4> shape;
    std::vector<double> values;
};

ValidatedInput validate_input_metadata(const py::handle input_object)
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

    ValidatedInput validated{};
    for (py::ssize_t axis = 0; axis < 4; ++axis) {
        const py::ssize_t extent = input.shape(axis);
        if (extent <= 0) {
            throw py::value_error("all four input dimensions must be nonempty");
        }
        validated.shape[static_cast<std::size_t>(axis)] = extent;
    }
    validated.dimensions = {
        static_cast<std::size_t>(validated.shape[0]),
        static_cast<std::size_t>(validated.shape[1]),
        static_cast<std::size_t>(validated.shape[2]),
        static_cast<std::size_t>(validated.shape[3]),
    };
    validated.data = static_cast<const double*>(input.data());
    validated.size = static_cast<std::size_t>(input.size());
    return validated;
}

void validate_finite_values(const ValidatedInput& input)
{
    for (std::size_t index = 0; index < input.size; ++index) {
        if (!std::isfinite(input.data[index])) {
            throw py::value_error("input values must all be finite");
        }
    }
}

CopiedInput validate_and_copy_input(const py::handle input_object)
{
    const ValidatedInput validated = validate_input_metadata(input_object);
    CopiedInput copied{validated.dimensions, validated.shape, {}};

    // The CUDA baseline and opt-in comparison paths retain the original
    // NumPy-to-vector copy while checking finite values in the same pass.
    copied.values.resize(validated.size);
    for (std::size_t index = 0; index < copied.values.size(); ++index) {
        if (!std::isfinite(validated.data[index])) {
            throw py::value_error("input values must all be finite");
        }
        copied.values[index] = validated.data[index];
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

    std::memcpy(
        result.mutable_data(),
        output.data(),
        output.size() * sizeof(double));
    return result;
}

template <typename Filter>
py::array_t<double> run_copied_filter(const py::handle input_object, Filter&& filter)
{
    CopiedInput input = validate_and_copy_input(input_object);
    std::vector<double> output;
    {
        py::gil_scoped_release release;
        output = filter(input.values, input.dimensions);
    }
    return copy_output_to_numpy(input.shape, output);
}

template <typename Filter>
py::array_t<double> run_direct_filter(const py::handle input_object, Filter&& filter)
{
    const ValidatedInput input = validate_input_metadata(input_object);
    validate_finite_values(input);

    py::array_t<double> output(input.shape);
    double* const output_data = output.mutable_data();
    {
        // The Python input and output owners remain alive on this stack while
        // the native call accesses only the captured contiguous buffers.
        py::gil_scoped_release release;
        filter(input.data, output_data, input.dimensions);
    }
    return output;
}

} // namespace

PYBIND11_MODULE(fourdstem_median, module)
{
    module.doc() =
        "Python bindings for the Phase A 4D-STEM fixed median filter.";

    module.def(
        "fixed_median_serial",
        [](const py::object& input) {
            return run_direct_filter(
                input,
                [](const double* input_data,
                   double* output_data,
                   const phase_a::Dimensions4D& dimensions) {
                    phase_a::fixed_median_3x3_median9_direct_addressing_buffer(
                        input_data,
                        output_data,
                        dimensions);
                });
        },
        py::arg("array"),
        "Apply the optimized serial fixed 3x3 scan-space median filter.");

    module.def(
        "fixed_median_openmp",
        [](const py::object& input, const int thread_count) {
            return run_direct_filter(
                input,
                [thread_count](
                    const double* input_data,
                    double* output_data,
                    const phase_a::Dimensions4D& dimensions) {
                    phase_a::fixed_median_3x3_median9_direct_addressing_openmp_buffer(
                        input_data,
                        output_data,
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
            return run_copied_filter(
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

#ifdef PHASE_A_PYTHON_COPY_BENCHMARK
    module.def(
        "_benchmark_fixed_median_serial_copied",
        [](const py::object& input) {
            return run_copied_filter(
                input,
                [](const std::vector<double>& values, const phase_a::Dimensions4D& dimensions) {
                    return phase_a::fixed_median_3x3_median9_direct_addressing(
                        values,
                        dimensions);
                });
        },
        py::arg("array"));

    module.def(
        "_benchmark_fixed_median_openmp_copied",
        [](const py::object& input, const int thread_count) {
            return run_copied_filter(
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
        py::arg("thread_count"));

    module.def(
        "_benchmark_validate_finite",
        [](const py::object& input) {
            const ValidatedInput validated = validate_input_metadata(input);
            validate_finite_values(validated);
        },
        py::arg("array"));
#endif
}
