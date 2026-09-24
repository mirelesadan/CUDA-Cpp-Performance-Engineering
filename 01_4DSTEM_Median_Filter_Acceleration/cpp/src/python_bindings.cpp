#include "fixed_median.hpp"

#ifdef PHASE_A_PYTHON_CUDA_ENABLED
#include "adaptive_median_cuda.hpp"
#include "fixed_median_cuda.hpp"
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
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

phase_a::Dimensions4D validate_shape(const py::handle shape_object)
{
    if (!py::isinstance<py::sequence>(shape_object)) {
        throw py::type_error("shape must be a sequence of four positive dimensions");
    }
    const py::sequence shape = py::reinterpret_borrow<py::sequence>(shape_object);
    if (py::len(shape) != 4) {
        throw py::value_error("shape must have exactly four dimensions");
    }
    std::array<std::size_t, 4> extents{};
    for (py::ssize_t axis = 0; axis < 4; ++axis) {
        const py::handle value = shape[axis];
        if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value)) {
            throw py::type_error("shape dimensions must be integers");
        }
        const py::ssize_t extent = py::cast<py::ssize_t>(value);
        if (extent <= 0) {
            throw py::value_error("shape dimensions must be positive");
        }
        extents[static_cast<std::size_t>(axis)] = static_cast<std::size_t>(extent);
    }
    return {extents[0], extents[1], extents[2], extents[3]};
}

bool same_dimensions(const phase_a::Dimensions4D& left,
                     const phase_a::Dimensions4D& right)
{
    return left.scan_y == right.scan_y && left.scan_x == right.scan_x &&
           left.detector_y == right.detector_y && left.detector_x == right.detector_x;
}

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
        "Python bindings for Project 1 fixed and adaptive 4D-STEM median filters.";

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
    py::class_<phase_b::CudaAdaptiveMedianBuffer>(module, "CudaAdaptiveMedianBuffer")
        .def(
            py::init([](const py::object& shape_object) {
                const phase_a::Dimensions4D dimensions = validate_shape(shape_object);
                std::unique_ptr<phase_b::CudaAdaptiveMedianBuffer> buffer;
                {
                    py::gil_scoped_release release;
                    buffer = std::make_unique<phase_b::CudaAdaptiveMedianBuffer>(dimensions);
                }
                return buffer;
            }),
            py::arg("shape"),
            "Allocate persistent input, output, plane-minimum, and fallback-flag buffers.")
        .def(
            "upload",
            [](phase_b::CudaAdaptiveMedianBuffer& buffer, const py::object& input_object) {
                const ValidatedInput input = validate_input_metadata(input_object);
                if (!same_dimensions(input.dimensions, buffer.dimensions())) {
                    throw py::value_error("input shape does not match the allocated shape");
                }
                validate_finite_values(input);
                {
                    // The Python owner remains alive until synchronous H2D completes.
                    py::gil_scoped_release release;
                    buffer.upload(input.data);
                }
            },
            py::arg("array"),
            "Replace the resident input explicitly; invalidates the prior output.")
        .def(
            "filter",
            [](phase_b::CudaAdaptiveMedianBuffer& buffer) {
                py::gil_scoped_release release;
                buffer.filter();
            },
            "Run plane minimum, balanced 3x3 common path, and rare fallback on the uploaded input.")
        .def(
            "download",
            [](const phase_b::CudaAdaptiveMedianBuffer& buffer) {
                const auto dimensions = buffer.dimensions();
                const std::array<py::ssize_t, 4> shape = {
                    static_cast<py::ssize_t>(dimensions.scan_y),
                    static_cast<py::ssize_t>(dimensions.scan_x),
                    static_cast<py::ssize_t>(dimensions.detector_y),
                    static_cast<py::ssize_t>(dimensions.detector_x),
                };
                py::array_t<double> output(shape);
                double* const output_data = output.mutable_data();
                {
                    py::gil_scoped_release release;
                    buffer.download(output_data);
                }
                return output;
            },
            "Download the resident result to a new independent NumPy array.")
        .def_property_readonly(
            "shape",
            [](const phase_b::CudaAdaptiveMedianBuffer& buffer) {
                const auto dimensions = buffer.dimensions();
                return py::make_tuple(dimensions.scan_y, dimensions.scan_x,
                                      dimensions.detector_y, dimensions.detector_x);
            },
            "The fixed resident shape in scan_y, scan_x, detector_y, detector_x order.");

    py::class_<phase_a::CudaMedianBuffer>(module, "CudaMedianBuffer")
        .def(
            py::init([](const py::object& input_object) {
                const ValidatedInput input =
                    validate_input_metadata(input_object);
                validate_finite_values(input);

                std::unique_ptr<phase_a::CudaMedianBuffer> buffer;
                {
                    // input_object remains alive while the synchronous upload
                    // reads its validated NumPy buffer without a host-side copy.
                    py::gil_scoped_release release;
                    buffer = std::make_unique<phase_a::CudaMedianBuffer>(
                        input.data,
                        input.dimensions);
                }
                return buffer;
            }),
            py::arg("array"),
            "Allocate resident CUDA buffers and upload one validated input.")
        .def(
            "filter",
            [](phase_a::CudaMedianBuffer& buffer) {
                py::gil_scoped_release release;
                // Every call reads the original resident input and overwrites
                // the resident output; calls do not chain output back to input.
                buffer.filter();
            },
            "Apply the fixed median to the originally uploaded input.")
        .def(
            "download",
            [](const phase_a::CudaMedianBuffer& buffer) {
                const phase_a::Dimensions4D dimensions = buffer.dimensions();
                const std::array<py::ssize_t, 4> shape = {
                    static_cast<py::ssize_t>(dimensions.scan_y),
                    static_cast<py::ssize_t>(dimensions.scan_x),
                    static_cast<py::ssize_t>(dimensions.detector_y),
                    static_cast<py::ssize_t>(dimensions.detector_x),
                };
                py::array_t<double> output(shape);
                double* const output_data = output.mutable_data();
                {
                    py::gil_scoped_release release;
                    buffer.download(output_data);
                }
                return output;
            },
            "Download the current resident output into a new NumPy array.")
        .def_property_readonly(
            "shape",
            [](const phase_a::CudaMedianBuffer& buffer) {
                const phase_a::Dimensions4D dimensions = buffer.dimensions();
                return py::make_tuple(
                    dimensions.scan_y,
                    dimensions.scan_x,
                    dimensions.detector_y,
                    dimensions.detector_x);
            },
            "The resident array shape in scan_y, scan_x, detector_y, detector_x order.");

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
