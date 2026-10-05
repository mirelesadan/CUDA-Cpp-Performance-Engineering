#include "kmeans_cuda.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

std::size_t positive_integer(py::handle value, const char* name) {
    if (!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value)) {
        throw py::type_error(std::string(name) + " must be a Python integer");
    }
    const auto parsed = PyLong_AsLongLong(value.ptr());
    if (PyErr_Occurred()) throw py::error_already_set();
    if (parsed <= 0) throw py::value_error(std::string(name) + " must be positive");
    return static_cast<std::size_t>(parsed);
}

struct Input {
    std::size_t n, d;
    const void* data;
};

Input input_metadata(py::handle object) {
    if (!py::isinstance<py::array>(object)) {
        throw py::type_error("input must be a NumPy ndarray");
    }
    const auto array = py::reinterpret_borrow<py::array>(object);
    if (array.ndim() != 2) throw py::value_error("input must have exactly two dimensions");
    if (!array.dtype().is(py::dtype::of<float>())) {
        throw py::type_error("input dtype must be exactly native NumPy float32");
    }
    if (!(array.flags() & py::array::c_style)) {
        throw py::value_error("input must be C-contiguous");
    }
    if (array.shape(0) < 2 || array.shape(0) > (1u << 20) ||
        array.shape(1) < 1 || array.shape(1) > 32) {
        throw py::value_error("require 2 <= N <= 2^20 and 1 <= D <= 32");
    }
    return {static_cast<std::size_t>(array.shape(0)),
            static_cast<std::size_t>(array.shape(1)), array.data()};
}

std::vector<float> copy_input(const Input& input) {
    // An explicit NumPy -> native vector copy, with GIL held. memcpy also
    // supports read-only and unaligned contiguous NumPy buffers. The existing
    // native API performs the single authoritative finite/magnitude scan.
    std::vector<float> values(input.n * input.d);
    std::memcpy(values.data(), input.data, values.size() * sizeof(float));
    return values;
}

py::dict metadata_result(const kmeans::FitMetadata& result) {
    py::dict output;
    output["update_count"] = result.update_count;
    output["converged"] = result.converged;
    return output;
}

py::dict numpy_result(const kmeans::Result& result, std::size_t n,
                      std::size_t d, std::size_t k) {
    if (result.labels.size() != n || result.centroids.size() != k * d) {
        throw std::runtime_error("native result dimensions do not match the input");
    }
    py::array_t<std::int32_t> labels(static_cast<py::ssize_t>(n));
    py::array_t<float> centroids({static_cast<py::ssize_t>(k),
                                static_cast<py::ssize_t>(d)});
    // Explicit native result vectors -> new NumPy-owned arrays. No borrowed
    // views or capsules outlive temporary vectors; GIL is reacquired here.
    std::memcpy(labels.mutable_data(), result.labels.data(), n * sizeof(std::int32_t));
    std::memcpy(centroids.mutable_data(), result.centroids.data(), k * d * sizeof(float));
    py::dict output = metadata_result({result.update_count, result.converged});
    output["labels"] = std::move(labels);
    output["centroids"] = std::move(centroids);
    return output;
}

py::dict one_shot(const py::object& array, py::handle clusters, std::size_t cap) {
    const auto input = input_metadata(array);
    const auto k = positive_integer(clusters, "K");
    if (k < 2 || k > 32 || k > input.n) {
        throw py::value_error("require 2 <= K <= min(N,32)");
    }
    if (cap < 1 || cap > 100) throw py::value_error("update cap must be 1..100");
    const auto values = copy_input(input);
    kmeans::Result result;
    {
        py::gil_scoped_release release;
        result = cap == 100
            ? kmeans::kmeans_cuda_parallel_count(values, input.n, input.d, k)
            : kmeans::kmeans_cuda_parallel_count_with_update_cap(
                  values, input.n, input.d, k, cap);
    }
    return numpy_result(result, input.n, input.d, k);
}

// Exactly one native owner; only fixed host dimensions and a call guard are
// added. The native owner is not thread-safe. Reject overlapping calls on the
// same Python object instead of racing while the GIL is released.
class PythonCudaBuffer {
public:
    const std::size_t n, d, k;
    PythonCudaBuffer(std::size_t samples, std::size_t features, std::size_t clusters)
        : n(samples), d(features), k(clusters),
          owner_(std::make_unique<kmeans::CudaKMeansBuffer>(n, d, k)) {}

    ~PythonCudaBuffer() noexcept {
        // Python normally destroys this wrapper with the GIL held. The native
        // destructor touches no Python objects and may wait while freeing.
        if (PyGILState_Check()) {
            py::gil_scoped_release release;
            owner_.reset();
        } else {
            owner_.reset();
        }
    }

    void upload(const py::object& array) {
        auto guard = acquire();
        const auto input = input_metadata(array);
        if (input.n != n || input.d != d) {
            throw py::value_error("upload shape must match the owner's fixed (N,D)");
        }
        const auto values = copy_input(input);
        py::gil_scoped_release release;
        owner_->upload(values);  // one O(ND) validation scan, then H2D
    }

    py::dict fit(std::size_t cap) {
        auto guard = acquire();
        kmeans::FitMetadata metadata;
        {
            py::gil_scoped_release release;
            metadata = cap == 100 ? owner_->fit() : owner_->fit_with_update_cap(cap);
        }
        return metadata_result(metadata); // no labels/centroids downloaded
    }

    py::dict download() {
        auto guard = acquire();
        kmeans::Result result;
        {
            py::gil_scoped_release release;
            result = owner_->download();
        }
        return numpy_result(result, n, d, k);
    }

private:
    std::unique_lock<std::mutex> acquire() {
        std::unique_lock<std::mutex> guard(mutex_, std::try_to_lock);
        if (!guard.owns_lock()) {
            throw std::runtime_error("CUDA owner is busy; concurrent calls on one object are unsupported");
        }
        return guard;
    }
    std::unique_ptr<kmeans::CudaKMeansBuffer> owner_;
    std::mutex mutex_;
};

} // namespace

PYBIND11_MODULE(kmeans_native, module) {
    module.doc() = "Project 2 synchronous CUDA K-means with explicit host copies and residency.";
    module.def("kmeans_cuda", [](const py::object& array, py::handle k) {
        return one_shot(array, k, 100);
    }, py::arg("array"), py::arg("K"), "Run the retained complete one-shot CUDA fit.");

    py::class_<PythonCudaBuffer>(module, "CudaKMeansBuffer")
        .def(py::init([](py::handle samples, py::handle features, py::handle clusters) {
            const auto n = positive_integer(samples, "N");
            const auto d = positive_integer(features, "D");
            const auto k = positive_integer(clusters, "K");
            std::unique_ptr<PythonCudaBuffer> buffer;
            {
                py::gil_scoped_release release;
                buffer = std::make_unique<PythonCudaBuffer>(n, d, k);
            }
            return buffer;
        }), py::arg("N"), py::arg("D"), py::arg("K"))
        .def("upload", &PythonCudaBuffer::upload, py::arg("array"),
             "Validate/copy/upload a fixed-shape input; invalidate previous output.")
        .def("fit", [](PythonCudaBuffer& owner) { return owner.fit(100); },
             "Run an independent fit of resident input and return only metadata.")
        .def("download", &PythonCudaBuffer::download,
             "Download to new independent NumPy arrays without consuming resident output.")
        .def("_fit_with_update_cap", [](PythonCudaBuffer& owner, py::handle cap) {
            return owner.fit(positive_integer(cap, "update cap"));
        }, py::arg("max_updates"), "Diagnostic hook for frozen nonconvergence validation only.");

    // Underscored reduced-cap hooks only support the frozen public fixture.
    // Native benchmark/profiler entry points are not exposed by this module.
    module.def("_kmeans_cuda_with_update_cap", [](const py::object& array,
                                                 py::handle k, py::handle cap) {
        return one_shot(array, k, positive_integer(cap, "update cap"));
    }, py::arg("array"), py::arg("K"), py::arg("max_updates"));
}
