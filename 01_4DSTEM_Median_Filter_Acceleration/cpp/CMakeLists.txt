cmake_minimum_required(VERSION 3.24)

project(fourdstem_median_filter LANGUAGES CXX)

find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(phase_a_fixed_median
    src/main.cpp
    src/fixed_median.cpp
)

add_executable(phase_a_baseline_benchmark
    src/benchmark.cpp
    src/fixed_median.cpp
)

add_executable(phase_b_adaptive_median_validation
    src/adaptive_validation.cpp
    src/adaptive_median.cpp
    src/fixed_median.cpp
)

add_executable(phase_b_adaptive_median_benchmark
    src/adaptive_benchmark.cpp
    src/adaptive_median.cpp
    src/fixed_median.cpp
)

target_include_directories(phase_a_fixed_median
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libnpy/include"
)

target_include_directories(phase_a_baseline_benchmark
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libnpy/include"
)

target_include_directories(phase_b_adaptive_median_validation
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libnpy/include"
)

target_include_directories(phase_b_adaptive_median_benchmark
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libnpy/include"
)

target_link_libraries(phase_a_fixed_median PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(phase_a_baseline_benchmark PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(phase_b_adaptive_median_validation PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(phase_b_adaptive_median_benchmark PRIVATE OpenMP::OpenMP_CXX)

# Embed the source-tree fixture paths so the executable does not depend on the
# directory from which it is launched. Different paths can still be supplied
# through command-line arguments.
set(PHASE_A_REFERENCE_INPUT_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../reference_data/public_synthetic/reference_input.npy"
)
set(PHASE_A_REFERENCE_OUTPUT_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../reference_data/public_synthetic/reference_output_python.npy"
)

set(PHASE_A_BENCHMARK_INPUT_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../benchmark_data/median_filter_input.npy"
)

set(PHASE_B_REFERENCE_INPUT_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../reference_data/public_adaptive/reference_input.npy"
)
set(PHASE_B_REFERENCE_OUTPUT_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../reference_data/public_adaptive/reference_output_python.npy"
)

target_compile_definitions(phase_a_fixed_median
    PRIVATE
        PHASE_A_REFERENCE_INPUT_PATH="${PHASE_A_REFERENCE_INPUT_PATH}"
        PHASE_A_REFERENCE_OUTPUT_PATH="${PHASE_A_REFERENCE_OUTPUT_PATH}"
)

target_compile_definitions(phase_a_baseline_benchmark
    PRIVATE
        PHASE_A_BENCHMARK_INPUT_PATH="${PHASE_A_BENCHMARK_INPUT_PATH}"
)

target_compile_definitions(phase_b_adaptive_median_validation
    PRIVATE
        PHASE_B_REFERENCE_INPUT_PATH="${PHASE_B_REFERENCE_INPUT_PATH}"
        PHASE_B_REFERENCE_OUTPUT_PATH="${PHASE_B_REFERENCE_OUTPUT_PATH}"
)

target_compile_definitions(phase_b_adaptive_median_benchmark
    PRIVATE
        PHASE_B_BENCHMARK_INPUT_PATH="${PHASE_A_BENCHMARK_INPUT_PATH}"
)

set_target_properties(phase_a_fixed_median PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

set_target_properties(phase_a_baseline_benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

set_target_properties(phase_b_adaptive_median_validation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

set_target_properties(phase_b_adaptive_median_benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Preserve the normal optimized Release configuration while optionally
# emitting symbols for line- and function-level CPU sampling analysis.
option(PHASE_A_ENABLE_PROFILING_SYMBOLS "Emit optimized Release profiling symbols" OFF)

if(MSVC AND PHASE_A_ENABLE_PROFILING_SYMBOLS)
    target_compile_options(phase_a_baseline_benchmark
        PRIVATE
            $<$<CONFIG:Release>:/Zi>
            $<$<CONFIG:Release>:/Zo>
    )
    target_link_options(phase_a_baseline_benchmark
        PRIVATE
            $<$<CONFIG:Release>:/DEBUG:FULL>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>
    )
endif()

option(PHASE_B_ENABLE_PROFILING_SYMBOLS
    "Emit optimized Release symbols for the Phase B benchmark" OFF
)

if(MSVC AND PHASE_B_ENABLE_PROFILING_SYMBOLS)
    target_compile_options(phase_b_adaptive_median_benchmark
        PRIVATE
            $<$<CONFIG:Release>:/Zi>
            $<$<CONFIG:Release>:/Zo>
    )
    target_link_options(phase_b_adaptive_median_benchmark
        PRIVATE
            $<$<CONFIG:Release>:/DEBUG:FULL>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>
    )
endif()

# Keep the portable CPU-only build available by default. CUDA targets are
# enabled explicitly on systems with a CUDA compiler and runtime.
option(PHASE_A_ENABLE_CUDA "Build the correctness-first CUDA baseline targets" OFF)
option(PHASE_A_ENABLE_PYTHON "Build the correctness-first pybind11 Python module" OFF)
option(PHASE_A_ENABLE_PYTHON_COPY_BENCHMARK
    "Expose private copied CPU paths for the Python interface experiment" OFF
)
set(PHASE_A_CUDA_ARCHITECTURES "89" CACHE STRING
    "CUDA architectures for Phase A targets (for example, 89 for Ada)"
)

if(PHASE_A_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(Threads REQUIRED)

    add_library(phase_a_cuda_baseline STATIC
        src/fixed_median_cuda.cu
    )
    target_include_directories(phase_a_cuda_baseline
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(phase_a_cuda_baseline PUBLIC CUDA::cudart Threads::Threads)
    set_target_properties(phase_a_cuda_baseline PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS NO
        CUDA_ARCHITECTURES "${PHASE_A_CUDA_ARCHITECTURES}"
    )

    add_library(phase_b_adaptive_cuda_baseline STATIC
        src/adaptive_median_cuda.cu
    )
    target_include_directories(phase_b_adaptive_cuda_baseline
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(phase_b_adaptive_cuda_baseline PUBLIC CUDA::cudart)
    set_target_properties(phase_b_adaptive_cuda_baseline PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS NO
        CUDA_ARCHITECTURES "${PHASE_A_CUDA_ARCHITECTURES}"
    )

    add_executable(phase_a_cuda_validation
        src/cuda_validation.cpp
        src/fixed_median.cpp
    )
    add_executable(phase_a_cuda_benchmark
        src/cuda_benchmark.cpp
        src/fixed_median.cpp
    )
    add_executable(phase_a_cuda_kernel_benchmark
        src/cuda_kernel_benchmark.cpp
        src/fixed_median.cpp
    )
    add_executable(phase_a_cuda_transfer_characterization
        src/cuda_transfer_characterization.cpp
        src/fixed_median.cpp
    )
    add_executable(phase_b_adaptive_cuda_validation
        src/adaptive_cuda_validation.cpp
        src/adaptive_median.cpp
        src/fixed_median.cpp
    )
    add_executable(phase_b_adaptive_cuda_benchmark
        src/adaptive_cuda_benchmark.cpp
        src/adaptive_median.cpp
        src/fixed_median.cpp
    )

    foreach(cuda_target IN ITEMS
            phase_a_cuda_validation
            phase_a_cuda_benchmark
            phase_a_cuda_kernel_benchmark
            phase_a_cuda_transfer_characterization)
        target_include_directories(${cuda_target}
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libnpy/include"
        )
        target_link_libraries(${cuda_target}
            PRIVATE
                phase_a_cuda_baseline
                OpenMP::OpenMP_CXX
        )
        set_target_properties(${cuda_target} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES
            CXX_EXTENSIONS NO
        )
    endforeach()

    foreach(adaptive_cuda_target IN ITEMS
            phase_b_adaptive_cuda_validation
            phase_b_adaptive_cuda_benchmark)
        target_include_directories(${adaptive_cuda_target}
            PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libnpy/include"
        )
        target_link_libraries(${adaptive_cuda_target}
            PRIVATE
                phase_b_adaptive_cuda_baseline
                OpenMP::OpenMP_CXX
        )
        set_target_properties(${adaptive_cuda_target} PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED YES
            CXX_EXTENSIONS NO
        )
    endforeach()

    target_compile_definitions(phase_a_cuda_validation
        PRIVATE
            PHASE_A_REFERENCE_INPUT_PATH="${PHASE_A_REFERENCE_INPUT_PATH}"
            PHASE_A_REFERENCE_OUTPUT_PATH="${PHASE_A_REFERENCE_OUTPUT_PATH}"
    )
    target_compile_definitions(phase_a_cuda_benchmark
        PRIVATE
            PHASE_A_BENCHMARK_INPUT_PATH="${PHASE_A_BENCHMARK_INPUT_PATH}"
    )
    target_compile_definitions(phase_a_cuda_kernel_benchmark
        PRIVATE
            PHASE_A_BENCHMARK_INPUT_PATH="${PHASE_A_BENCHMARK_INPUT_PATH}"
    )
    target_compile_definitions(phase_a_cuda_transfer_characterization
        PRIVATE
            PHASE_A_BENCHMARK_INPUT_PATH="${PHASE_A_BENCHMARK_INPUT_PATH}"
    )
    target_compile_definitions(phase_b_adaptive_cuda_validation
        PRIVATE
            PHASE_B_REFERENCE_INPUT_PATH="${PHASE_B_REFERENCE_INPUT_PATH}"
            PHASE_B_REFERENCE_OUTPUT_PATH="${PHASE_B_REFERENCE_OUTPUT_PATH}"
    )
    target_compile_definitions(phase_b_adaptive_cuda_benchmark
        PRIVATE
            PHASE_B_BENCHMARK_INPUT_PATH="${PHASE_A_BENCHMARK_INPUT_PATH}"
    )
endif()

if(PHASE_A_ENABLE_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        RESULT_VARIABLE PHASE_A_PYBIND11_RESULT
        OUTPUT_VARIABLE PHASE_A_PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE PHASE_A_PYBIND11_ERROR
    )
    if(NOT PHASE_A_PYBIND11_RESULT EQUAL 0)
        message(FATAL_ERROR
            "pybind11 is required in the selected Python environment: "
            "${PHASE_A_PYBIND11_ERROR}"
        )
    endif()

    find_package(pybind11 CONFIG REQUIRED
        PATHS "${PHASE_A_PYBIND11_CMAKE_DIR}"
        NO_DEFAULT_PATH
    )

    pybind11_add_module(fourdstem_median MODULE
        src/python_bindings.cpp
        src/fixed_median.cpp
    )
    target_include_directories(fourdstem_median
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(fourdstem_median PRIVATE OpenMP::OpenMP_CXX)
    set_target_properties(fourdstem_median PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )

    if(PHASE_A_ENABLE_PYTHON_COPY_BENCHMARK)
        target_compile_definitions(fourdstem_median
            PRIVATE PHASE_A_PYTHON_COPY_BENCHMARK=1
        )
    endif()

    if(PHASE_A_ENABLE_CUDA)
        target_link_libraries(fourdstem_median PRIVATE phase_a_cuda_baseline)
        target_compile_definitions(fourdstem_median
            PRIVATE PHASE_A_PYTHON_CUDA_ENABLED=1
        )
        if(WIN32)
            # Python 3.8+ uses a restricted DLL search path for extension
            # dependencies. Keep the CUDA runtime beside the local module.
            find_file(PHASE_A_CUDART_DLL
                NAMES "cudart64_${CUDAToolkit_VERSION_MAJOR}.dll"
                HINTS "${CUDAToolkit_BIN_DIR}"
                REQUIRED
                NO_DEFAULT_PATH
            )
            add_custom_command(TARGET fourdstem_median POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${PHASE_A_CUDART_DLL}"
                    "$<TARGET_FILE_DIR:fourdstem_median>"
                COMMENT "Copying the CUDA runtime beside the Python module"
            )
        endif()
    endif()
endif()
