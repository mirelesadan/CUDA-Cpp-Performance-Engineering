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

target_link_libraries(phase_a_fixed_median PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(phase_a_baseline_benchmark PRIVATE OpenMP::OpenMP_CXX)

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

target_compile_definitions(phase_a_fixed_median
    PRIVATE
        PHASE_A_REFERENCE_INPUT_PATH="${PHASE_A_REFERENCE_INPUT_PATH}"
        PHASE_A_REFERENCE_OUTPUT_PATH="${PHASE_A_REFERENCE_OUTPUT_PATH}"
)

target_compile_definitions(phase_a_baseline_benchmark
    PRIVATE
        PHASE_A_BENCHMARK_INPUT_PATH="${PHASE_A_BENCHMARK_INPUT_PATH}"
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
