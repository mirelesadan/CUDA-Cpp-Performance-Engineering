cmake_minimum_required(VERSION 3.24)
project(project2_kmeans_serial LANGUAGES CXX)

# Standalone CPU-only build; Project 1's OpenMP/CUDA/Python options do not apply.
add_library(project2_kmeans_serial STATIC src/kmeans_serial.cpp)
target_include_directories(project2_kmeans_serial PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(project2_kmeans_serial PUBLIC cxx_std_17)
set_target_properties(project2_kmeans_serial PROPERTIES CXX_EXTENSIONS OFF)

# The frozen assignment needs separate, ordered FP32 subtract/multiply/add.
# Never enable fast-math or floating-point contraction for this baseline.
if(MSVC)
    target_compile_options(project2_kmeans_serial PRIVATE /fp:precise)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(project2_kmeans_serial PRIVATE -ffp-contract=off)
endif()

add_executable(phase2_kmeans_serial src/serial_cli.cpp)
target_link_libraries(phase2_kmeans_serial PRIVATE project2_kmeans_serial)

add_executable(phase2_kmeans_serial_contract_tests src/serial_contract_tests.cpp)
target_link_libraries(phase2_kmeans_serial_contract_tests PRIVATE project2_kmeans_serial)
enable_testing()
add_test(NAME phase2_kmeans_serial_contract_tests
         COMMAND phase2_kmeans_serial_contract_tests)

# Profiling-only targets use the identical source and strict FP behavior.
# Default Release remains free of phase timers and profiling symbols.
option(PROJECT2_ENABLE_PHASE_TIMING "Build separate coarse phase-timing target" OFF)
option(PROJECT2_ENABLE_PROFILING_SYMBOLS "Emit Release source symbols for sampling" OFF)
option(PROJECT2_ENABLE_OPENMP "Build the assignment-only OpenMP implementation" OFF)

if(PROJECT2_ENABLE_PHASE_TIMING)
    add_library(project2_kmeans_phase_timing STATIC src/kmeans_serial.cpp)
    target_include_directories(project2_kmeans_phase_timing PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(project2_kmeans_phase_timing PUBLIC cxx_std_17)
    target_compile_definitions(project2_kmeans_phase_timing PUBLIC KMEANS_PHASE_TIMING)
    set_target_properties(project2_kmeans_phase_timing PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(project2_kmeans_phase_timing PRIVATE /fp:precise)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(project2_kmeans_phase_timing PRIVATE -ffp-contract=off)
    endif()
    add_executable(phase2_kmeans_serial_phase_timing src/serial_cli.cpp)
    target_link_libraries(phase2_kmeans_serial_phase_timing
        PRIVATE project2_kmeans_phase_timing)
endif()

if(PROJECT2_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    add_library(project2_kmeans_openmp STATIC src/kmeans_openmp.cpp)
    target_include_directories(project2_kmeans_openmp PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(project2_kmeans_openmp PUBLIC cxx_std_17)
    set_target_properties(project2_kmeans_openmp PROPERTIES CXX_EXTENSIONS OFF)
    target_link_libraries(project2_kmeans_openmp PUBLIC
        project2_kmeans_serial OpenMP::OpenMP_CXX)
    if(MSVC)
        target_compile_options(project2_kmeans_openmp PRIVATE /fp:precise)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(project2_kmeans_openmp PRIVATE -ffp-contract=off)
    endif()

    add_executable(phase2_kmeans_openmp_scaling src/openmp_scaling.cpp)
    target_link_libraries(phase2_kmeans_openmp_scaling PRIVATE project2_kmeans_openmp)

    add_executable(phase2_kmeans_openmp_contract_tests src/openmp_contract_tests.cpp)
    target_link_libraries(phase2_kmeans_openmp_contract_tests PRIVATE
        project2_kmeans_openmp)
    add_test(NAME phase2_kmeans_openmp_contract_tests
             COMMAND phase2_kmeans_openmp_contract_tests)

    if(PROJECT2_ENABLE_PHASE_TIMING)
        add_library(project2_kmeans_openmp_phase_timing STATIC src/kmeans_openmp.cpp)
        target_include_directories(project2_kmeans_openmp_phase_timing PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_features(project2_kmeans_openmp_phase_timing PUBLIC cxx_std_17)
        target_compile_definitions(project2_kmeans_openmp_phase_timing PUBLIC
            KMEANS_PHASE_TIMING)
        set_target_properties(project2_kmeans_openmp_phase_timing PROPERTIES
            CXX_EXTENSIONS OFF)
        target_link_libraries(project2_kmeans_openmp_phase_timing PUBLIC
            project2_kmeans_phase_timing OpenMP::OpenMP_CXX)
        if(MSVC)
            target_compile_options(project2_kmeans_openmp_phase_timing PRIVATE /fp:precise)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(project2_kmeans_openmp_phase_timing PRIVATE
                -ffp-contract=off)
        endif()
        add_executable(phase2_kmeans_openmp_scaling_phase_timing
            src/openmp_scaling.cpp)
        target_link_libraries(phase2_kmeans_openmp_scaling_phase_timing PRIVATE
            project2_kmeans_openmp_phase_timing)
    endif()
endif()

if(MSVC AND PROJECT2_ENABLE_PROFILING_SYMBOLS)
    target_compile_options(project2_kmeans_serial PRIVATE
        $<$<CONFIG:Release>:/Zi> $<$<CONFIG:Release>:/Zo>)
    foreach(target phase2_kmeans_serial phase2_kmeans_serial_contract_tests)
        target_compile_options(${target} PRIVATE
            $<$<CONFIG:Release>:/Zi> $<$<CONFIG:Release>:/Zo>)
        target_link_options(${target} PRIVATE
            $<$<CONFIG:Release>:/DEBUG:FULL>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>)
    endforeach()
    if(PROJECT2_ENABLE_PHASE_TIMING)
        target_compile_options(project2_kmeans_phase_timing PRIVATE
            $<$<CONFIG:Release>:/Zi> $<$<CONFIG:Release>:/Zo>)
        target_compile_options(phase2_kmeans_serial_phase_timing PRIVATE
            $<$<CONFIG:Release>:/Zi> $<$<CONFIG:Release>:/Zo>)
        target_link_options(phase2_kmeans_serial_phase_timing PRIVATE
            $<$<CONFIG:Release>:/DEBUG:FULL>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>)
    endif()
    add_executable(phase2_kmeans_serial_sampler src/serial_sampler.cpp)
    target_link_libraries(phase2_kmeans_serial_sampler PRIVATE
        project2_kmeans_serial dbghelp winmm)
    target_compile_options(phase2_kmeans_serial_sampler PRIVATE
        $<$<CONFIG:Release>:/Zi> $<$<CONFIG:Release>:/Zo>)
    target_link_options(phase2_kmeans_serial_sampler PRIVATE
        $<$<CONFIG:Release>:/DEBUG:FULL>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>)
endif()
