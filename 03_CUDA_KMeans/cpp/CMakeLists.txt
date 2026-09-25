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
