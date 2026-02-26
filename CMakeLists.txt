cmake_minimum_required(VERSION 3.18)

# Project configuration
project(GPUCompression LANGUAGES CXX CUDA C)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Optional: Find HDF5 for filter plugin
find_package(HDF5 COMPONENTS C)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /tmp/include
    /usr/local/cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link directories
link_directories(
    /tmp/lib
)

# ============================================================
# Core Sources (reusable across targets)
# ============================================================
set(FACTORY_SOURCES
    src/compression/compression_factory.cpp
    src/compression/compression_factory.hpp
)

set(PREPROCESSING_SOURCES
    src/preprocessing/byte_shuffle_kernels.cu
    src/preprocessing/quantization_kernels.cu
)

set(LIB_SOURCES
    src/api/gpucompress_api.cpp
    src/stats/entropy_kernel.cu
    src/stats/stats_cpu.cpp
    src/nn/nn_gpu.cu
    src/nn/nn_reinforce.cpp
    src/stats/stats_kernel.cu
    src/nn/experience_buffer.cpp
)

# Set language property for .cpp files that use CUDA
set_source_files_properties(
    src/compression/compression_factory.cpp
    src/cli/compress.cpp
    src/cli/decompress.cpp
    src/api/gpucompress_api.cpp
    src/nn/nn_reinforce.cpp
    src/nn/experience_buffer.cpp
    PROPERTIES LANGUAGE CUDA
)

# ============================================================
# Shared Library: libgpucompress.so
# ============================================================
add_library(gpucompress SHARED
    ${LIB_SOURCES}
    ${PREPROCESSING_SOURCES}
    ${FACTORY_SOURCES}
)

target_include_directories(gpucompress
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(gpucompress
    PRIVATE
    nvcomp
    CUDA::cudart
)

set_target_properties(gpucompress PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
    OUTPUT_NAME gpucompress
)

# ============================================================
# HDF5 Filter Plugin: libH5Zgpucompress.so (Optional)
# ============================================================
if(HDF5_FOUND)
    message(STATUS "HDF5 found: ${HDF5_VERSION}")
    message(STATUS "  Include dirs: ${HDF5_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${HDF5_C_LIBRARIES}")

    add_library(H5Zgpucompress SHARED
        src/hdf5/H5Zgpucompress.c
    )

    target_include_directories(H5Zgpucompress
        PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hdf5
    )

    target_link_libraries(H5Zgpucompress
        PRIVATE
        gpucompress
        ${HDF5_C_LIBRARIES}
    )

    set_target_properties(H5Zgpucompress PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        VERSION 1.0.0
        SOVERSION 1
        OUTPUT_NAME H5Zgpucompress
    )

    # Define for plugin exports
    target_compile_definitions(H5Zgpucompress PRIVATE GPUCOMPRESS_BUILD_HDF5_PLUGIN)

    # HDF5 filter integration test
    add_executable(test_hdf5_filter tests/test_hdf5_filter.c)
    target_include_directories(test_hdf5_filter PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(test_hdf5_filter PRIVATE
        gpucompress
        H5Zgpucompress
        ${HDF5_C_LIBRARIES}
        m
    )

    # HDF5 plugin unit tests
    add_executable(test_hdf5_plugin tests/test_hdf5_plugin.c)
    target_include_directories(test_hdf5_plugin PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(test_hdf5_plugin PRIVATE
        gpucompress
        H5Zgpucompress
        ${HDF5_C_LIBRARIES}
        m
    )

else()
    message(STATUS "HDF5 not found - skipping filter plugin libH5Zgpucompress.so")
    message(STATUS "  To build HDF5 plugin, install HDF5 development libraries:")
    message(STATUS "    Ubuntu/Debian: sudo apt-get install libhdf5-dev")
    message(STATUS "    RHEL/CentOS:   sudo yum install hdf5-devel")
endif()

# ============================================================
# Executable 1: GPU Compress (with verification)
# ============================================================
add_executable(gpu_compress
    src/cli/compress.cpp
    ${PREPROCESSING_SOURCES}
    ${FACTORY_SOURCES}
)

target_link_libraries(gpu_compress
    PRIVATE
    nvcomp
    cufile
    CUDA::cudart
)

set_target_properties(gpu_compress PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable 2: GPU Decompress
# ============================================================
add_executable(gpu_decompress
    src/cli/decompress.cpp
    ${PREPROCESSING_SOURCES}
    ${FACTORY_SOURCES}
)

target_link_libraries(gpu_decompress
    PRIVATE
    nvcomp
    cufile
    CUDA::cudart
)

set_target_properties(gpu_decompress PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable 3: Quantization Test Suite
# ============================================================
add_executable(test_quantization
    tests/quantization/test_quantization_suite.cu
    ${PREPROCESSING_SOURCES}
)

target_link_libraries(test_quantization
    PRIVATE
    CUDA::cudart
    CUDA::curand
)

set_target_properties(test_quantization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable 4: Benchmark
# ============================================================
add_executable(benchmark scripts/benchmark.cpp)
target_link_libraries(benchmark PRIVATE gpucompress pthread)

# ============================================================
# Executable: Compression Core Test Suite
# ============================================================
add_executable(test_compression_core
    tests/test_compression_core.cu
    ${PREPROCESSING_SOURCES}
)
target_link_libraries(test_compression_core PRIVATE CUDA::cudart)
set_target_properties(test_compression_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable: Preprocessing Test Suite
# ============================================================
add_executable(test_preprocessing
    tests/test_preprocessing.cu
    ${PREPROCESSING_SOURCES}
)
target_link_libraries(test_preprocessing PRIVATE CUDA::cudart)
set_target_properties(test_preprocessing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable: Stats Test Suite
# ============================================================
add_executable(test_stats tests/test_stats.cu)
target_link_libraries(test_stats PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_stats PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable: CLI Test Suite
# ============================================================
add_executable(test_cli tests/test_cli.cu)
target_link_libraries(test_cli PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_cli PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable: NN Test Suite (nn/ fixes)
# ============================================================
add_executable(test_nn tests/test_nn.cu)
target_link_libraries(test_nn PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_nn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable: API Test Suite
# ============================================================
add_executable(test_api tests/test_api.cu)
target_link_libraries(test_api PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_api PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable 5: NN Reinforce Test Suite
# ============================================================
add_executable(test_nn_reinforce tests/test_nn_reinforce.cpp)
set_source_files_properties(tests/test_nn_reinforce.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_nn_reinforce PRIVATE gpucompress CUDA::cudart pthread)

# ============================================================
# Executable 6: NN Pipeline End-to-End Test
# ============================================================
add_executable(test_nn_pipeline tests/test_nn_pipeline.cpp)
target_link_libraries(test_nn_pipeline PRIVATE gpucompress pthread)

# ============================================================
# Executable 7: Simulation Evaluation
# ============================================================
add_executable(eval_simulation eval/eval_simulation.cpp)
target_link_libraries(eval_simulation PRIVATE gpucompress pthread)

# ============================================================
# Executable 7: Active Learning Integration Test
# ============================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/neural_net/test_active_learning.cu")
    add_executable(test_active_learning
        neural_net/test_active_learning.cu
    )

    target_link_libraries(test_active_learning
        PRIVATE
        gpucompress
        CUDA::cudart
    )

    set_target_properties(test_active_learning PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
else()
    message(STATUS "neural_net/test_active_learning.cu not found - skipping test_active_learning target")
endif()

# ============================================================
# Executable: NN Shuffle Selection Test
# ============================================================
add_executable(test_nn_shuffle tests/test_nn_shuffle.cu)
target_link_libraries(test_nn_shuffle PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_nn_shuffle PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable: Shuffle Test Suite
# ============================================================
add_executable(test_shuffle tests/test_shuffle.cu)
target_link_libraries(test_shuffle PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_shuffle PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable: CPU Stats Correctness Test
# ============================================================
add_executable(test_cpu_stats tests/test_cpu_stats.cu)
target_link_libraries(test_cpu_stats PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_cpu_stats PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable: 128MB Experiment
# ============================================================
add_executable(experiment_128mb tests/experiment_128mb.cu)
target_link_libraries(experiment_128mb PRIVATE gpucompress CUDA::cudart)
set_target_properties(experiment_128mb PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================
# Executable: HDF5 128MB Experiment
# ============================================================
if(HDF5_FOUND)
    add_executable(experiment_hdf5_128mb tests/experiment_hdf5_128mb.c)
    target_include_directories(experiment_hdf5_128mb PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(experiment_hdf5_128mb PRIVATE
        gpucompress
        H5Zgpucompress
        ${HDF5_C_LIBRARIES}
        m
    )

    add_executable(experiment_hdf5_mixed tests/experiment_hdf5_mixed.c)
    target_include_directories(experiment_hdf5_mixed PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(experiment_hdf5_mixed PRIVATE
        gpucompress
        H5Zgpucompress
        ${HDF5_C_LIBRARIES}
        m
    )

    # HDF5 Benchmark: No Compression vs Static vs Dynamic (NN)
    add_executable(benchmark_hdf5 tests/benchmark_hdf5.c)
    target_include_directories(benchmark_hdf5 PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(benchmark_hdf5 PRIVATE
        gpucompress
        H5Zgpucompress
        ${HDF5_C_LIBRARIES}
        m
    )
endif()

# ============================================================
# Installation
# ============================================================
include(GNUInstallDirs)

# Install shared library
install(TARGETS gpucompress
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Install executables
install(TARGETS gpu_compress gpu_decompress
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install HDF5 plugin to standard plugin directory
if(HDF5_FOUND)
    install(TARGETS H5Zgpucompress
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/hdf5/plugin
    )
endif()

# ============================================================
# Build Summary
# ============================================================
message(STATUS "")
message(STATUS "=== GPUCompress Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  - libgpucompress.so    (shared library with C API)")
message(STATUS "  - gpu_compress         (compression executable)")
message(STATUS "  - gpu_decompress       (decompression executable)")
message(STATUS "  - test_quantization    (quantization test suite)")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/neural_net/test_active_learning.cu")
    message(STATUS "  - test_active_learning (active learning integration test)")
endif()
if(HDF5_FOUND)
    message(STATUS "  - libH5Zgpucompress.so (HDF5 filter plugin)")
endif()
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  - 8 compression algorithms (LZ4, Snappy, Deflate, Gdeflate, Zstd, ANS, Cascaded, Bitcomp)")
message(STATUS "  - GPU-accelerated entropy calculation")
message(STATUS "  - Neural network based algorithm selection (.nnwt)")
message(STATUS "  - Error-bound quantization (lossy)")
message(STATUS "  - Byte shuffle preprocessing")
message(STATUS "  - GPUDirect Storage (GDS) support")
if(HDF5_FOUND)
    message(STATUS "  - HDF5 filter plugin (set HDF5_PLUGIN_PATH to use)")
endif()
message(STATUS "")
