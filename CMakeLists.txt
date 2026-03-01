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
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math")

# Target the actual GPU architecture (sm_80 = Ampere A100).
# Override at configure time: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to target")
endif()

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
    src/stats/stats_kernel.cu
)

# Set language property for .cpp files that use CUDA
set_source_files_properties(
    src/compression/compression_factory.cpp
    src/cli/compress.cpp
    src/cli/decompress.cpp
    src/api/gpucompress_api.cpp
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

    # HDF5 filter integration test (source missing)
    # add_executable(test_hdf5_filter tests/test_hdf5_filter.c)
    # target_include_directories(test_hdf5_filter PRIVATE
    #     ${HDF5_INCLUDE_DIRS}
    #     ${CMAKE_CURRENT_SOURCE_DIR}/include
    #     ${CMAKE_CURRENT_SOURCE_DIR}/src
    # )
    # target_link_libraries(test_hdf5_filter PRIVATE
    #     gpucompress
    #     H5Zgpucompress
    #     ${HDF5_C_LIBRARIES}
    #     m
    # )

    # HDF5 plugin unit tests (source missing)
    # add_executable(test_hdf5_plugin tests/test_hdf5_plugin.c)
    # target_include_directories(test_hdf5_plugin PRIVATE
    #     ${HDF5_INCLUDE_DIRS}
    #     ${CMAKE_CURRENT_SOURCE_DIR}/include
    #     ${CMAKE_CURRENT_SOURCE_DIR}/src
    # )
    # target_link_libraries(test_hdf5_plugin PRIVATE
    #     gpucompress
    #     H5Zgpucompress
    #     ${HDF5_C_LIBRARIES}
    #     m
    # )

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
# add_executable(test_quantization  # source missing
#     tests/quantization/test_quantization_suite.cu
#     ${PREPROCESSING_SOURCES}
# )
# target_link_libraries(test_quantization PRIVATE CUDA::cudart CUDA::curand)
# set_target_properties(test_quantization PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================
# Executable 4: Benchmark
# ============================================================
add_executable(benchmark scripts/benchmark.cpp)
target_link_libraries(benchmark PRIVATE gpucompress pthread)

# ============================================================
# Quantization round-trip visual verification test
# ============================================================
add_executable(test_quantization_roundtrip tests/test_quantization_roundtrip.c)
target_include_directories(test_quantization_roundtrip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_quantization_roundtrip PRIVATE gpucompress m)

# ============================================================
# Executable: Compression Core Test Suite
# ============================================================
# add_executable(test_compression_core  # source missing
#     tests/test_compression_core.cu
#     ${PREPROCESSING_SOURCES}
# )
# target_link_libraries(test_compression_core PRIVATE CUDA::cudart)
# set_target_properties(test_compression_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(test_preprocessing  # source missing
#     tests/test_preprocessing.cu
#     ${PREPROCESSING_SOURCES}
# )
# target_link_libraries(test_preprocessing PRIVATE CUDA::cudart)
# set_target_properties(test_preprocessing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================
# Executable: Stats Test Suite
# ============================================================
# add_executable(test_stats tests/test_stats.cu)  # source missing
# target_link_libraries(test_stats PRIVATE gpucompress CUDA::cudart)
# set_target_properties(test_stats PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(test_cli tests/test_cli.cu)  # source missing
# target_link_libraries(test_cli PRIVATE gpucompress CUDA::cudart)
# set_target_properties(test_cli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_nn tests/functionalityTests/test_nn.cu)
target_link_libraries(test_nn PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(test_api tests/test_api.cu)  # source missing
# target_link_libraries(test_api PRIVATE gpucompress CUDA::cudart)
# set_target_properties(test_api PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_nn_reinforce tests/functionalityTests/test_nn_reinforce.cpp)
set_source_files_properties(tests/functionalityTests/test_nn_reinforce.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(test_nn_reinforce PRIVATE gpucompress CUDA::cudart pthread)

add_executable(test_nn_pipeline tests/functionalityTests/test_nn_pipeline.cpp)
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
# add_executable(test_nn_shuffle tests/test_nn_shuffle.cu)  # source missing
# target_link_libraries(test_nn_shuffle PRIVATE gpucompress CUDA::cudart)
# set_target_properties(test_nn_shuffle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(test_shuffle tests/test_shuffle.cu)  # source missing
# target_link_libraries(test_shuffle PRIVATE gpucompress CUDA::cudart)
# set_target_properties(test_shuffle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(test_cpu_stats tests/test_cpu_stats.cu)  # source missing
# target_link_libraries(test_cpu_stats PRIVATE gpucompress CUDA::cudart)
# set_target_properties(test_cpu_stats PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(bench_stats tests/bench_stats.cu)  # source missing
# target_link_libraries(bench_stats PRIVATE gpucompress CUDA::cudart)
# set_target_properties(bench_stats PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================
# Executable: 128MB Experiment
# ============================================================
# add_executable(experiment_128mb tests/experiment_128mb.cu)
# target_link_libraries(experiment_128mb PRIVATE gpucompress CUDA::cudart)
# set_target_properties(experiment_128mb PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# ============================================================
# Executable: HDF5 128MB Experiment
# ============================================================
if(HDF5_FOUND)
    # add_executable(experiment_hdf5_128mb tests/experiment_hdf5_128mb.c)
    # target_include_directories(experiment_hdf5_128mb PRIVATE
    #     ${HDF5_INCLUDE_DIRS}
    #     ${CMAKE_CURRENT_SOURCE_DIR}/include
    #     ${CMAKE_CURRENT_SOURCE_DIR}/src
    # )
    # target_link_libraries(experiment_hdf5_128mb PRIVATE
    #     gpucompress
    #     H5Zgpucompress
    #     ${HDF5_C_LIBRARIES}
    #     m
    # )

    # add_executable(experiment_hdf5_mixed tests/experiment_hdf5_mixed.c)
    # target_include_directories(experiment_hdf5_mixed PRIVATE
    #     ${HDF5_INCLUDE_DIRS}
    #     ${CMAKE_CURRENT_SOURCE_DIR}/include
    #     ${CMAKE_CURRENT_SOURCE_DIR}/src
    # )
    # target_link_libraries(experiment_hdf5_mixed PRIVATE
    #     gpucompress
    #     H5Zgpucompress
    #     ${HDF5_C_LIBRARIES}
    #     m
    # )

    # DESIGN-6 regression: chunk tracker grows past 256 without overflow
    add_executable(test_design6_chunk_tracker tests/test_design6_chunk_tracker.c)
    target_include_directories(test_design6_chunk_tracker PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(test_design6_chunk_tracker PRIVATE
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

    # NN Adaptation Benchmark (source not yet created)
    # add_executable(bench_adaptation tests/bench_adaptation.c)
    # target_include_directories(bench_adaptation PRIVATE
    #     ${HDF5_INCLUDE_DIRS}
    #     ${CMAKE_CURRENT_SOURCE_DIR}/include
    #     ${CMAKE_CURRENT_SOURCE_DIR}/src
    # )
    # target_link_libraries(bench_adaptation PRIVATE
    #     gpucompress
    #     H5Zgpucompress
    #     ${HDF5_C_LIBRARIES}
    #     m
    # )

    # LR x MAPE sweep benchmark
    add_executable(bench_lr_sweep tests/bench_lr_sweep.c)
    target_include_directories(bench_lr_sweep PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(bench_lr_sweep PRIVATE
        gpucompress
        H5Zgpucompress
        ${HDF5_C_LIBRARIES}
        m
    )

    # Mixed-algorithm lossless verification test (source not yet created)
    # add_executable(test_mixed_algo_verify tests/test_mixed_algo_verify.c)
    # target_include_directories(test_mixed_algo_verify PRIVATE
    #     ${HDF5_INCLUDE_DIRS}
    #     ${CMAKE_CURRENT_SOURCE_DIR}/include
    #     ${CMAKE_CURRENT_SOURCE_DIR}/src
    # )
    # target_link_libraries(test_mixed_algo_verify PRIVATE
    #     gpucompress
    #     H5Zgpucompress
    #     ${HDF5_C_LIBRARIES}
    #     m
    # )

    # ============================================================
    # HDF5 VOL Connector: libH5VLgpucompress.so
    # Requires HDF5 >= 1.12 for public VOL API.
    # Use HDF5 2.2.0 built at /tmp/hdf5-install if available.
    # ============================================================
    set(HDF5_VOL_INCLUDE /tmp/hdf5-install/include)
    set(HDF5_VOL_LIB     /tmp/hdf5-install/lib/libhdf5.so)

    if(EXISTS "${HDF5_VOL_LIB}")
        add_library(H5VLgpucompress SHARED
            src/hdf5/H5VLgpucompress.cu
        )

        set_source_files_properties(src/hdf5/H5VLgpucompress.cu PROPERTIES LANGUAGE CUDA)

        target_include_directories(H5VLgpucompress
            PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/hdf5
        )

        target_link_libraries(H5VLgpucompress
            PRIVATE
            gpucompress
            ${HDF5_VOL_LIB}
            CUDA::cudart
        )

        set_target_properties(H5VLgpucompress PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            VERSION 1.0.0
            SOVERSION 1
            OUTPUT_NAME H5VLgpucompress
        )

        # VOL GPU write/read test
        add_executable(test_vol_gpu_write tests/test_vol_gpu_write.cu)

        set_source_files_properties(tests/test_vol_gpu_write.cu PROPERTIES LANGUAGE CUDA)

        target_include_directories(test_vol_gpu_write PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )

        target_link_libraries(test_vol_gpu_write PRIVATE
            gpucompress
            H5VLgpucompress
            ${HDF5_VOL_LIB}
            CUDA::cudart
            m
        )

        set_target_properties(test_vol_gpu_write PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )

        # VOL-2 regression test: GPU ptr on non-gpucompress dataset → D↔H fallback
        add_executable(test_vol2_gpu_fallback tests/test_vol2_gpu_fallback.cu)
        set_source_files_properties(tests/test_vol2_gpu_fallback.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_vol2_gpu_fallback PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_vol2_gpu_fallback PRIVATE
            gpucompress
            H5VLgpucompress
            ${HDF5_VOL_LIB}
            CUDA::cudart
            m
        )
        set_target_properties(test_vol2_gpu_fallback PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )

        # Comprehensive VOL test suite
        add_executable(test_vol_comprehensive tests/test_vol_comprehensive.cu)
        set_source_files_properties(tests/test_vol_comprehensive.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_vol_comprehensive PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_vol_comprehensive PRIVATE
            gpucompress
            H5VLgpucompress
            ${HDF5_VOL_LIB}
            CUDA::cudart
            m
        )
        set_target_properties(test_vol_comprehensive PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )

        # Minimal lossless round-trip sample
        add_executable(sample_gpu_compress_roundtrip examples/sample_gpu_compress_roundtrip.cu)
        set_source_files_properties(examples/sample_gpu_compress_roundtrip.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(sample_gpu_compress_roundtrip PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(sample_gpu_compress_roundtrip PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(sample_gpu_compress_roundtrip PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # GPU pipeline demo: full write→inspect→read→verify walkthrough
        add_executable(demo_gpu_pipeline examples/demo_gpu_pipeline.cu)
        set_source_files_properties(examples/demo_gpu_pipeline.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(demo_gpu_pipeline PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(demo_gpu_pipeline PRIVATE
            gpucompress H5VLgpucompress H5Zgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(demo_gpu_pipeline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # VOL 8MB round-trip test (mirrors test_h5z_8mb but uses GPU device pointers)
        add_executable(test_vol_8mb tests/test_vol_8mb.cu)
        set_source_files_properties(tests/test_vol_8mb.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_vol_8mb PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_vol_8mb PRIVATE
            gpucompress H5VLgpucompress H5Zgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(test_vol_8mb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # GPU-resident data benchmark: No Compression vs LZ4, no NN
        add_executable(benchmark_lz4_vs_nocomp tests/benchmark_lz4_vs_nocomp.cu)
        set_source_files_properties(tests/benchmark_lz4_vs_nocomp.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(benchmark_lz4_vs_nocomp PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(benchmark_lz4_vs_nocomp PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(benchmark_lz4_vs_nocomp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # Qualitative analysis: LZ4 + quantization values and error inspection
        add_executable(analyze_quantization_vol tests/analyze_quantization_vol.cu)
        set_source_files_properties(tests/analyze_quantization_vol.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(analyze_quantization_vol PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(analyze_quantization_vol PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(analyze_quantization_vol PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # BUG-3 audit: nnSGDKernel gradient partition (no race confirmed)
        add_executable(test_bug3_sgd_gradients tests/test_bug3_sgd_gradients.cu)
        set_source_files_properties(tests/test_bug3_sgd_gradients.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_bug3_sgd_gradients PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_bug3_sgd_gradients PRIVATE
            gpucompress CUDA::cudart pthread
        )
        set_target_properties(test_bug3_sgd_gradients PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # BUG-4 regression: %u format string for uint64_t original_size
        add_executable(test_bug4_format_string tests/test_bug4_format_string.cu)
        set_source_files_properties(tests/test_bug4_format_string.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_bug4_format_string PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_bug4_format_string PRIVATE
            gpucompress CUDA::cudart
        )
        set_target_properties(test_bug4_format_string PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # BUG-5 regression: truncated .nnwt file detection via gcount() checks
        add_executable(test_bug5_truncated_nnwt tests/test_bug5_truncated_nnwt.cu)
        set_source_files_properties(tests/test_bug5_truncated_nnwt.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_bug5_truncated_nnwt PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_bug5_truncated_nnwt PRIVATE
            gpucompress CUDA::cudart
        )
        set_target_properties(test_bug5_truncated_nnwt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # BUG-8 regression: g_sgd_ever_fired atomic visibility under concurrent SGD+inference
        add_executable(test_bug8_sgd_concurrent tests/test_bug8_sgd_concurrent.cu)
        set_source_files_properties(tests/test_bug8_sgd_concurrent.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_bug8_sgd_concurrent PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_bug8_sgd_concurrent PRIVATE
            gpucompress CUDA::cudart pthread
        )
        set_target_properties(test_bug8_sgd_concurrent PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # SGD weight update verification: asserts weights change and prediction converges
        add_executable(test_sgd_weight_update tests/test_sgd_weight_update.cu)
        set_source_files_properties(tests/test_sgd_weight_update.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_sgd_weight_update PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_sgd_weight_update PRIVATE
            gpucompress CUDA::cudart
        )
        set_target_properties(test_sgd_weight_update PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # BUG-7 regression: concurrent quantization range corruption
        add_executable(test_bug7_concurrent_quantize tests/test_bug7_concurrent_quantize.cu)
        set_source_files_properties(tests/test_bug7_concurrent_quantize.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_bug7_concurrent_quantize PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_bug7_concurrent_quantize PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(test_bug7_concurrent_quantize PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # QW-4 regression: s_gpu_writes/s_chunks_comp atomic counter integrity
        add_executable(test_qw4_atomic_counters tests/test_qw4_atomic_counters.cu)
        set_source_files_properties(tests/test_qw4_atomic_counters.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_qw4_atomic_counters PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_qw4_atomic_counters PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m pthread
        )
        set_target_properties(test_qw4_atomic_counters PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # PERF-4: batched D→H copy in runNNInference() correctness
        add_executable(test_perf4_batched_dh tests/test_perf4_batched_dh.cu)
        set_source_files_properties(tests/test_perf4_batched_dh.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_perf4_batched_dh PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_perf4_batched_dh PRIVATE
            gpucompress CUDA::cudart m
        )
        set_target_properties(test_perf4_batched_dh PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # PERF-14: native atomicAdd(double*) correctness vs CPU reference stats
        add_executable(test_perf14_atomic_double tests/test_perf14_atomic_double.cu)
        set_source_files_properties(tests/test_perf14_atomic_double.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_perf14_atomic_double PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_perf14_atomic_double PRIVATE
            gpucompress CUDA::cudart m
        )
        set_target_properties(test_perf14_atomic_double PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # PERF-2 speedup: bitonic sort vs insertion sort
        add_executable(test_perf2_sort_speedup tests/test_perf2_sort_speedup.cu)
        set_source_files_properties(tests/test_perf2_sort_speedup.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_perf2_sort_speedup PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_perf2_sort_speedup PRIVATE CUDA::cudart)
        set_target_properties(test_perf2_sort_speedup PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # PERF-16 regression: gather/scatter kernel on dedicated non-default stream
        add_executable(test_perf16_gather_stream tests/test_perf16_gather_stream.cu)
        set_source_files_properties(tests/test_perf16_gather_stream.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_perf16_gather_stream PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_perf16_gather_stream PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m pthread
        )
        set_target_properties(test_perf16_gather_stream PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # SGD parallel chunk demo: 128MB dataset, 16MB chunks, LR=0.4, MAPE=20%
        add_executable(demo_sgd_parallel_chunks tests/demo_sgd_parallel_chunks.cu)
        set_source_files_properties(tests/demo_sgd_parallel_chunks.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(demo_sgd_parallel_chunks PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(demo_sgd_parallel_chunks PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(demo_sgd_parallel_chunks PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # File-size verification: no-comp vs LZ4, keeps output files for inspection
        add_executable(check_filesize_vol tests/check_filesize_vol.cu)
        set_source_files_properties(tests/check_filesize_vol.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(check_filesize_vol PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(check_filesize_vol PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(check_filesize_vol PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # Correctness test: all 8 algos × 2 shuffle modes × 2 patterns, GPU round-trip
        add_executable(test_correctness_vol tests/test_correctness_vol.cu)
        set_source_files_properties(tests/test_correctness_vol.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(test_correctness_vol PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(test_correctness_vol PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(test_correctness_vol PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # GPU-resident data benchmark: No Compression vs NN, single ramp pattern
        add_executable(benchmark_gpu_resident tests/benchmark_gpu_resident.cu)
        set_source_files_properties(tests/benchmark_gpu_resident.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(benchmark_gpu_resident PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(benchmark_gpu_resident PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(benchmark_gpu_resident PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # NN VOL demo: 64 MB random data, ALGO_AUTO lossless, no online learning
        add_executable(nn_vol_demo examples/nn_vol_demo.cu)
        set_source_files_properties(examples/nn_vol_demo.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(nn_vol_demo PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(nn_vol_demo PRIVATE
            gpucompress H5VLgpucompress H5Zgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(nn_vol_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # Quick algorithm × pattern sweep (no HDF5)
        add_executable(quick_algo_sweep tests/quick_algo_sweep.cu)
        set_source_files_properties(tests/quick_algo_sweep.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(quick_algo_sweep PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(quick_algo_sweep PRIVATE gpucompress CUDA::cudart)
        set_target_properties(quick_algo_sweep PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # GPU-native VOL benchmark: no-comp vs static vs NN+SGD, GPU-generated data
        add_executable(benchmark_vol_gpu tests/benchmark_vol_gpu.cu)
        set_source_files_properties(tests/benchmark_vol_gpu.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(benchmark_vol_gpu PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(benchmark_vol_gpu PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(benchmark_vol_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # Algorithm sweep: all 8 algos × 2 preprocessing on 8 fill patterns
        add_executable(benchmark_algo_sweep tests/benchmark_algo_sweep.cu)
        set_source_files_properties(tests/benchmark_algo_sweep.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(benchmark_algo_sweep PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(benchmark_algo_sweep PRIVATE
            gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(benchmark_algo_sweep PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        # Chunking/header verification example
        add_executable(hdf5_chunk_verify examples/hdf5_chunk_verify.cu)
        set_source_files_properties(examples/hdf5_chunk_verify.cu PROPERTIES LANGUAGE CUDA)
        target_include_directories(hdf5_chunk_verify PRIVATE
            ${HDF5_VOL_INCLUDE}
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(hdf5_chunk_verify PRIVATE
            gpucompress H5VLgpucompress H5Zgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
        )
        set_target_properties(hdf5_chunk_verify PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        message(STATUS "HDF5 VOL connector: using ${HDF5_VOL_LIB}")
    else()
        message(STATUS "HDF5 VOL connector: skipped (HDF5 2.x not found at ${HDF5_VOL_LIB})")
        message(STATUS "  Build HDF5 2.x: cmake -S /home/cc/hdf5 -B /tmp/hdf5-build ...")
    endif()

    # HDF5 configuration validation test (all algos × preproc × patterns)
    add_executable(test_hdf5_configs tests/test_hdf5_configs.c)
    target_include_directories(test_hdf5_configs PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(test_hdf5_configs PRIVATE
        gpucompress
        H5Zgpucompress
        ${HDF5_C_LIBRARIES}
        m
    )

    add_executable(test_f9_transfers tests/test_f9_transfers.c)
    target_include_directories(test_f9_transfers PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(test_f9_transfers PRIVATE
        gpucompress
        H5Zgpucompress
        ${HDF5_C_LIBRARIES}
        m
    )

    # H5Z filter round-trip: GPU-generated 8 MB dataset, 4 MB chunks
    add_executable(test_h5z_8mb tests/test_h5z_8mb.cu)
    set_source_files_properties(tests/test_h5z_8mb.cu PROPERTIES LANGUAGE CUDA)
    target_include_directories(test_h5z_8mb PRIVATE
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(test_h5z_8mb PRIVATE
        gpucompress
        H5Zgpucompress
        ${HDF5_C_LIBRARIES}
        CUDA::cudart
        m
    )
    set_target_properties(test_h5z_8mb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
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

# Install HDF5 plugin and VOL connector to standard plugin directory
if(HDF5_FOUND)
    install(TARGETS H5Zgpucompress
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/hdf5/plugin
    )
    install(TARGETS H5VLgpucompress
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
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
    message(STATUS "  - libH5Zgpucompress.so  (HDF5 filter plugin)")
    message(STATUS "  - libH5VLgpucompress.so (HDF5 VOL connector)")
    message(STATUS "  - test_vol_gpu_write    (VOL GPU write/read test)")
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
