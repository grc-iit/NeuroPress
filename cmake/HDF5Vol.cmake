# ============================================================
# HDF5 VOL Connector: libH5VLgpucompress.so
# Requires HDF5 >= 1.12 for public VOL API.
# Use HDF5 2.2.0 built at /tmp/hdf5-install if available.
# ============================================================
if(NOT HDF5_FOUND)
    return()
endif()

set(HDF5_VOL_PREFIX "/tmp/hdf5-install" CACHE PATH "HDF5 2.x install prefix for VOL connector")
set(HDF5_VOL_INCLUDE "${HDF5_VOL_PREFIX}/include" CACHE PATH "HDF5 VOL include directory")
set(HDF5_VOL_LIB     "${HDF5_VOL_PREFIX}/lib/libhdf5.so" CACHE PATH "HDF5 VOL library path")

if(NOT EXISTS "${HDF5_VOL_LIB}")
    message(STATUS "HDF5 VOL connector: skipped (HDF5 2.x not found at ${HDF5_VOL_LIB})")
    message(STATUS "  Build HDF5 2.x: cmake -S /home/cc/hdf5 -B /tmp/hdf5-build ...")
    return()
endif()

message(STATUS "HDF5 VOL connector: using ${HDF5_VOL_LIB}")

# ---- Helper macros for repetitive target patterns ----
macro(add_vol_test TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${HDF5_VOL_INCLUDE}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(${TARGET_NAME} PRIVATE
        gpucompress H5VLgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
    )
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endmacro()

macro(add_vol_demo TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${HDF5_VOL_INCLUDE}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(${TARGET_NAME} PRIVATE
        gpucompress H5VLgpucompress H5Zgpucompress ${HDF5_VOL_LIB} CUDA::cudart m
    )
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endmacro()

# ---- VOL Library ----
add_library(H5VLgpucompress SHARED src/hdf5/H5VLgpucompress.cu)
set_source_files_properties(src/hdf5/H5VLgpucompress.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(H5VLgpucompress PRIVATE
    ${HDF5_VOL_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hdf5
)
target_link_libraries(H5VLgpucompress PRIVATE
    gpucompress ${HDF5_VOL_LIB} CUDA::cudart
)
set_target_properties(H5VLgpucompress PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
    OUTPUT_NAME H5VLgpucompress
)

# ============================================================
# VOL Tests
# ============================================================
# (Removed: test_execution_flow — stale input, needs CLI arg)
# (Removed: test_vol_gpu_write — HOST_PTR_ABORT)
# (Removed: test_vol2_gpu_fallback — HOST_PTR_ABORT)
# (Removed: test_vol_comprehensive — HOST_PTR_ABORT)
add_vol_test(test_vol_8mb             tests/hdf5/test_vol_8mb.cu)
target_link_libraries(test_vol_8mb PRIVATE H5Zgpucompress)
add_vol_test(test_correctness_vol     tests/hdf5/test_correctness_vol.cu)
add_vol_test(test_vol_xfer_audit     tests/hdf5/test_vol_xfer_audit.cu)
# (Removed: test_vol_pipeline_comprehensive — API_CHANGE)
add_vol_test(test_vol_verify_gpu_path        tests/hdf5/test_vol_verify_gpu_path.cu)
add_vol_test(test_vol_2d_chunk_roundtrip    tests/hdf5/test_vol_2d_chunk_roundtrip.cu)
add_vol_test(test_nn_algo_convergence        tests/test_nn_algo_convergence.cu)
add_vol_test(test_nn_bitcomp                 tests/test_nn_bitcomp.cu)
add_vol_test(test_nn_predict_vs_actual       tests/test_nn_predict_vs_actual.cu)
add_vol_test(test_vol_nn_predictions         tests/hdf5/test_vol_nn_predictions.cu)
add_vol_test(test_vol_lossless_stress        tests/hdf5/test_vol_lossless_stress.cu)
add_vol_test(test_vol_algo_shuffle_verify   tests/hdf5/test_vol_algo_shuffle_verify.cu)
add_vol_test(test_nn_vol_correctness         tests/hdf5/test_nn_vol_correctness.cu)
add_vol_test(test_hdf5_compat                tests/hdf5/test_hdf5_compat.cu)
target_link_libraries(test_hdf5_compat PRIVATE H5Zgpucompress)
add_vol_test(test_lru1_manager_cache         tests/hdf5/test_lru1_manager_cache.cu)
add_vol_test(test_vol_exploration            tests/hdf5/test_vol_exploration.cu)

# ============================================================
# VOL Regression Tests
# ============================================================
add_vol_test(test_bug7_concurrent_quantize  tests/regression/test_bug7_concurrent_quantize.cu)
add_vol_test(test_qw4_atomic_counters       tests/regression/test_qw4_atomic_counters.cu)
target_link_libraries(test_qw4_atomic_counters PRIVATE pthread)
add_vol_test(test_perf16_gather_stream      tests/perf/test_perf16_gather_stream.cu)
target_link_libraries(test_perf16_gather_stream PRIVATE pthread)
add_vol_test(test_h6_transfer_counter_race  tests/regression/test_h6_transfer_counter_race.cu)
target_link_libraries(test_h6_transfer_counter_race PRIVATE pthread)
# (Removed: test_h7_null_calloc — HOST_PTR_ABORT)
add_vol_test(test_vol_c4c8h7_defensive     tests/hdf5/test_vol_c4c8h7_defensive.cu)
target_link_libraries(test_vol_c4c8h7_defensive PRIVATE H5Zgpucompress)
add_vol_test(test_h1_vol_read_stream_sync  tests/hdf5/test_h1_vol_read_stream_sync.cu)
target_link_libraries(test_h1_vol_read_stream_sync PRIVATE H5Zgpucompress)
add_vol_test(test_vol_host_ptr_reject     tests/hdf5/test_vol_host_ptr_reject.cu)
target_link_libraries(test_vol_host_ptr_reject PRIVATE H5Zgpucompress)
add_vol_test(test_m4_write_buffer_reuse  tests/hdf5/test_m4_write_buffer_reuse.cu)
target_link_libraries(test_m4_write_buffer_reuse PRIVATE H5Zgpucompress)
add_vol_test(test_s6_parallel_exploration  tests/hdf5/test_s6_parallel_exploration.cu)
target_link_libraries(test_s6_parallel_exploration PRIVATE H5Zgpucompress)
# (Removed: test_n3_exploration_alloc — API_CHANGE)
add_vol_test(test_vol_modes                tests/hdf5/test_vol_modes.cu)

# VOL bypass mode + program wall / compute_ms timing tests
add_vol_test(test_vol_bypass_roundtrip  tests/hdf5/test_vol_bypass_roundtrip.cu)
target_link_libraries(test_vol_bypass_roundtrip PRIVATE H5Zgpucompress)
add_vol_test(test_vol_bypass_timing     tests/hdf5/test_vol_bypass_timing.cu)
target_link_libraries(test_vol_bypass_timing PRIVATE H5Zgpucompress)
add_vol_test(test_vol_program_wall      tests/hdf5/test_vol_program_wall.cu)
target_link_libraries(test_vol_program_wall PRIVATE H5Zgpucompress)

# calloc fault injection interposer (plain C, no CUDA)
add_library(calloc_fault SHARED tests/regression/calloc_fault.c)
target_link_libraries(calloc_fault PRIVATE dl)
set_target_properties(calloc_fault PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""       # produce calloc_fault.so, not libcalloc_fault.so
)

# ============================================================
# Examples / Demos (only build if source files exist)
# ============================================================
foreach(_demo
    sample_gpu_compress_roundtrip:examples/sample_gpu_compress_roundtrip.cu:test
    demo_gpu_pipeline:examples/demo_gpu_pipeline.cu:demo
    nn_vol_demo:examples/nn_vol_demo.cu:demo
    grayscott_vol_demo:examples/grayscott_vol_demo.cu:demo
    vpic_vol_demo:examples/vpic_vol_demo.cu:demo
    minimal_nn_vol_profile:examples/minimal_nn_vol_profile.cu:demo
    hdf5_replay:examples/hdf5_replay.cu:demo
)
    string(REPLACE ":" ";" _parts ${_demo})
    list(GET _parts 0 _target)
    list(GET _parts 1 _source)
    list(GET _parts 2 _type)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
        if(_type STREQUAL "test")
            add_vol_test(${_target} ${_source})
        else()
            add_vol_demo(${_target} ${_source})
        endif()
    else()
        message(STATUS "  Skipping ${_target}: ${_source} not found")
    endif()
endforeach()
# ============================================================
# VOL Benchmarks
# ============================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/grayscott/grayscott-benchmark-pm.cu")
    add_vol_demo(grayscott_benchmark_pm    benchmarks/grayscott/grayscott-benchmark-pm.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/synthetic/synthetic-nn-benchmark.cu")
    add_vol_demo(synthetic_nn_benchmark    benchmarks/synthetic/synthetic-nn-benchmark.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/synthetic/synthetic-workloads-benchmark.cu")
    add_vol_demo(synthetic_workloads_benchmark benchmarks/synthetic/synthetic-workloads-benchmark.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/diagnostic/nn-diagnostic-benchmark.cu")
    add_vol_demo(nn_diagnostic_benchmark benchmarks/diagnostic/nn-diagnostic-benchmark.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/sdrbench/generic-benchmark.cu")
    add_vol_demo(generic_benchmark       benchmarks/sdrbench/generic-benchmark.cu)
    if(TARGET generic_benchmark AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/vpic-kokkos/vpic_psnr.cu")
        target_sources(generic_benchmark PRIVATE benchmarks/vpic-kokkos/vpic_psnr.cu)
        set_source_files_properties(benchmarks/vpic-kokkos/vpic_psnr.cu PROPERTIES LANGUAGE CUDA)
    endif()
endif()

# ── Optional MPI linking for multi-GPU benchmark support ──
if(MPI_FOUND)
    message(STATUS "  MPI found — enabling multi-GPU benchmark support")
    foreach(_mpi_target grayscott_benchmark_pm generic_benchmark)
        if(TARGET ${_mpi_target})
            target_link_libraries(${_mpi_target} PRIVATE MPI::MPI_CXX)
            target_compile_definitions(${_mpi_target} PRIVATE GPUCOMPRESS_USE_MPI)
        endif()
    endforeach()
endif()

# ── EMA reset integration test (requires H5VLgpucompress) ──
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/test_ema_reset.cu")
    add_vol_demo(test_ema_reset tests/unit/test_ema_reset.cu)
endif()

# ============================================================
# CTest registrations — VOL tests
# ============================================================

# VOL correctness / roundtrip
gpucompress_add_test(test_vol_8mb                   vol  180)
gpucompress_add_test(test_correctness_vol            vol  300)
gpucompress_add_test(test_vol_xfer_audit             vol  180)
gpucompress_add_test(test_vol_verify_gpu_path        vol  180)
gpucompress_add_test(test_vol_2d_chunk_roundtrip     vol  180)
gpucompress_add_test(test_vol_lossless_stress        vol  300)
gpucompress_add_test(test_vol_algo_shuffle_verify    vol  180)
gpucompress_add_test(test_hdf5_compat                vol  180)
gpucompress_add_test(test_lru1_manager_cache         vol  180)
gpucompress_add_test(test_vol_exploration            vol  180)
gpucompress_add_test(test_vol_host_ptr_reject        vol  180)
gpucompress_add_test(test_vol_c4c8h7_defensive       vol  180)

# VOL + NN
gpucompress_add_test(test_nn_algo_convergence        vol  180)
gpucompress_add_test(test_nn_bitcomp                 vol  180)
gpucompress_add_test(test_nn_predict_vs_actual       vol  180)
gpucompress_add_test(test_vol_nn_predictions         vol  180)
gpucompress_add_test(test_nn_vol_correctness         vol  180)

# VOL regression
gpucompress_add_test(test_bug7_concurrent_quantize   regression  120)
gpucompress_add_test(test_qw4_atomic_counters        regression  120)
gpucompress_add_test(test_perf16_gather_stream       perf        120)
gpucompress_add_test(test_h6_transfer_counter_race   regression  120)
gpucompress_add_test(test_h1_vol_read_stream_sync    regression  180)
gpucompress_add_test(test_m4_write_buffer_reuse      regression  180)
gpucompress_add_test(test_s6_parallel_exploration    regression  180)
gpucompress_add_test(test_vol_modes                  vol         600)

# sample roundtrip (example, but it's registered as a test target)
if(TARGET sample_gpu_compress_roundtrip)
    gpucompress_add_test(sample_gpu_compress_roundtrip vol 180)
endif()

# EMA reset integration test
if(TARGET test_ema_reset)
    gpucompress_add_test(test_ema_reset unit 120)
endif()
