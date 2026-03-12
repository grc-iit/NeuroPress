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
add_library(H5VLgpucompress SHARED src/hdf5/H5VLgpucompress.cu src/xfer_tracker.cpp)
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
add_vol_test(test_vol_gpu_write       tests/hdf5/test_vol_gpu_write.cu)
add_vol_test(test_vol2_gpu_fallback   tests/hdf5/test_vol2_gpu_fallback.cu)
add_vol_test(test_vol_comprehensive   tests/hdf5/test_vol_comprehensive.cu)
add_vol_test(test_vol_8mb             tests/hdf5/test_vol_8mb.cu)
target_link_libraries(test_vol_8mb PRIVATE H5Zgpucompress)
add_vol_test(test_correctness_vol     tests/hdf5/test_correctness_vol.cu)
add_vol_test(test_vol_xfer_audit     tests/hdf5/test_vol_xfer_audit.cu)
add_vol_test(test_vol_pipeline_comprehensive  tests/hdf5/test_vol_pipeline_comprehensive.cu)
add_vol_test(test_vol_verify_gpu_path        tests/hdf5/test_vol_verify_gpu_path.cu)
add_vol_test(test_nn_algo_convergence        tests/test_nn_algo_convergence.cu)
add_vol_test(test_nn_bitcomp                 tests/test_nn_bitcomp.cu)
add_vol_test(test_nn_predict_vs_actual       tests/test_nn_predict_vs_actual.cu)

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
add_vol_test(test_h7_null_calloc            tests/regression/test_h7_null_calloc.cu)

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
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/grayscott/grayscott-benchmark.cu")
    add_vol_demo(grayscott_benchmark       benchmarks/grayscott/grayscott-benchmark.cu)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/synthetic/synthetic-nn-benchmark.cu")
    add_vol_demo(synthetic_nn_benchmark    benchmarks/synthetic/synthetic-nn-benchmark.cu)
endif()
