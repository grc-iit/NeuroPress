# ============================================================
# HDF5 VOL Connector: libH5VLgpucompress.so
# Requires HDF5 >= 1.12 for public VOL API.
# Use HDF5 2.2.0 built at /tmp/hdf5-install if available.
# ============================================================
if(NOT HDF5_FOUND)
    return()
endif()

set(HDF5_VOL_INCLUDE /tmp/hdf5-install/include)
set(HDF5_VOL_LIB     /tmp/hdf5-install/lib/libhdf5.so)

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
add_vol_test(test_vol_gpu_write       tests/hdf5/test_vol_gpu_write.cu)
add_vol_test(test_vol2_gpu_fallback   tests/hdf5/test_vol2_gpu_fallback.cu)
add_vol_test(test_vol_comprehensive   tests/hdf5/test_vol_comprehensive.cu)
add_vol_test(test_vol_8mb             tests/hdf5/test_vol_8mb.cu)
target_link_libraries(test_vol_8mb PRIVATE H5Zgpucompress)
add_vol_test(test_correctness_vol     tests/hdf5/test_correctness_vol.cu)

# ============================================================
# VOL Regression Tests
# ============================================================
add_vol_test(test_bug7_concurrent_quantize  tests/regression/test_bug7_concurrent_quantize.cu)
add_vol_test(test_qw4_atomic_counters       tests/regression/test_qw4_atomic_counters.cu)
target_link_libraries(test_qw4_atomic_counters PRIVATE pthread)
add_vol_test(test_perf16_gather_stream      tests/perf/test_perf16_gather_stream.cu)
target_link_libraries(test_perf16_gather_stream PRIVATE pthread)

# ============================================================
# Examples / Demos
# ============================================================
add_vol_test(sample_gpu_compress_roundtrip  examples/sample_gpu_compress_roundtrip.cu)
add_vol_demo(demo_gpu_pipeline              examples/demo_gpu_pipeline.cu)
add_vol_demo(nn_vol_demo                    examples/nn_vol_demo.cu)
add_vol_demo(grayscott_vol_demo             examples/grayscott_vol_demo.cu)
add_vol_demo(vpic_vol_demo                  examples/vpic_vol_demo.cu)
# ============================================================
# VOL Benchmarks
# ============================================================
add_vol_demo(benchmark_grayscott_vol       tests/benchmarks/benchmark_grayscott_vol.cu)
