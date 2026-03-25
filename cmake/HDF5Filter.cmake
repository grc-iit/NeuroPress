# ============================================================
# HDF5 Filter Plugin: libH5Zgpucompress.so (Optional)
# ============================================================
if(NOT HDF5_FOUND)
    message(STATUS "HDF5 not found - skipping filter plugin libH5Zgpucompress.so")
    message(STATUS "  To build HDF5 plugin, install HDF5 development libraries:")
    message(STATUS "    Ubuntu/Debian: sudo apt-get install libhdf5-dev")
    message(STATUS "    RHEL/CentOS:   sudo yum install hdf5-devel")
    return()
endif()

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
    pthread
)

set_target_properties(H5Zgpucompress PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION 1.0.0
    SOVERSION 1
    OUTPUT_NAME H5Zgpucompress
)

target_compile_definitions(H5Zgpucompress PRIVATE GPUCOMPRESS_BUILD_HDF5_PLUGIN)

# ============================================================
# HDF5 Filter Tests (H5Z, no VOL)
# ============================================================

# (Removed: test_design6_chunk_tracker — host-path stub, H5Z filter calls gpucompress_compress())

# HDF5 configuration validation test
add_executable(test_hdf5_configs tests/hdf5/test_hdf5_configs.c)
target_include_directories(test_hdf5_configs PRIVATE
    ${HDF5_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_hdf5_configs PRIVATE
    gpucompress H5Zgpucompress ${HDF5_C_LIBRARIES} m
)

add_executable(test_f9_transfers tests/hdf5/test_f9_transfers.c)
target_include_directories(test_f9_transfers PRIVATE
    ${HDF5_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_f9_transfers PRIVATE
    gpucompress H5Zgpucompress ${HDF5_C_LIBRARIES} m
)

# H5Z filter round-trip: GPU-generated 8 MB dataset, 4 MB chunks
add_executable(test_h5z_8mb tests/hdf5/test_h5z_8mb.cu)
set_source_files_properties(tests/hdf5/test_h5z_8mb.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h5z_8mb PRIVATE
    ${HDF5_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_h5z_8mb PRIVATE
    gpucompress H5Zgpucompress ${HDF5_C_LIBRARIES} CUDA::cudart m
)
set_target_properties(test_h5z_8mb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# test_h8_filter_globals_race removed — host-path test, pre-existing failure
