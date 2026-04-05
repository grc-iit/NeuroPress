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
    src/api/gpucompress_pool.cpp
    src/api/gpucompress_learning.cpp
    src/api/gpucompress_compress.cpp
    src/stats/entropy_kernel.cu
    src/nn/nn_gpu.cu
    src/api/gpucompress_diagnostics.cpp
    src/stats/stats_kernel.cu
    src/selection/heuristic.cu
    src/gray-scott/gray_scott_gpu.cu
    src/gray-scott/gray_scott_sim.cu
    src/vpic/vpic_adapter.cu
    src/nyx/nyx_adapter.cu
    src/cesm/cesm_adapter.cu
)

# Set language property for .cpp files that use CUDA
set_source_files_properties(
    src/compression/compression_factory.cpp
    src/cli/compress.cpp
    src/cli/decompress.cpp
    src/api/gpucompress_api.cpp
    src/api/gpucompress_pool.cpp
    src/api/gpucompress_learning.cpp
    src/api/gpucompress_compress.cpp
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
# CLI Executables (require cuFile/GDS for direct GPU I/O)
# ============================================================
find_library(CUFILE_LIBRARY cufile HINTS /usr/local/cuda/lib64 /opt/nvidia/cuda-12.8/targets/x86_64-linux/lib)

if(CUFILE_LIBRARY)
    message(STATUS "cuFile found: ${CUFILE_LIBRARY}")

    add_executable(gpu_compress
        src/cli/compress.cpp
        ${PREPROCESSING_SOURCES}
        ${FACTORY_SOURCES}
    )
    target_link_libraries(gpu_compress PRIVATE nvcomp ${CUFILE_LIBRARY} CUDA::cudart)
    set_target_properties(gpu_compress PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_executable(gpu_decompress
        src/cli/decompress.cpp
        ${PREPROCESSING_SOURCES}
        ${FACTORY_SOURCES}
    )
    target_link_libraries(gpu_decompress PRIVATE nvcomp ${CUFILE_LIBRARY} CUDA::cudart)
    set_target_properties(gpu_decompress PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    message(STATUS "cuFile not found — skipping CLI tools (gpu_compress, gpu_decompress)")
endif()
