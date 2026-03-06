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
    src/gray-scott/gray_scott_gpu.cu
    src/gray-scott/gray_scott_sim.cu
    src/vpic/vpic_adapter.cu
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
# CLI Executables
# ============================================================
add_executable(gpu_compress
    src/cli/compress.cpp
    ${PREPROCESSING_SOURCES}
    ${FACTORY_SOURCES}
)
target_link_libraries(gpu_compress PRIVATE nvcomp cufile CUDA::cudart)
set_target_properties(gpu_compress PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(gpu_decompress
    src/cli/decompress.cpp
    ${PREPROCESSING_SOURCES}
    ${FACTORY_SOURCES}
)
target_link_libraries(gpu_decompress PRIVATE nvcomp cufile CUDA::cudart)
set_target_properties(gpu_decompress PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================
# Benchmark executable
# ============================================================
add_executable(benchmark scripts/benchmark.cpp)
target_link_libraries(benchmark PRIVATE gpucompress pthread)

# ============================================================
# Evaluation pipeline
# ============================================================
add_executable(eval_simulation eval/eval_simulation.cpp)
target_link_libraries(eval_simulation PRIVATE gpucompress pthread)
