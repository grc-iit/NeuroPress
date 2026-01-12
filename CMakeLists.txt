cmake_minimum_required(VERSION 3.18)

# Project configuration
project(GPUCompression LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Debug symbols
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    $ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/libnvcomp/include
    /usr/local/cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link directories
link_directories(
    $ENV{HOME}/.local/lib/python3.12/site-packages/nvidia/libnvcomp/lib64
)

# Source files
set(FACTORY_SOURCES
    src/CompressionFactory.cpp
    src/CompressionFactory.hpp
)

# Set language property for .cpp files that use CUDA
set_source_files_properties(
    src/CompressionFactory.cpp
    src/GPU_Compress.cpp
    src/GPU_Decompress.cpp
    PROPERTIES LANGUAGE CUDA
)

# ========== Executable 1: GPU Compress (with verification) ==========
add_executable(gpu_compress 
    src/GPU_Compress.cpp
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

# ========== Executable 2: GPU Decompress ==========
add_executable(gpu_decompress 
    src/GPU_Decompress.cpp
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

# Display build configuration
message(STATUS "Building GPU Compression/Decompression with modular factory pattern")
message(STATUS "Executables:")
message(STATUS "  - gpu_compress: Compression with data verification")
message(STATUS "  - gpu_decompress: Decompression (auto-detects algorithm)")
message(STATUS "Features:")
message(STATUS "  - CompressionFactory.cpp/hpp (Factory pattern)")
message(STATUS "  - nvcomp library support (LZ4, Snappy, Deflate, Gdeflate, Zstd, ANS, Cascaded, Bitcomp)")
message(STATUS "  - GPUDirect Storage (cuFile) support for direct GPU<->NVMe transfers")
message(STATUS "  - 4KB alignment for optimal GDS performance")
message(STATUS "Note: Using .cpp extension with CUDA compilation")

