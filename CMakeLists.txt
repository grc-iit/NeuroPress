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

set(MAIN_SOURCES
    src/GPU_Compress.cpp
)

# Create executable with GDS support
add_executable(gpu_compress 
    ${MAIN_SOURCES}
    ${FACTORY_SOURCES}
)

# Set language property for .cpp files that use CUDA
set_source_files_properties(
    src/CompressionFactory.cpp
    src/GPU_Compress.cpp
    PROPERTIES LANGUAGE CUDA
)

# Link libraries for GDS version
target_link_libraries(gpu_compress
    PRIVATE
    nvcomp
    cufile
    CUDA::cudart
)

# Set CUDA architectures - use native detection
set_target_properties(gpu_compress PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Display build configuration
message(STATUS "Building GPU Compression with modular factory pattern")
message(STATUS "  - CompressionFactory.cpp/hpp (Factory pattern)")
message(STATUS "  - GPU_Compress.cpp (GDS version)")
message(STATUS "  - nvcomp library support")
message(STATUS "  - GPUDirect Storage (cuFile) support for GDS version")
message(STATUS "Note: Using .cpp extension with CUDA compilation (no custom kernels)")

