cmake_minimum_required(VERSION 3.18)

# Project configuration
project(GPUCompression LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(
    /usr/include/nvcomp_13
    /usr/local/cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link directories
link_directories(
    /usr/lib/x86_64-linux-gnu/nvcomp/13
)

# Source files
set(FACTORY_SOURCES
    src/CompressionFactory.cpp
    src/CompressionFactory.hpp
)

set(MAIN_SOURCES
    src/GPU_Compress.cpp
)

# Create executable
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

# Link libraries
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
message(STATUS "  - GPU_Compress.cpp (Main application)")
message(STATUS "  - nvcomp library support")
message(STATUS "  - GPUDirect Storage (cuFile) support")
message(STATUS "Note: Using .cpp extension with CUDA compilation (no custom kernels)")

