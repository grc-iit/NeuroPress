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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Target the actual GPU architecture (sm_80 = Ampere A100).
# Override at configure time: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to target")
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Optional: Find HDF5 for filter plugin
find_package(HDF5 COMPONENTS C)

# Optional: Find MPI for multi-GPU benchmark support
find_package(MPI)

# External dependency prefix (nvcomp, etc.)
set(NVCOMP_PREFIX "/tmp" CACHE PATH "nvcomp install prefix (include/ and lib/)")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NVCOMP_PREFIX}/include
    /usr/local/cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link directories
# - nvcomp 5.1.0 libraries
# - cuFile path: may vary by system; override with -DCMAKE_LIBRARY_PATH=<path>
link_directories(
    ${NVCOMP_PREFIX}/lib
    /opt/nvidia/cuda-12.8/targets/x86_64-linux/lib
)

# ============================================================
# Modular build configuration
# ============================================================
include(cmake/CoreLibrary.cmake)   # libgpucompress.so, CLIs, benchmark, eval
include(cmake/Tests.cmake)         # Unit, NN, regression, perf tests
include(cmake/HDF5Filter.cmake)    # libH5Zgpucompress.so + H5Z tests
include(cmake/HDF5Vol.cmake)       # libH5VLgpucompress.so + VOL tests/demos/benchmarks
include(cmake/Install.cmake)       # Installation rules

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
if(HDF5_FOUND)
    message(STATUS "  - libH5Zgpucompress.so  (HDF5 filter plugin)")
    if(TARGET H5VLgpucompress)
        message(STATUS "  - libH5VLgpucompress.so (HDF5 VOL connector)")
    endif()
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
