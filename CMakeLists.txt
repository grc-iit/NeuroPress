cmake_minimum_required(VERSION 3.18)

# Project configuration
project(GPUCompression LANGUAGES CXX CUDA C)

enable_testing()

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

# Target the actual GPU architecture (sm_80 = Ampere A100). A100 is the
# reference platform for this artifact; override at configure time for
# H100 or Hopper via -DCMAKE_CUDA_ARCHITECTURES=90. sm_70+ is required
# because we use atomicAdd(double*, double) which is a sm_60+ intrinsic
# (we use sm_80 default; CMake's "native" or lower defaults won't work).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES LESS 60)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to target" FORCE)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES unset or too low; forcing 80 (A100)")
endif()

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Dependency auto-detection. Try two conventions in order:
#   1. ${CMAKE_SOURCE_DIR}/.deps/   (bare-metal: scripts/install_deps.sh)
#   2. /tmp/ and /tmp/hdf5-install  (Docker/Apptainer: docker/Dockerfile)
# Explicit -DHDF5_ROOT / -DNVCOMP_PREFIX still override both.
if(NOT DEFINED HDF5_ROOT)
    if(EXISTS "${CMAKE_SOURCE_DIR}/.deps/hdf5/lib/libhdf5.so")
        set(HDF5_ROOT "${CMAKE_SOURCE_DIR}/.deps/hdf5" CACHE PATH "HDF5 install prefix (auto from .deps/)")
        message(STATUS "Auto-detected HDF5_ROOT=${HDF5_ROOT} from .deps/")
    elseif(EXISTS "/tmp/hdf5-install/lib/libhdf5.so")
        set(HDF5_ROOT "/tmp/hdf5-install" CACHE PATH "HDF5 install prefix (auto from /tmp/hdf5-install)")
        message(STATUS "Auto-detected HDF5_ROOT=${HDF5_ROOT} from /tmp/ (Docker convention)")
    endif()
endif()

# Optional: Find HDF5 for filter plugin
find_package(HDF5 COMPONENTS C)

# Optional: Find MPI for multi-GPU benchmark support
find_package(MPI)

# If HDF5 was built with parallel (MPI) support, every header that
# includes H5public.h transitively #includes <mpi.h>. CMake's
# find_package(HDF5) does NOT automatically propagate MPI include dirs
# to consumers, so nvcc fails with "mpi.h: No such file or directory"
# on any test/benchmark that pulls in hdf5.h. Add them globally.
if(HDF5_IS_PARALLEL AND MPI_C_FOUND)
    include_directories(${MPI_C_INCLUDE_DIRS})
    message(STATUS "HDF5 is parallel; propagating MPI include dirs: ${MPI_C_INCLUDE_DIRS}")
endif()

# External dependency prefix (nvcomp, etc.). Mirror the HDF5 auto-detect:
# try .deps/ first, then /tmp/ (Docker convention); default is /tmp so the
# existing container build keeps working even if neither header is found.
if(EXISTS "${CMAKE_SOURCE_DIR}/.deps/include/nvcomp.hpp")
    set(NVCOMP_PREFIX "${CMAKE_SOURCE_DIR}/.deps" CACHE PATH "nvcomp install prefix (include/ and lib/)")
    message(STATUS "Auto-detected NVCOMP_PREFIX=${NVCOMP_PREFIX} from .deps/")
elseif(EXISTS "/tmp/include/nvcomp.hpp")
    set(NVCOMP_PREFIX "/tmp" CACHE PATH "nvcomp install prefix (include/ and lib/)")
    message(STATUS "Auto-detected NVCOMP_PREFIX=${NVCOMP_PREFIX} from /tmp/ (Docker convention)")
else()
    set(NVCOMP_PREFIX "/tmp" CACHE PATH "nvcomp install prefix (include/ and lib/)")
endif()

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
add_subdirectory(synthetic)        # GPU synthetic benchmark generator

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
