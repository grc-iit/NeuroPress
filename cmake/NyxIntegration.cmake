# ============================================================
# Nyx/AMReX Integration: build Nyx with GPUCompress compression
#
# This file is NOT included by default in the main CMakeLists.txt.
# The Nyx adapter (nyx_adapter.cu) IS built into libgpucompress.so.
#
# To build Nyx with GPUCompress, add these flags to the Nyx CMake command:
#
#   cmake -S /path/to/Nyx -B build \
#     -DNyx_GPU_BACKEND=CUDA \
#     -DNyx_MPI=ON \
#     -DAMReX_HDF5=YES \
#     -DHDF5_ROOT=/tmp/hdf5-install \
#     -DCMAKE_CXX_FLAGS="-DAMREX_USE_GPUCOMPRESS -I/path/to/GPUCompress/include -I/path/to/GPUCompress/examples -I/tmp/hdf5-install/include" \
#     -DCMAKE_EXE_LINKER_FLAGS="-L/path/to/GPUCompress/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L/tmp/hdf5-install/lib -lhdf5 -L/tmp/lib -lnvcomp"
#
# At runtime:
#   export LD_LIBRARY_PATH=/path/to/GPUCompress/build:/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
#
# In Nyx input file, set:
#   nyx.write_hdf5 = 1
#   nyx.use_gpucompress = 1
# ============================================================

find_library(GPUCOMPRESS_LIBRARY NAMES gpucompress
    HINTS ${GPUCOMPRESS_DIR} ${GPUCOMPRESS_DIR}/lib
    ENV GPUCOMPRESS_DIR
)

find_path(GPUCOMPRESS_INCLUDE_DIR NAMES gpucompress.h
    HINTS ${GPUCOMPRESS_DIR}/../include
          ${GPUCOMPRESS_DIR}/include
    ENV GPUCOMPRESS_DIR
    PATH_SUFFIXES include
)

if(GPUCOMPRESS_LIBRARY AND GPUCOMPRESS_INCLUDE_DIR)
    message(STATUS "GPUCompress found: ${GPUCOMPRESS_LIBRARY}")
    message(STATUS "GPUCompress includes: ${GPUCOMPRESS_INCLUDE_DIR}")
else()
    message(WARNING "GPUCompress not found. Set GPUCOMPRESS_DIR to the build directory.")
endif()
