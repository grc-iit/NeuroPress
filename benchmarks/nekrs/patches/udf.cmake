# udf.cmake — link GPUCompress libraries into the nekRS UDF
#
# After `./run.sh build` (which does cmake --install + ldconfig),
# everything is in /usr/local and no explicit paths are needed.
#
# Override with env vars if using a non-standard install:
#   export GPUCOMPRESS_DIR=/custom/prefix
#   export HDF5_DIR=/opt/hdf5

set(GPUCOMPRESS_DIR "$ENV{GPUCOMPRESS_DIR}")
if(GPUCOMPRESS_DIR STREQUAL "")
    set(GPUCOMPRESS_DIR "/usr/local")
endif()

set(HDF5_DIR "$ENV{HDF5_DIR}")
if(HDF5_DIR STREQUAL "")
    set(HDF5_DIR "/opt/hdf5")
endif()

# Include paths (only HDF5 needs an explicit hint; /usr/local/include is already searched)
target_include_directories(udf PRIVATE
    ${HDF5_DIR}/include
)

# Link GPUCompress core + VOL connector + bridge + HDF5
target_link_libraries(udf PRIVATE
    ${GPUCOMPRESS_DIR}/lib/libnekrs_gpucompress_udf.so
    ${GPUCOMPRESS_DIR}/lib/libgpucompress.so
    ${GPUCOMPRESS_DIR}/lib/libH5VLgpucompress.so
    ${GPUCOMPRESS_DIR}/lib/libH5Zgpucompress.so
    ${HDF5_DIR}/lib/libhdf5.so
)

# CUDA runtime
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(udf PRIVATE CUDA::cudart)
else()
    target_link_libraries(udf PRIVATE -L/usr/local/cuda/lib64 -lcudart)
endif()
