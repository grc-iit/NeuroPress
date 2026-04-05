# GPUCompress: GPU-accelerated compression for in-situ I/O
#
# Finds libgpucompress and HDF5 for compressed output.
# Set GPUCOMPRESS_PREFIX to point to the GPUCompress install/build directory.
#
# Sets:
#   WarpX_HAVE_GPUCOMPRESS  - TRUE if found and usable

if(WarpX_GPUCOMPRESS)
    # Find gpucompress library
    find_library(GPUCOMPRESS_LIBRARY
        NAMES gpucompress
        HINTS
            ${GPUCOMPRESS_PREFIX}/lib
            ${GPUCOMPRESS_PREFIX}
            $ENV{GPUCOMPRESS_PREFIX}/lib
            $ENV{GPUCOMPRESS_PREFIX}
            $ENV{HOME}/GPUCompress
    )

    # Find gpucompress headers
    find_path(GPUCOMPRESS_INCLUDE_DIR
        NAMES gpucompress.h
        HINTS
            ${GPUCOMPRESS_PREFIX}/include
            ${GPUCOMPRESS_PREFIX}/../include
            $ENV{GPUCOMPRESS_PREFIX}/include
            $ENV{HOME}/GPUCompress/include
    )

    # Find HDF5 (required for VOL connector path)
    find_package(HDF5 COMPONENTS C)

    if(GPUCOMPRESS_LIBRARY AND GPUCOMPRESS_INCLUDE_DIR AND HDF5_FOUND)
        message(STATUS "GPUCompress found: ${GPUCOMPRESS_LIBRARY}")
        message(STATUS "GPUCompress headers: ${GPUCOMPRESS_INCLUDE_DIR}")

        # Create imported target
        if(NOT TARGET gpucompress::gpucompress)
            add_library(gpucompress::gpucompress SHARED IMPORTED)
            set_target_properties(gpucompress::gpucompress PROPERTIES
                IMPORTED_LOCATION ${GPUCOMPRESS_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES ${GPUCOMPRESS_INCLUDE_DIR}
            )
        endif()

        set(WarpX_HAVE_GPUCOMPRESS TRUE)
    else()
        if(NOT GPUCOMPRESS_LIBRARY)
            message(WARNING "GPUCompress library not found. "
                "Set GPUCOMPRESS_PREFIX to the GPUCompress build/install directory.")
        endif()
        if(NOT GPUCOMPRESS_INCLUDE_DIR)
            message(WARNING "GPUCompress headers not found.")
        endif()
        if(NOT HDF5_FOUND)
            message(WARNING "HDF5 not found (required for GPUCompress VOL connector).")
        endif()
        message(FATAL_ERROR "GPUCompress requested but dependencies not found.")
    endif()
endif()
