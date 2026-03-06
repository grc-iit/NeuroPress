# ============================================================
# VPIC-Kokkos Integration: build VPIC decks linked against GPUCompress
#
# This file is NOT included by default in the main CMakeLists.txt.
# It is a standalone CMake module meant to be used from the VPIC-Kokkos
# build system or a separate project that links both VPIC and GPUCompress.
#
# Usage from a standalone CMakeLists.txt:
#   set(GPUCOMPRESS_DIR "/path/to/GPUCompress/build")
#   set(VPIC_DIR        "/path/to/vpic-kokkos")
#   include(cmake/VpicIntegration.cmake)
#   build_vpic_with_gpucompress(my_deck /path/to/deck.cxx)
#
# Or just use the flags directly when building vpic-kokkos:
#   cmake -S /path/to/vpic-kokkos -B build-vpic \
#     -DENABLE_KOKKOS_CUDA=ON \
#     -DUSER_DECKS="/path/to/GPUCompress/examples/vpic_compress_deck.cxx" \
#     -DCMAKE_CXX_FLAGS="-I/path/to/GPUCompress/include" \
#     -DCMAKE_EXE_LINKER_FLAGS="-L/path/to/GPUCompress/build -lgpucompress -L/tmp/lib -lnvcomp"
# ============================================================

# Find GPUCompress installed library
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
    return()
endif()

# Macro: build a VPIC deck with GPUCompress linked in
# Assumes VPIC's build_a_vpic macro is available, or provides a standalone target
macro(build_vpic_with_gpucompress target_name deck_path)
    if(COMMAND build_a_vpic)
        # We are inside the VPIC-Kokkos build system
        build_a_vpic(${target_name} ${deck_path})
        target_include_directories(${target_name} PRIVATE ${GPUCOMPRESS_INCLUDE_DIR})
        target_link_libraries(${target_name} ${GPUCOMPRESS_LIBRARY})
    else()
        message(FATAL_ERROR
            "build_vpic_with_gpucompress requires the VPIC build system.\n"
            "Include this file from within the VPIC-Kokkos CMakeLists.txt or\n"
            "use the manual link flags approach described in the header comment.")
    endif()
endmacro()
