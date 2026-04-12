# ============================================================
# Installation
# ============================================================
include(GNUInstallDirs)

# Install shared library
install(TARGETS gpucompress
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Install executables
install(TARGETS gpu_compress gpu_decompress
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install HDF5 plugin and VOL connector to standard plugin directory
if(HDF5_FOUND)
    install(TARGETS H5Zgpucompress
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/hdf5/plugin
    )
    # Also install to standard lib dir so -lH5Zgpucompress works in simulation builds
    install(TARGETS H5Zgpucompress
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    if(TARGET H5VLgpucompress)
        install(TARGETS H5VLgpucompress
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()
endif()

# Install bridge headers (examples/) so simulations don't need -I${GPUC_DIR}/examples
install(DIRECTORY examples/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install benchmarks utility header (needed by nyx_amrex_bridge.hpp and WarpX FlushFormat)
install(FILES benchmarks/kendall_tau_profiler.cuh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
