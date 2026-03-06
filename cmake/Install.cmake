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
    if(TARGET H5VLgpucompress)
        install(TARGETS H5VLgpucompress
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()
endif()
