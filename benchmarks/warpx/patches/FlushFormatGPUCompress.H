#ifndef WARPX_FLUSHFORMATGPUCOMPRESS_H_
#define WARPX_FLUSHFORMATGPUCOMPRESS_H_

#include "FlushFormat.H"

#include "Diagnostics/ParticleDiag/ParticleDiag_fwd.H"

#include <AMReX_Geometry.H>
#include <AMReX_Vector.H>

#include <AMReX_BaseFwd.H>

#include <string>

/**
 * \brief Flush format that writes field data to compressed HDF5 files
 * using the GPUCompress library. Borrows GPU-resident device pointers
 * from AMReX MultiFabs and compresses them in-situ via nvcomp, avoiding
 * device-to-host transfers for the compression step.
 *
 * Requires linking against libgpucompress.so and HDF5.
 * Enable with: <diag_name>.format = gpucompress
 */
class FlushFormatGPUCompress : public FlushFormat
{
public:
    FlushFormatGPUCompress ();
    ~FlushFormatGPUCompress () override;

    /** Flush fields and particles to compressed HDF5 */
    void WriteToFile (
        const amrex::Vector<std::string>& varnames,
        const amrex::Vector<amrex::MultiFab>& mf,
        amrex::Vector<amrex::Geometry>& geom,
        amrex::Vector<int> iteration, double time,
        const amrex::Vector<ParticleDiag>& particle_diags, int nlev,
        std::string prefix, int file_min_digits,
        bool plot_raw_fields,
        bool plot_raw_fields_guards,
        int verbose = 2,
        bool use_pinned_pc = false,
        bool isBTD = false, int snapshotID = -1,
        int bufferID = 1, int numBuffers = 1,
        const amrex::Geometry& full_BTD_snapshot = amrex::Geometry(),
        bool isLastBTDFlush = false) const override;

    FlushFormatGPUCompress ( FlushFormatGPUCompress const &)             = delete;
    FlushFormatGPUCompress& operator= ( FlushFormatGPUCompress const & ) = delete;
    FlushFormatGPUCompress ( FlushFormatGPUCompress&& )                  = default;
    FlushFormatGPUCompress& operator= ( FlushFormatGPUCompress&& )       = default;

private:
    bool m_initialized = false;
};

#endif // WARPX_FLUSHFORMATGPUCOMPRESS_H_
