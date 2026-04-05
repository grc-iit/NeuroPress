/**
 * @file nyx_amrex_bridge.hpp
 * @brief Header-only bridge between AMReX MultiFab and GPUCompress Nyx adapter
 *
 * Provides convenience functions to attach AMReX MultiFab FArrayBox device
 * pointers to the GPUCompress Nyx adapter, and to write compressed HDF5
 * using the VOL connector directly from GPU-resident MultiFab data.
 *
 * Follows the same pattern as vpic_kokkos_bridge.hpp.
 */

#ifndef NYX_AMREX_BRIDGE_HPP
#define NYX_AMREX_BRIDGE_HPP

#include "gpucompress_nyx.h"
#include "gpucompress_hdf5_vol.h"
#include "gpucompress_hdf5.h"
#include "gpucompress.h"

#include <AMReX_MultiFab.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>

#include <hdf5.h>
#include <string>
#include <vector>
#include <cstdio>

namespace gpucompress_nyx_bridge {

/**
 * Write an entire MultiFab to a compressed HDF5 file using the VOL connector.
 *
 * Each FArrayBox in the MultiFab is written as a separate dataset.
 * Data flows: GPU device memory → VOL detects device pointer → GPU compression
 *           → H5Dwrite_chunk (pre-compressed) → HDF5 file
 *
 * @param filename     Output HDF5 file path
 * @param mf           MultiFab to write (data should be on GPU)
 * @param varnames     Variable names for each component
 * @param fapl         File access property list with VOL connector configured
 * @param chunk_bytes  HDF5 chunk size in bytes (default 4 MiB)
 * @param algo         Compression algorithm (default AUTO)
 * @param error_bound  Error bound (0.0 = lossless)
 * @return Number of chunks compressed, or -1 on error
 */
inline int write_multifab_compressed(
    const std::string& filename,
    const amrex::MultiFab& mf,
    const amrex::Vector<std::string>& varnames,
    hid_t fapl,
    size_t chunk_bytes = 4 * 1024 * 1024,
    gpucompress_algorithm_t algo = GPUCOMPRESS_ALGO_AUTO,
    double error_bound = 0.0)
{
    int ncomp = mf.nComp();
    int total_chunks_compressed = 0;

    hid_t fid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (fid < 0) return -1;

    /* Write each component as a separate 1D dataset for simplicity.
     * Each local FArrayBox contributes its cells for that component. */
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::FArrayBox& fab = mf[mfi];
        const amrex::Box& box = mfi.validbox();
        long ncells = box.numPts();

        /* Get device pointer to the entire FArrayBox data.
         * In AMReX GPU builds, fab.dataPtr() returns a managed/device pointer. */
        const amrex::Real* d_ptr = fab.dataPtr();

        /* Total floats: ncells * ncomp (contiguous in FArrayBox) */
        size_t total_reals = (size_t)ncells * ncomp;
        size_t total_bytes = total_reals * sizeof(amrex::Real);

        /* Determine chunk size in elements */
        size_t chunk_elems = chunk_bytes / sizeof(amrex::Real);
        if (chunk_elems > total_reals) chunk_elems = total_reals;
        if (chunk_elems == 0) chunk_elems = total_reals;

        hsize_t dims[1]  = { (hsize_t)total_reals };
        hsize_t cdims[1] = { (hsize_t)chunk_elems };

        /* Create dataset name: "fab_XXXX" */
        char dset_name[64];
        snprintf(dset_name, sizeof(dset_name), "fab_%04d", mfi.index());

        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, cdims);

        /* Determine HDF5 type based on AMReX Real precision */
        hid_t h5type = (sizeof(amrex::Real) == 8)
                     ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;

        /* Set GPUCompress filter with chosen algorithm */
        unsigned int shuffle_size = (unsigned int)sizeof(amrex::Real);
        H5Pset_gpucompress(dcpl, algo, 0, shuffle_size, error_bound);

        hid_t dset = H5Dcreate2(fid, dset_name, h5type, space,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset >= 0)
        {
            /* Pass device pointer directly — VOL connector detects it
             * and compresses on GPU without CPU round-trip */
            H5VL_gpucompress_reset_stats();
            H5Dwrite(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     (const void*)d_ptr);

            int n_comp = 0;
            H5VL_gpucompress_get_stats(NULL, NULL, &n_comp, NULL);
            total_chunks_compressed += n_comp;

            H5Dclose(dset);
        }

        H5Pclose(dcpl);
        H5Sclose(space);
    }

    /* Write metadata: variable names as attribute */
    if (varnames.size() > 0)
    {
        hid_t attr_space = H5Screate(H5S_SCALAR);
        hid_t str_type = H5Tcopy(H5T_C_S1);

        /* Concatenate varnames with comma separator */
        std::string all_names;
        for (int i = 0; i < (int)varnames.size(); i++) {
            if (i > 0) all_names += ",";
            all_names += varnames[i];
        }
        H5Tset_size(str_type, all_names.size() + 1);

        hid_t attr = H5Acreate2(fid, "varnames", str_type, attr_space,
                                 H5P_DEFAULT, H5P_DEFAULT);
        if (attr >= 0) {
            H5Awrite(attr, str_type, all_names.c_str());
            H5Aclose(attr);
        }
        H5Tclose(str_type);
        H5Sclose(attr_space);
    }

    H5Fclose(fid);
    return total_chunks_compressed;
}

/**
 * Write a single FArrayBox (one grid patch) to compressed HDF5.
 * Useful for per-box compression during checkpoint writes.
 */
inline int write_fab_compressed(
    const std::string& filename,
    const std::string& dset_name,
    const amrex::FArrayBox& fab,
    int ncomp,
    hid_t fapl,
    size_t chunk_bytes = 4 * 1024 * 1024,
    gpucompress_algorithm_t algo = GPUCOMPRESS_ALGO_AUTO,
    double error_bound = 0.0)
{
    const amrex::Real* d_ptr = fab.dataPtr();
    long ncells = fab.box().numPts();
    size_t total_reals = (size_t)ncells * ncomp;

    size_t chunk_elems = chunk_bytes / sizeof(amrex::Real);
    if (chunk_elems > total_reals) chunk_elems = total_reals;
    if (chunk_elems == 0) chunk_elems = total_reals;

    hsize_t dims[1]  = { (hsize_t)total_reals };
    hsize_t cdims[1] = { (hsize_t)chunk_elems };

    hid_t fid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (fid < 0) return -1;

    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    hid_t h5type = (sizeof(amrex::Real) == 8)
                 ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
    unsigned int shuffle_size = (unsigned int)sizeof(amrex::Real);
    H5Pset_gpucompress(dcpl, algo, 0, shuffle_size, error_bound);

    hid_t dset = H5Dcreate2(fid, dset_name.c_str(), h5type, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    int n_compressed = 0;
    if (dset >= 0) {
        H5VL_gpucompress_reset_stats();
        H5Dwrite(dset, h5type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 (const void*)d_ptr);
        H5VL_gpucompress_get_stats(NULL, NULL, &n_compressed, NULL);
        H5Dclose(dset);
    }

    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(fid);
    return n_compressed;
}

} /* namespace gpucompress_nyx_bridge */

#endif /* NYX_AMREX_BRIDGE_HPP */
