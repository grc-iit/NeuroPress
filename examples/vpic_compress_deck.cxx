/**
 * @file vpic_compress_deck.cxx
 * @brief VPIC deck with real-time GPU compression to HDF5 via VOL connector
 *
 * This is a real VPIC-Kokkos input deck (Harris sheet reconnection) that
 * intercepts GPU-resident field, hydro, and particle data during the
 * simulation and writes them to HDF5 files using the GPUCompress VOL
 * connector — compression happens entirely on the GPU with zero CPU copies.
 *
 * Pipeline per timestep:
 *   Kokkos::View (GPU) → .data() → H5Dwrite(d_ptr) → VOL intercepts →
 *   GPU compress (nvcomp) → H5Dwrite_chunk (pre-compressed) → HDF5 file
 *
 * The VOL connector detects that the pointer passed to H5Dwrite() is a
 * CUDA device pointer, compresses each chunk on the GPU, and writes the
 * pre-compressed bytes via H5Dwrite_chunk() — bypassing the HDF5 filter
 * pipeline entirely.
 *
 * BUILD (single-line, from vpic-kokkos directory):
 *   export CRAYPE_LINK_TYPE=dynamic
 *   cmake -S . -B build-compress -DENABLE_KOKKOS_CUDA=ON -DBUILD_INTERNAL_KOKKOS=ON -DENABLE_KOKKOS_OPENMP=OFF -DCMAKE_CXX_STANDARD=17 -DKokkos_ARCH_AMPERE80=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DUSER_DECKS="/u/imuradli/GPUCompress/examples/vpic_compress_deck.cxx" '-DCMAKE_CXX_FLAGS=-I/u/imuradli/GPUCompress/include -I/u/imuradli/GPUCompress/examples -I/tmp/hdf5-install/include' '-DCMAKE_EXE_LINKER_FLAGS=-L/u/imuradli/GPUCompress/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L/tmp/hdf5-install/lib -lhdf5 -L/tmp/lib -lnvcomp'
 *   cmake --build build-compress -j$(nproc)
 *
 * RUN:
 *   export LD_LIBRARY_PATH=/u/imuradli/GPUCompress/build:/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
 *   mpirun -np 1 ./build-compress/vpic_compress_deck
 *
 * OUTPUT FILES (one per diagnostic dump):
 *   /tmp/vpic_fields_NNNNNN.h5    — compressed EM field data
 *   /tmp/vpic_hydro_NNNNNN.h5     — compressed hydro moments
 *   /tmp/vpic_particles_NNNNNN.h5 — compressed particle data (all species)
 */

// ============================================================
// GPUCompress + HDF5 headers
// ============================================================
#include "gpucompress.h"
#include "gpucompress_vpic.h"
#include "gpucompress_hdf5_vol.h"
#include "gpucompress_hdf5.h"
#include "vpic_kokkos_bridge.hpp"

#include <hdf5.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// ============================================================
// Globals: persist across timesteps
// ============================================================
begin_globals {
    int                diag_interval;      // Compress+write every N steps
    size_t             chunk_bytes;        // HDF5 chunk size in bytes
    gpucompress_vpic_t vpic_fields_h;      // Adapter handle for fields
    gpucompress_vpic_t vpic_hydro_h;       // Adapter handle for hydro
    gpucompress_vpic_t vpic_particles_h;   // Adapter handle for particles
    hid_t              vol_fapl;           // File access property list with VOL
    hid_t              vol_id;             // VOL connector ID
    int                gpucompress_ready;  // 1 if init succeeded
};

// Helper: write a GPU-resident float array to a compressed HDF5 dataset
static void write_gpu_to_hdf5(const char* filename, const char* dset_name,
                               float* d_data, size_t n_floats,
                               size_t chunk_floats, hid_t fapl)
{
    if (chunk_floats > n_floats) chunk_floats = n_floats;
    hsize_t dims[1]  = { (hsize_t)n_floats };
    hsize_t cdims[1] = { (hsize_t)chunk_floats };

    hid_t fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (fid < 0) return;

    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);

    hid_t dset = H5Dcreate2(fid, dset_name, H5T_NATIVE_FLOAT, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    // d_data is a CUDA device pointer — the VOL connector detects this
    // and compresses on the GPU, then writes pre-compressed chunks
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(space);
    H5Fclose(fid);
}

// ============================================================
// Initialization: Harris sheet + GPUCompress + VOL setup
// ============================================================
begin_initialization {
    // ---- Physics (minimal Harris sheet) ----
    double L    = 1;
    double ec   = 1;
    double me   = 1;
    double c    = 1;
    double eps0 = 1;

    double mi_me   = 25;
    double rhoi_L  = 1;
    double Ti_Te   = 1;
    double wpe_wce = 3;

    // Grid: 256x128x32 cells, 70 particles/cell
    // Fields: ~67 MB, Hydro: ~59 MB, Particles: ~3.8 GB → ~4 GB per snapshot
    double Lx   = 64*L;
    double Ly   = 32*L;
    double Lz   = 8*L;
    double nx   = 256;
    double ny   = 128;
    double nz   = 32;
    double nppc = 70;

    double damp      = 0.001;
    double cfl_req   = 0.99;
    double wpedt_max = 0.36;

    double mi   = me*mi_me;
    double kTe  = me*c*c/(2*wpe_wce*wpe_wce*(1+Ti_Te));
    double kTi  = kTe*Ti_Te;
    double vthi = sqrt(2*kTi/mi);
    double wci  = vthi/(rhoi_L*L);
    double wce  = wci*mi_me;
    double wpe  = wce*wpe_wce;
    double vdre = c*c*wce/(wpe*wpe*L*(1+Ti_Te));
    double vdri = -Ti_Te*vdre;
    double b0   = me*wce/ec;
    double n0   = me*eps0*wpe*wpe/(ec*ec);
    double Npe  = 2*n0*Ly*Lz*L*tanh(0.5*Lx/L);
    double Npi  = Npe;
    double Ne   = 0.5*nppc*nx*ny*nz;
    Ne = trunc_granular(Ne, nproc());
    double Ni   = Ne;
    double we   = Npe/Ne;
    double wi   = Npi/Ni;
    double gdri = 1/sqrt(1 - vdri*vdri/(c*c));
    double gdre = 1/sqrt(1 - vdre*vdre/(c*c));
    double udri = vdri*gdri;
    double udre = vdre*gdre;
    double uthi = sqrt(kTi/mi)/c;
    double uthe = sqrt(kTe/me)/c;

    double dg = courant_length(Lx, Ly, Lz, nx, ny, nz);
    double dt = cfl_req*dg/c;
    if (wpe*dt > wpedt_max) dt = wpedt_max/wpe;

    // Simulation parameters
    num_step        = 50;
    status_interval = 10;
    clean_div_e_interval = 10;
    clean_div_b_interval = 10;

    // Diagnostics: write compressed HDF5 every 10 steps (5 snapshots total)
    global->diag_interval = 10;
    global->chunk_bytes   = 4 * 1024 * 1024;  // 64 MB chunks

    // Grid setup
    define_units(c, eps0);
    define_timestep(dt);
    define_periodic_grid(-0.5*Lx, 0, 0,
                          0.5*Lx, Ly, Lz,
                          nx, ny, nz,
                          1, nproc(), 1);

    set_domain_field_bc(BOUNDARY(-1,0,0), pec_fields);
    set_domain_field_bc(BOUNDARY( 1,0,0), pec_fields);
    set_domain_particle_bc(BOUNDARY(-1,0,0), reflect_particles);
    set_domain_particle_bc(BOUNDARY( 1,0,0), reflect_particles);

    define_material("vacuum", 1);
    define_field_array(NULL, damp);

    species_t* ion      = define_species("ion",       ec, mi, 1.5*Ni/nproc(), -1, 40, 1);
    species_t* electron = define_species("electron", -ec, me, 1.5*Ne/nproc(), -1, 20, 1);

    // Load fields
    set_region_field(everywhere,
                     0, 0, 0,
                     0, 0, b0*tanh(x/L));

    // Load particles
    double ymin = rank()*Ly/nproc();
    double ymax = (rank()+1)*Ly/nproc();

    repeat(Ni/nproc()) {
        double px, py, pz, ux, uy, uz, d0;
        do { px = L*atanh(uniform(rng(0), -1, 1)); }
        while (px <= -0.5*Lx || px >= 0.5*Lx);
        py = uniform(rng(0), ymin, ymax);
        pz = uniform(rng(0), 0, Lz);

        ux = normal(rng(0), 0, uthi);
        uy = normal(rng(0), 0, uthi);
        uz = normal(rng(0), 0, uthi);
        d0 = gdri*uy + sqrt(ux*ux + uy*uy + uz*uz + 1)*udri;
        uy = d0; uz = uz;
        inject_particle(ion, px, py, pz, ux, uy, uz, wi, 0, 0);

        ux = normal(rng(0), 0, uthe);
        uy = normal(rng(0), 0, uthe);
        uz = normal(rng(0), 0, uthe);
        d0 = gdre*uy + sqrt(ux*ux + uy*uy + uz*uz + 1)*udre;
        uy = d0; uz = uz;
        inject_particle(electron, px, py, pz, ux, uy, uz, we, 0, 0);
    }

    // ---- GPUCompress + HDF5 VOL initialization ----
    global->gpucompress_ready = 0;

    gpucompress_error_t gerr = gpucompress_init(NULL);  // NULL = no NN weights, fallback to LZ4
    if (gerr != GPUCOMPRESS_SUCCESS) {
        sim_log("WARNING: gpucompress_init failed (" << gerr << "), compression disabled");
    } else {
        // Register HDF5 filter and VOL connector
        H5Z_gpucompress_register();
        global->vol_id = H5VL_gpucompress_register();

        hid_t native_id = H5VLget_connector_id_by_name("native");
        global->vol_fapl = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_gpucompress(global->vol_fapl, native_id, NULL);
        H5VLclose(native_id);

        H5VL_gpucompress_set_trace(0);
        global->gpucompress_ready = 1;

        sim_log("GPUCompress VOL connector initialized");
    }

    // Create VPIC adapter handles
    VpicSettings fs = vpic_default_settings();
    fs.data_type = VPIC_DATA_FIELDS;
    fs.n_cells   = grid->nv;
    gpucompress_vpic_create(&global->vpic_fields_h, &fs);

    VpicSettings hs = vpic_default_settings();
    hs.data_type = VPIC_DATA_HYDRO;
    hs.n_cells   = grid->nv;
    gpucompress_vpic_create(&global->vpic_hydro_h, &hs);

    VpicSettings ps = vpic_default_settings();
    ps.data_type   = VPIC_DATA_PARTICLES;
    ps.n_particles = 0;
    gpucompress_vpic_create(&global->vpic_particles_h, &ps);

    size_t field_bytes = (size_t)grid->nv * 16 * sizeof(float);
    size_t hydro_bytes = (size_t)grid->nv * 14 * sizeof(float);
    sim_log("VPIC data sizes on GPU:");
    sim_log("  Fields : " << grid->nv << " voxels x 16 vars = "
            << field_bytes / (1024*1024) << " MB");
    sim_log("  Hydro  : " << grid->nv << " voxels x 14 vars = "
            << hydro_bytes / (1024*1024) << " MB");
    sim_log("  Chunk  : " << global->chunk_bytes / (1024*1024) << " MB");
}

// ============================================================
// Diagnostics: intercept GPU data → compress on GPU → write HDF5
// ============================================================
begin_diagnostics {
    if (step() % global->diag_interval != 0) return;
    if (!global->gpucompress_ready) return;

    char filename[256];
    size_t chunk_floats = global->chunk_bytes / sizeof(float);

    sim_log("Step " << step() << ": writing GPU-compressed HDF5...");

    // ---- 1) Fields → HDF5 (GPU-resident, zero CPU copies) ----
    {
        vpic_attach_fields(global->vpic_fields_h, field_array->k_f_d);

        float*  d_data = NULL;
        size_t  nbytes = 0;
        gpucompress_vpic_get_device_ptrs(global->vpic_fields_h,
                                         &d_data, NULL, &nbytes, NULL);
        size_t n_floats = nbytes / sizeof(float);

        snprintf(filename, sizeof(filename),
                 "/tmp/vpic_fields_%06d.h5", step());

        H5VL_gpucompress_reset_stats();
        write_gpu_to_hdf5(filename, "fields", d_data, n_floats,
                          chunk_floats, global->vol_fapl);

        int n_comp = 0;
        H5VL_gpucompress_get_stats(NULL, NULL, &n_comp, NULL);
        sim_log("  Fields: " << nbytes/(1024*1024) << " MB -> "
                << filename << " (" << n_comp << " chunks compressed on GPU)");
    }

    // ---- 2) Hydro → HDF5 ----
    {
        vpic_attach_hydro(global->vpic_hydro_h, hydro_array->k_h_d);

        float*  d_data = NULL;
        size_t  nbytes = 0;
        gpucompress_vpic_get_device_ptrs(global->vpic_hydro_h,
                                         &d_data, NULL, &nbytes, NULL);
        size_t n_floats = nbytes / sizeof(float);

        snprintf(filename, sizeof(filename),
                 "/tmp/vpic_hydro_%06d.h5", step());

        H5VL_gpucompress_reset_stats();
        write_gpu_to_hdf5(filename, "hydro", d_data, n_floats,
                          chunk_floats, global->vol_fapl);

        int n_comp = 0;
        H5VL_gpucompress_get_stats(NULL, NULL, &n_comp, NULL);
        sim_log("  Hydro : " << nbytes/(1024*1024) << " MB -> "
                << filename << " (" << n_comp << " chunks compressed on GPU)");
    }

    // ---- 3) Particles → HDF5 (one file per dump, one dataset per species) ----
    {
        snprintf(filename, sizeof(filename),
                 "/tmp/vpic_particles_%06d.h5", step());

        hid_t fid = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT,
                               global->vol_fapl);

        species_t* sp;
        LIST_FOR_EACH(sp, species_list) {
            if (sp->np == 0) continue;

            vpic_attach_particles(global->vpic_particles_h,
                                  sp->k_p_d, sp->k_p_i_d);

            float*  d_data = NULL;
            size_t  nbytes_f = 0;
            gpucompress_vpic_get_device_ptrs(global->vpic_particles_h,
                                             &d_data, NULL, &nbytes_f, NULL);
            size_t n_floats = nbytes_f / sizeof(float);

            hsize_t dims[1]  = { (hsize_t)n_floats };
            hsize_t cdims[1] = { (hsize_t)chunk_floats };
            if (cdims[0] > dims[0]) cdims[0] = dims[0];

            hid_t space = H5Screate_simple(1, dims, NULL);
            hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcpl, 1, cdims);
            H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);

            hid_t dset = H5Dcreate2(fid, sp->name, H5T_NATIVE_FLOAT, space,
                                     H5P_DEFAULT, dcpl, H5P_DEFAULT);

            H5VL_gpucompress_reset_stats();
            H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                     H5P_DEFAULT, d_data);

            int n_comp = 0;
            H5VL_gpucompress_get_stats(NULL, NULL, &n_comp, NULL);
            sim_log("  Particles (" << sp->name << "): "
                    << sp->np << " particles, "
                    << nbytes_f/(1024*1024) << " MB ("
                    << n_comp << " chunks compressed on GPU)");

            H5Dclose(dset);
            H5Pclose(dcpl);
            H5Sclose(space);
        }

        H5Fclose(fid);
    }

    sim_log("  Done. Files written to /tmp/vpic_*_" << step() << ".h5");
}

begin_particle_injection {
    // No injection
}

begin_current_injection {
    // No injection
}

begin_field_injection {
    // No injection
}

begin_particle_collisions {
    // No collisions
}
