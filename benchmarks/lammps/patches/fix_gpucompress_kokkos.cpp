/* ----------------------------------------------------------------------
   LAMMPS fix for GPU-accelerated compression via GPUCompress.

   Borrows KOKKOS device pointers and writes compressed HDF5 using
   GPUCompress's HDF5 VOL connector. Zero simulation source changes
   beyond this fix file.
------------------------------------------------------------------------- */

#include "fix_gpucompress_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "kokkos_type.h"
#include "update.h"
#include "comm.h"
#include "error.h"
#include "memory.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <cuda_runtime.h>

/* GPUCompress bridge — linked via cmake */
#include "lammps_gpucompress_udf.h"
#include "gpucompress.h"

/* Ranking profiler (CUDA, linked from lammps_ranking_profiler.cu) */
struct RankingMilestoneResult { double mean_tau, std_tau, mean_regret, profiling_ms; };
extern "C" int lammps_run_ranking_profiler(
    const void* d_data, size_t total_bytes, size_t chunk_bytes,
    double error_bound, float w0, float w1, float w2, float bw_bytes_per_ms,
    int n_repeats, FILE* csv, FILE* costs_csv,
    const char* phase_name, int timestep, RankingMilestoneResult* out);
extern "C" int lammps_is_ranking_milestone(int t, int total);
extern "C" void lammps_write_ranking_csv_header(FILE* csv);
extern "C" void lammps_write_ranking_costs_csv_header(FILE* csv);

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixGPUCompressKokkos::FixGPUCompressKokkos(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 4) error->all(FLERR, "Illegal fix gpucompress command");

  /* Parse: fix ID group gpucompress N [positions] [velocities] [forces] */
  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  if (nevery <= 0) error->all(FLERR, "fix gpucompress: N must be > 0");

  dump_every = nevery;
  dump_positions = 0;
  dump_velocities = 0;
  dump_forces = 0;

  for (int i = 4; i < narg; i++) {
    if (strcmp(arg[i], "positions") == 0) dump_positions = 1;
    else if (strcmp(arg[i], "velocities") == 0) dump_velocities = 1;
    else if (strcmp(arg[i], "forces") == 0) dump_forces = 1;
    else error->all(FLERR, "fix gpucompress: unknown field");
  }

  /* Default: dump all if nothing specified */
  if (!dump_positions && !dump_velocities && !dump_forces) {
    dump_positions = 1;
    dump_velocities = 1;
    dump_forces = 1;
  }

  algo_name = getenv("GPUCOMPRESS_ALGO");
  if (!algo_name) algo_name = "auto";

  const char *venv = getenv("GPUCOMPRESS_VERIFY");
  verify = (venv && atoi(venv)) ? 1 : 0;

  const char *denv = getenv("LAMMPS_DUMP_FIELDS");
  dump_raw_fields = (denv && atoi(denv)) ? 1 : 0;

  const char *lcenv = getenv("LAMMPS_LOG_CHUNKS");
  log_chunks = (lcenv && atoi(lcenv)) ? 1 : 0;
  tc_csv = NULL;
  ranking_csv = NULL;
  ranking_costs_csv = NULL;
  write_count = 0;
  const char *tw_env = getenv("GPUCOMPRESS_TOTAL_WRITES");
  total_writes = tw_env ? atoi(tw_env) : 10;
  cw0 = 1; cw1 = 1; cw2 = 1;

  gpuc_ready = 0;
}

FixGPUCompressKokkos::~FixGPUCompressKokkos()
{
  /* Don't finalize here — CUDA context may already be torn down.
   * GPUCompress cleanup happens via atexit or OS process teardown. */
}

/* ---------------------------------------------------------------------- */

int FixGPUCompressKokkos::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixGPUCompressKokkos::init()
{
  /* Initialize GPUCompress on first call */
  if (!gpuc_ready) {
    const char *weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) weights = "model.nnwt";  /* set GPUCOMPRESS_WEIGHTS env var */

    const char *policy = getenv("GPUCOMPRESS_POLICY");
    if (!policy) policy = "ratio";

    int rc = gpucompress_lammps_init(weights, policy);
    if (rc == 0) {
      gpuc_ready = 1;

      /* Enable online SGD learning + exploration for NN-RL phase */
      if (strcmp(algo_name, "auto") == 0) {
        const char *lr_env = getenv("GPUCOMPRESS_LR");
        const char *mape_env = getenv("GPUCOMPRESS_MAPE");
        const char *ek_env = getenv("GPUCOMPRESS_EXPLORE_K");
        const char *et_env = getenv("GPUCOMPRESS_EXPLORE_THRESH");
        float lr   = lr_env   ? (float)atof(lr_env)   : 0.2f;
        float mape = mape_env ? (float)atof(mape_env)  : 0.10f;
        int   ek   = ek_env   ? atoi(ek_env)           : 4;
        float et   = et_env   ? (float)atof(et_env)    : 0.20f;

        gpucompress_enable_online_learning();
        gpucompress_set_reinforcement(1, lr, mape, 0.0f);
        gpucompress_set_exploration(1);
        gpucompress_set_exploration_k(ek);
        gpucompress_set_exploration_threshold(et);

        if (comm->me == 0)
          fprintf(stdout, "[GPUCompress-LAMMPS] Online learning: lr=%.2f mape=%.2f explore_k=%d explore_thresh=%.2f\n",
                  lr, mape, ek, et);
      }

      if (comm->me == 0) {
        fprintf(stdout, "[GPUCompress-LAMMPS] Initialized: algo=%s policy=%s verify=%d\n",
                algo_name, policy, verify);
        fprintf(stdout, "[GPUCompress-LAMMPS] Fields: pos=%d vel=%d force=%d every=%d\n",
                dump_positions, dump_velocities, dump_forces, dump_every);

        /* Open per-chunk CSV if requested */
        if (log_chunks && !tc_csv) {
          const char *csv_dir = getenv("LAMMPS_LOG_DIR");
          if (!csv_dir) csv_dir = ".";
          mkdir(csv_dir, 0755);
          char csv_path[700];
          snprintf(csv_path, sizeof(csv_path),
                   "%s/benchmark_lammps_timestep_chunks.csv", csv_dir);
          tc_csv = fopen(csv_path, "w");
          if (tc_csv) {
            fprintf(tc_csv, "rank,phase,timestep,chunk,"
                    "predicted_ratio,actual_ratio,"
                    "predicted_comp_ms,actual_comp_ms_raw,"
                    "mape_ratio,mape_comp,"
                    "sgd_fired,exploration_triggered,"
                    "cost_model_error_pct,actual_cost,predicted_cost\n");
          }

          /* Ranking CSV (top1_regret from oracle comparison) */
          snprintf(csv_path, sizeof(csv_path),
                   "%s/benchmark_lammps_ranking.csv", csv_dir);
          ranking_csv = fopen(csv_path, "w");
          if (ranking_csv) lammps_write_ranking_csv_header(ranking_csv);

          snprintf(csv_path, sizeof(csv_path),
                   "%s/benchmark_lammps_ranking_costs.csv", csv_dir);
          ranking_costs_csv = fopen(csv_path, "w");
          if (ranking_costs_csv) lammps_write_ranking_costs_csv_header(ranking_costs_csv);

          /* Cost model weights for ranking profiler */
          if (strcmp(policy, "speed") == 0)    { cw0=1; cw1=0; cw2=0; }
          else if (strcmp(policy, "balanced") == 0) { cw0=1; cw1=1; cw2=1; }
          else { cw0=0; cw1=0; cw2=1; }

          fprintf(stdout, "[GPUCompress-LAMMPS] Chunk + ranking CSV: %s/\n", csv_dir);
        }
        fflush(stdout);
      }
    } else {
      if (comm->me == 0)
        fprintf(stderr, "[GPUCompress-LAMMPS] Init FAILED\n");
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixGPUCompressKokkos::setup(int /* vflag */)
{
  end_of_step();
}

/* ---------------------------------------------------------------------- */

void FixGPUCompressKokkos::end_of_step()
{
  if (!gpuc_ready) return;

  bigint ntimestep = update->ntimestep;

  /* Only dump at specified intervals */
  if (ntimestep % dump_every != 0) return;

  auto *atomKK = dynamic_cast<AtomKokkos *>(atom);
  if (!atomKK) {
    if (comm->me == 0)
      fprintf(stderr, "[GPUCompress-LAMMPS] Not running with KOKKOS!\n");
    return;
  }

  int nlocal = atom->nlocal;
  int elem_bytes = sizeof(KK_FLOAT);  /* auto-detect: 4 for SINGLE, 8 for DOUBLE */

  /* Create output directory */
  char dir[256];
  snprintf(dir, sizeof(dir), "gpuc_step_%010lld", (long long)ntimestep);
  if (comm->me == 0) mkdir(dir, 0755);
  MPI_Barrier(world);

  char fname[512];
  int rank = comm->me;
  int rc;

  /* Sync device views to ensure data is current */
  atomKK->sync(LAMMPS_NS::Device, X_MASK | V_MASK | F_MASK);

  /* Dump raw binary fields for cross-workload regret benchmarks */
  if (dump_raw_fields) {
    const char *raw_dir = getenv("LAMMPS_DUMP_DIR");
    if (!raw_dir) raw_dir = ".";

    auto write_raw = [&](const char *tag, const void *d_ptr, size_t n_elem) {
      size_t nbytes = n_elem * elem_bytes;
      void *h_buf = malloc(nbytes);
      if (!h_buf) return;
      cudaError_t cerr = cudaMemcpy(h_buf, d_ptr, nbytes, cudaMemcpyDeviceToHost);
      if (cerr != cudaSuccess) {
        fprintf(stderr, "[GPUCompress-LAMMPS] cudaMemcpy failed: %s\n",
                cudaGetErrorString(cerr));
        free(h_buf);
        return;
      }
      char rpath[700];
      snprintf(rpath, sizeof(rpath), "%s/%s_step%010lld.f32",
               raw_dir, tag, (long long)ntimestep);
      FILE *fp = fopen(rpath, "wb");
      if (fp) {
        if (elem_bytes == 4) {
          fwrite(h_buf, sizeof(float), n_elem, fp);
        } else {
          /* KK_FLOAT is double — downcast to float32 for generic_benchmark */
          float *f32 = (float *)malloc(n_elem * sizeof(float));
          if (f32) {
            const double *src = (const double *)h_buf;
            for (size_t i = 0; i < n_elem; i++) f32[i] = (float)src[i];
            fwrite(f32, sizeof(float), n_elem, fp);
            free(f32);
          }
        }
        fclose(fp);
      }
      free(h_buf);
    };

    if (dump_positions) {
      auto d_x = atomKK->k_x.view_device();
      write_raw("positions", d_x.data(), (size_t)nlocal * 3);
    }
    if (dump_velocities) {
      auto d_v = atomKK->k_v.view_device();
      write_raw("velocities", d_v.data(), (size_t)nlocal * 3);
    }
    if (dump_forces) {
      auto d_f = atomKK->k_f.view_device();
      write_raw("forces", d_f.data(), (size_t)nlocal * 3);
    }
    if (comm->me == 0) {
      fprintf(stdout, "[GPUCompress-LAMMPS] Step %lld: dumped raw fields to %s\n",
              (long long)ntimestep, raw_dir);
      fflush(stdout);
    }
  }

  /* Reset chunk history so per-write diagnostics are clean */
  gpucompress_reset_chunk_history();

  if (dump_positions) {
    /* Get raw CUDA device pointer from KOKKOS view
     * k_x is a DualView; view_device() returns the device Kokkos::View
     * .data() returns the raw pointer to the underlying CUDA allocation */
    auto d_x = atomKK->k_x.view_device();
    const void *d_ptr = (const void *)d_x.data();
    size_t n_elements = (size_t)nlocal * 3;

    snprintf(fname, sizeof(fname), "%s/x_rank%04d.h5", dir, rank);
    rc = gpucompress_lammps_write_field(fname, "positions", d_ptr,
                                         n_elements, elem_bytes,
                                         algo_name, 0.0, verify);
    if (comm->me == 0 && rc != 0)
      fprintf(stderr, "[GPUCompress-LAMMPS] positions write failed\n");
  }

  if (dump_velocities) {
    auto d_v = atomKK->k_v.view_device();
    const void *d_ptr = (const void *)d_v.data();
    size_t n_elements = (size_t)nlocal * 3;

    snprintf(fname, sizeof(fname), "%s/v_rank%04d.h5", dir, rank);
    rc = gpucompress_lammps_write_field(fname, "velocities", d_ptr,
                                         n_elements, elem_bytes,
                                         algo_name, 0.0, verify);
    if (comm->me == 0 && rc != 0)
      fprintf(stderr, "[GPUCompress-LAMMPS] velocities write failed\n");
  }

  if (dump_forces) {
    auto d_f = atomKK->k_f.view_device();
    const void *d_ptr = (const void *)d_f.data();
    size_t n_elements = (size_t)nlocal * 3;

    snprintf(fname, sizeof(fname), "%s/f_rank%04d.h5", dir, rank);
    rc = gpucompress_lammps_write_field(fname, "forces", d_ptr,
                                         n_elements, elem_bytes,
                                         algo_name, 0.0, verify);
    if (comm->me == 0 && rc != 0)
      fprintf(stderr, "[GPUCompress-LAMMPS] forces write failed\n");
  }

  /* Log per-chunk diagnostics to CSV */
  if (log_chunks && tc_csv && comm->me == 0) {
    int n_hist = gpucompress_get_chunk_history_count();
    const char *phase = (strcmp(algo_name, "auto") == 0) ? "nn-rl+exp50" : algo_name;
    for (int ci = 0; ci < n_hist; ci++) {
      gpucompress_chunk_diag_t dd;
      if (gpucompress_get_chunk_diag(ci, &dd) != 0) continue;
      double mr = 0, mc = 0;
      if (dd.actual_ratio > 0)
        mr = fabs(dd.predicted_ratio - dd.actual_ratio) / fabs(dd.actual_ratio) * 100.0;
      if (dd.compression_ms > 0)
        mc = fabs(dd.predicted_comp_time - dd.compression_ms) / fabs(dd.compression_ms) * 100.0;
      fprintf(tc_csv,
              "0,%s,%d,%d,"
              "%.4f,%.4f,%.4f,%.4f,"
              "%.2f,%.2f,%d,%d,"
              "%.4f,%.4f,%.4f\n",
              phase, write_count, ci,
              (double)dd.predicted_ratio, (double)dd.actual_ratio,
              (double)dd.predicted_comp_time, (double)dd.compression_ms_raw,
              mr, mc, dd.sgd_fired, dd.exploration_triggered,
              (double)dd.cost_model_error_pct,
              (double)dd.actual_cost, (double)dd.predicted_cost);
    }
    fflush(tc_csv);

    /* Ranking profiler at milestone writes */
    if (ranking_csv && lammps_is_ranking_milestone(write_count, total_writes)) {
      auto d_x = atomKK->k_x.view_device();
      const void *rank_ptr = (const void *)d_x.data();
      size_t rank_bytes = (size_t)nlocal * 3 * elem_bytes;
      float bw = gpucompress_get_bandwidth_bytes_per_ms();
      RankingMilestoneResult result = {};
      size_t chunk_bytes = 4 * 1024 * 1024; /* 4 MiB default */
      const char *cb_env = getenv("GPUCOMPRESS_CHUNK_MB");
      if (cb_env) chunk_bytes = (size_t)atoi(cb_env) * 1024 * 1024;
      lammps_run_ranking_profiler(rank_ptr, rank_bytes, chunk_bytes,
                                   0.0, cw0, cw1, cw2, bw,
                                   3, ranking_csv, ranking_costs_csv,
                                   phase, write_count, &result);
      fprintf(stdout, "    [ranking] T=%d: tau=%.3f regret=%.3fx (%.0fms)\n",
              write_count, result.mean_tau, result.mean_regret, result.profiling_ms);
      fflush(stdout);
    }
    write_count++;
  }
  gpucompress_reset_chunk_history();

  if (comm->me == 0) {
    int nfields = dump_positions + dump_velocities + dump_forces;
    double mb = (double)nlocal * 3 * elem_bytes * nfields / (1024.0 * 1024.0);
    fprintf(stdout, "[GPUCompress-LAMMPS] Step %lld: wrote %d fields (%.1f MB/rank) algo=%s\n",
            (long long)ntimestep, nfields, mb, algo_name);
    fflush(stdout);
  }
}
