/**
 * demo_gpu_pipeline.cu
 *
 * Step-by-step walkthrough of the GPU compression pipeline via the VOL connector:
 *
 *   Phase 1 — Generate 8 MB of float32 data entirely on the GPU
 *   Phase 2 — Write to HDF5: data stays on GPU, only compressed bytes cross to host
 *   Phase 3 — Inspect the written file: filter config, chunk sizes, on-disk header
 *   Phase 4 — Read back: compressed bytes go host→GPU, decompressed on GPU
 *   Phase 5 — Verify bitwise integrity against the original GPU buffer
 *
 * Transfer budget per chunk (4 MB chunk, ~2.37 MB compressed):
 *   WRITE  H→D   64 B   CompressionHeader into d_compressed
 *          D→H  ~2.37MB compressed payload → pinned host → HDF5 disk I/O
 *   READ   H→D  ~2.37MB disk → pinned host → d_compressed
 *          D→H   64 B   readHeaderFromDevice (decompressor reads params)
 *
 * NOTE: HDF5's filter pipeline is bypassed by the VOL.  The chunk_dims set in
 * the DCPL are read by the VOL for iteration, but the actual chunking and
 * compression are handled inside gpu_aware_chunked_write/read, not by HDF5.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "hdf5/H5Zgpucompress.h"
#include "compression/compression_header.h"   /* CompressionHeader struct */

/* ------------------------------------------------------------------ */
#define N_ELEM   (2097152)       /* 8 MB as float32 (2M elements)  */
#define CHUNK    (1048576)       /* 4 MB per chunk  (1M elements)  */
#define N_CHUNK  (N_ELEM / CHUNK)
#define FNAME    "/tmp/demo_gpu_pipeline.h5"
#define DSET     "sensor_data"

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ void gen_kernel(float *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = (float)i / n;
    d[i] = (__sinf(t * 628.318f) + 0.4f * __cosf(t * 232.478f)
            + 0.1f * __sinf(t * 1884.96f)) * __expf(-t * 3.0f);
}

__global__ void verify_kernel(const float *ref, const float *out,
                               int n, int *mismatches) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (ref[i] != out[i]) atomicAdd(mismatches, 1);
}

/* ------------------------------------------------------------------ */
static void separator(const char *title) {
    printf("\n══════════════════════════════════════════════\n");
    printf("  %s\n", title);
    printf("══════════════════════════════════════════════\n");
}

/* ------------------------------------------------------------------ */
int main(void)
{
    int rc = 0;

    /* ---- Init ---- */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n"); return 1;
    }
    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "H5Z_gpucompress_register failed\n"); return 1;
    }
    hid_t vol_id    = H5VL_gpucompress_register();
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl      = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    /* ==============================================================
     * Phase 1: Generate data on GPU
     * ============================================================== */
    separator("Phase 1 — Generate data on GPU");

    float *d_src = NULL, *d_dst = NULL;
    cudaMalloc(&d_src, N_ELEM * sizeof(float));
    cudaMalloc(&d_dst, N_ELEM * sizeof(float));
    cudaMemset(d_dst, 0, N_ELEM * sizeof(float));

    gen_kernel<<<(N_ELEM + 255) / 256, 256>>>(d_src, N_ELEM);
    cudaDeviceSynchronize();

    printf("  d_src = %p  (device memory, %d floats = %.1f MB)\n",
           (void*)d_src, N_ELEM, N_ELEM * 4.0 / (1<<20));
    printf("  CPU has no copy of raw data at this point.\n");

    /* ==============================================================
     * Phase 2: Write — GPU device pointer passed directly to H5Dwrite
     * ============================================================== */
    separator("Phase 2 — Write (GPU ptr → VOL → GPU compress → D→H → disk)");

    hsize_t dims[1]  = { N_ELEM };
    hsize_t cdims[1] = { CHUNK  };

    hid_t fid   = H5Fcreate(FNAME, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_LZ4, 0, 4, 0.0);
    hid_t dset = H5Dcreate2(fid, DSET, H5T_NATIVE_FLOAT, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    H5VL_gpucompress_reset_stats();
    double t0 = now_sec();
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_src);
    double write_s = now_sec() - t0;

    int writes, comp_chunks;
    H5VL_gpucompress_get_stats(&writes, NULL, &comp_chunks, NULL);
    printf("  Write time    : %.3f s  (%.1f MB/s)\n",
           write_s, N_ELEM * 4.0 / (1<<20) / write_s);
    printf("  VOL confirms  : gpu_writes=%d  chunks_compressed=%d\n",
           writes, comp_chunks);
    printf("  Transfer log  :\n");
    printf("    per chunk:  H→D  64 B   (CompressionHeader → d_compressed)\n");
    printf("    per chunk:  D→H  ~2.37MB (compressed payload → pinned host → disk)\n");
    printf("    raw floats: NO D→H transfer of uncompressed data\n");

    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(fid);

    /* ==============================================================
     * Phase 3: Inspect the written file
     * ============================================================== */
    separator("Phase 3 — Inspect written file");

    fid  = H5Fopen(FNAME, H5F_ACC_RDONLY, fapl);
    dset = H5Dopen2(fid, DSET, H5P_DEFAULT);

    /* File size on disk */
    struct stat st;
    stat(FNAME, &st);
    printf("  File on disk  : %.3f MB  (%s)\n\n",
           st.st_size / (1.0*(1<<20)), FNAME);

    /* DCPL: filter pipeline as stored in the file */
    hid_t dcpl2 = H5Dget_create_plist(dset);
    int nfilters = H5Pget_nfilters(dcpl2);
    printf("  DCPL filter pipeline (%d filter):\n", nfilters);
    for (int fi = 0; fi < nfilters; fi++) {
        unsigned int flags = 0;
        size_t       cd_nelmts = 16;
        unsigned int cd_values[16] = {0};
        char fname[64] = {0};
        H5Z_filter_t fid2 = H5Pget_filter2(dcpl2, (unsigned)fi, &flags,
                                             &cd_nelmts, cd_values,
                                             sizeof(fname), fname, NULL);
        printf("    filter[%d]  id=%-4d  name=\"%s\"\n", fi, (int)fid2, fname);
        printf("              cd_values[0..%zu]: ", cd_nelmts);
        for (size_t k = 0; k < cd_nelmts; k++) printf("%u ", cd_values[k]);
        printf("\n");
        printf("              cd_values[0]=algo(%u=LZ4)  [1]=preproc(%u)"
               "  [2]=shuffle_sz(%u)\n",
               cd_values[0], cd_values[1], cd_values[2]);
    }
    H5Pclose(dcpl2);

    /* Chunk layout */
    printf("\n  Chunk storage:\n");
    hsize_t total_comp = 0;
    for (int c = 0; c < N_CHUNK; c++) {
        hsize_t offset[1] = { (hsize_t)c * CHUNK };
        hsize_t csz = 0;
        H5Dget_chunk_storage_size(dset, offset, &csz);
        printf("    chunk[%d]  raw=%.1f MB  on-disk=%.3f MB  ratio=%.2fx\n",
               c, CHUNK * 4.0/(1<<20), csz/(1.0*(1<<20)), CHUNK*4.0/csz);
        total_comp += csz;
    }
    printf("    total     raw=%.1f MB  on-disk=%.3f MB  ratio=%.2fx\n",
           N_ELEM * 4.0/(1<<20), total_comp/(1.0*(1<<20)),
           N_ELEM*4.0/(double)total_comp);

    /* Read the raw first chunk from file and inspect the CompressionHeader */
    printf("\n  CompressionHeader (prepended to each compressed chunk on disk):\n");
    hsize_t first_csz = 0;
    hsize_t off0[1]   = {0};
    H5Dget_chunk_storage_size(dset, off0, &first_csz);

    void    *raw_chunk   = malloc(first_csz);
    uint32_t filter_mask = 0;
    size_t   chunk_buf_size = first_csz;
    /* H5Dread_chunk reads raw bytes without applying any filter pipeline */
    H5Dread_chunk(dset, H5P_DEFAULT, off0, &filter_mask, raw_chunk, &chunk_buf_size);

    CompressionHeader *hdr = (CompressionHeader*)raw_chunk;
    printf("    magic          : 0x%08X  (%s)\n", hdr->magic,
           hdr->magic == COMPRESSION_MAGIC ? "\"GPUC\" valid" : "INVALID");
    printf("    version        : %u\n", hdr->version);
    printf("    original_size  : %llu B  (%.1f MB)\n",
           (unsigned long long)hdr->original_size,
           hdr->original_size / (1.0*(1<<20)));
    printf("    compressed_size: %llu B  (%.3f MB)\n",
           (unsigned long long)hdr->compressed_size,
           hdr->compressed_size / (1.0*(1<<20)));
    printf("    shuffle        : %s (element_size=%u)\n",
           hdr->hasShuffleApplied() ? "yes" : "no", hdr->shuffle_element_size);
    printf("    quantization   : %s\n",
           hdr->hasQuantizationApplied() ? "yes" : "no");
    printf("    NOTE: algorithm ID is NOT in this header —\n");
    printf("          it is encoded in the nvcomp LZ4 payload that follows.\n");
    free(raw_chunk);

    H5Dclose(dset);
    H5Fclose(fid);

    /* ==============================================================
     * Phase 4: Read back — compressed bytes go disk → host → GPU → decompress
     * ============================================================== */
    separator("Phase 4 — Read back (disk → host → H→D → GPU decompress)");

    fid  = H5Fopen(FNAME, H5F_ACC_RDONLY, fapl);
    dset = H5Dopen2(fid, DSET, H5P_DEFAULT);

    printf("  d_dst = %p  (device memory, zeroed, will receive decompressed data)\n",
           (void*)d_dst);
    printf("  Transfer sequence per chunk:\n");
    printf("    1. HDF5 reads compressed bytes from disk → pinned host buffer\n");
    printf("    2. H→D  ~2.37MB  compressed bytes → d_compressed (GPU)\n");
    printf("    3. D→H    64 B   readHeaderFromDevice (decompressor reads CompressionHeader)\n");
    printf("    4. nvcomp LZ4 decompresses on GPU entirely → d_dst (direct)\n");

    H5VL_gpucompress_reset_stats();
    double t1     = now_sec();
    herr_t rret   = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, d_dst);
    double read_s = now_sec() - t1;

    if (rret < 0) { fprintf(stderr, "H5Dread failed\n"); rc = 1; goto cleanup; }

    int reads, decomp_chunks;
    H5VL_gpucompress_get_stats(NULL, &reads, NULL, &decomp_chunks);
    printf("\n  Read time     : %.3f s  (%.1f MB/s)\n",
           read_s, N_ELEM * 4.0/(1<<20) / read_s);
    printf("  VOL confirms  : gpu_reads=%d  chunks_decompressed=%d\n",
           reads, decomp_chunks);

    H5Dclose(dset);
    H5Fclose(fid);

    /* ==============================================================
     * Phase 5: Verify — bitwise comparison entirely on GPU
     * ============================================================== */
    separator("Phase 5 — Verify integrity (GPU bitwise comparison)");

    {
        int *d_mm = NULL;
        cudaMalloc(&d_mm, sizeof(int));
        cudaMemset(d_mm, 0, sizeof(int));
        verify_kernel<<<(N_ELEM + 255)/256, 256>>>(d_src, d_dst, N_ELEM, d_mm);
        cudaDeviceSynchronize();
        int mismatches = 0;
        cudaMemcpy(&mismatches, d_mm, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_mm);

        if (mismatches == 0)
            printf("  PASS  %d / %d elements match exactly (lossless)\n",
                   N_ELEM, N_ELEM);
        else {
            printf("  FAIL  %d / %d elements differ\n", mismatches, N_ELEM);
            rc = 1;
        }
    }

    printf("\n");

cleanup:
    cudaFree(d_src);
    cudaFree(d_dst);
    H5Pclose(fapl);
    H5VLclose(vol_id);
    gpucompress_cleanup();
    return rc;
}
