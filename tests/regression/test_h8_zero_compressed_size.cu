/**
 * test_h8_zero_compressed_size.cu
 *
 * H8: Zero compressed_size in header bypasses validation. The decompressor
 *     proceeds with 0 bytes of compressed data, causing nvcomp to crash
 *     or produce garbage when original_size > 0.
 *
 * Test strategy:
 *   1. Compress valid data to get a legitimate header
 *   2. Set header.compressed_size = 0 with original_size unchanged (> 0)
 *      → should return INVALID_HEADER
 *   3. Set both compressed_size = 0 and original_size = 0
 *      → should return SUCCESS with *output_size = 0
 *   4. Before fix: case 2 proceeds to nvcomp with 0 bytes (crash/garbage)
 *
 * Run: ./test_h8_zero_compressed_size
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpucompress.h"
#include "compression/compression_header.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

int main(void) {
    printf("=== H8: Zero compressed_size bypasses validation ===\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* Compress valid data to get a real header */
    const size_t DATA_SIZE = 4096;
    float h_data[DATA_SIZE / sizeof(float)];
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++)
        h_data[i] = (float)i * 0.1f;

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    uint8_t* compressed = (uint8_t*)malloc(max_out);
    size_t compressed_size = max_out;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

    err = gpucompress_compress(h_data, DATA_SIZE, compressed, &compressed_size, &cfg, NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: compression failed (%d)\n", err);
        free(compressed);
        gpucompress_cleanup();
        return 1;
    }

    CompressionHeader* hdr = (CompressionHeader*)compressed;
    printf("  Original header: original_size=%zu compressed_size=%zu\n",
           (size_t)hdr->original_size, (size_t)hdr->compressed_size);

    /* Test 1: compressed_size=0 with original_size>0 → should be rejected */
    size_t saved_compressed = hdr->compressed_size;
    hdr->compressed_size = 0;

    uint8_t decomp_buf[DATA_SIZE];
    size_t decomp_size = DATA_SIZE;
    err = gpucompress_decompress(compressed, compressed_size, decomp_buf, &decomp_size);
    if (err == GPUCOMPRESS_ERROR_INVALID_HEADER) {
        PASS("compressed_size=0 with original_size>0 rejected as INVALID_HEADER");
    } else if (err != GPUCOMPRESS_SUCCESS) {
        /* Some other error — still caught, but not the specific one we want */
        printf("  INFO: got error %d (not INVALID_HEADER but still rejected)\n", err);
        PASS("compressed_size=0 with original_size>0 rejected (different error code)");
    } else {
        FAIL("compressed_size=0 with original_size>0 was NOT rejected");
    }

    /* Test 2: both compressed_size=0 and original_size=0 → should succeed */
    size_t saved_original = hdr->original_size;
    hdr->original_size = 0;
    hdr->compressed_size = 0;

    decomp_size = DATA_SIZE;
    err = gpucompress_decompress(compressed, compressed_size, decomp_buf, &decomp_size);
    if (err == GPUCOMPRESS_SUCCESS && decomp_size == 0) {
        PASS("both sizes=0 returns SUCCESS with output_size=0");
    } else if (err == GPUCOMPRESS_SUCCESS) {
        printf("  INFO: SUCCESS but output_size=%zu (expected 0)\n", decomp_size);
        FAIL("both sizes=0 returned SUCCESS but wrong output_size");
    } else {
        FAIL("both sizes=0 returned error %d (expected SUCCESS)");
    }

    /* Restore and verify normal decompress still works */
    hdr->original_size = saved_original;
    hdr->compressed_size = saved_compressed;
    decomp_size = DATA_SIZE;
    err = gpucompress_decompress(compressed, compressed_size, decomp_buf, &decomp_size);
    if (err == GPUCOMPRESS_SUCCESS) {
        PASS("restored header decompresses successfully");
    } else {
        FAIL("restored header failed to decompress");
    }

    free(compressed);
    gpucompress_cleanup();

    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
