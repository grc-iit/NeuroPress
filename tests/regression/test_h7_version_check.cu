/**
 * test_h7_version_check.cu
 *
 * H7: Host decompress path checks only header.magic, not version.
 *     A header with valid magic but invalid version (e.g., 255) is
 *     accepted, risking misinterpretation of future/corrupt formats.
 *
 * Test strategy:
 *   1. Compress valid data to get a legitimate compressed buffer
 *   2. Corrupt the version field in the header to an invalid value
 *   3. Attempt decompression — should return GPUCOMPRESS_ERROR_INVALID_HEADER
 *   4. Before fix: version not checked, decompression proceeds (may crash or
 *      produce garbage)
 *   5. After fix: isValid() catches bad version, returns error
 *
 * Run: ./test_h7_version_check
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
    printf("=== H7: Host decompress skips version check ===\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* Step 1: Compress some valid data */
    const size_t DATA_SIZE = 4096;
    float h_data[DATA_SIZE / sizeof(float)];
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++)
        h_data[i] = (float)i * 0.1f;

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* compressed = malloc(max_out);
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
    printf("  Compressed %zu -> %zu bytes\n", DATA_SIZE, compressed_size);

    /* Step 2: Verify valid data decompresses successfully */
    float h_decomp[DATA_SIZE / sizeof(float)];
    size_t decomp_size = DATA_SIZE;
    err = gpucompress_decompress(compressed, compressed_size, h_decomp, &decomp_size);
    if (err == GPUCOMPRESS_SUCCESS) {
        PASS("valid compressed data decompresses successfully");
    } else {
        FAIL("valid compressed data failed to decompress");
    }

    /* Step 3: Corrupt version field to 255 (invalid) */
    CompressionHeader* hdr = (CompressionHeader*)compressed;
    printf("  Original header: magic=0x%08X version=%u\n", hdr->magic, hdr->version);

    uint8_t saved_version = hdr->version;
    hdr->version = 255;
    printf("  Corrupted header: magic=0x%08X version=%u\n", hdr->magic, hdr->version);

    /* Step 4: Attempt decompression with bad version */
    decomp_size = DATA_SIZE;
    err = gpucompress_decompress(compressed, compressed_size, h_decomp, &decomp_size);
    if (err == GPUCOMPRESS_ERROR_INVALID_HEADER) {
        PASS("bad version rejected with INVALID_HEADER error");
    } else {
        FAIL("bad version NOT rejected (decompress returned %d instead of INVALID_HEADER)");
    }

    /* Step 5: Also test version=0 (below valid range) */
    hdr->version = 0;
    decomp_size = DATA_SIZE;
    err = gpucompress_decompress(compressed, compressed_size, h_decomp, &decomp_size);
    if (err == GPUCOMPRESS_ERROR_INVALID_HEADER) {
        PASS("version=0 rejected with INVALID_HEADER error");
    } else {
        FAIL("version=0 NOT rejected (decompress returned %d instead of INVALID_HEADER)");
    }

    /* Restore and verify it still works */
    hdr->version = saved_version;
    decomp_size = DATA_SIZE;
    err = gpucompress_decompress(compressed, compressed_size, h_decomp, &decomp_size);
    if (err == GPUCOMPRESS_SUCCESS) {
        PASS("restored version decompresses successfully");
    } else {
        FAIL("restored version failed to decompress");
    }

    free(compressed);
    gpucompress_cleanup();

    printf("\n%d pass, %d fail\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
