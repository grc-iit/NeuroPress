/**
 * test_bug5_truncated_nnwt.cu
 *
 * BUG-5: loadNNFromBinary() lacked per-read gcount() checks.
 *
 * Before the fix, file.read() was called for each weight array, but only a
 * single aggregate !file check existed at the end.  A file truncated in the
 * middle of e.g. w2 (the 64 KB layer-2 weight matrix) would:
 *   - Partially fill h_weights.w2 with 0s or garbage
 *   - NOT trigger the !file check until after all reads had "succeeded"
 *   - cudaMemcpy uploads the partially-initialised struct to GPU
 *   - NN silently produces garbage predictions
 *
 * Fix applied (src/nn/nn_gpu.cu):
 *   - Header: consolidated into one 24-byte read with gcount() == 24 check
 *   - Each weight array: NN_READ macro checks gcount() == expected_bytes,
 *     returns false immediately on mismatch
 *
 * Test cases (using gpucompress_load_nn() which returns error on failure):
 *   1. Zero-byte file          → header read returns 0 bytes → fail
 *   2. Truncated header (8 B)  → 8 < 24 → fail
 *   3. Bad magic number        → correct size, wrong magic → fail
 *   4. Wrong architecture      → correct magic/size, wrong hidden_dim → fail
 *   5. Truncated in weights    → full header OK, w2 cut in half → fail
 *   6. Truncated at w3 start   → only 4 bytes of w3 → fail
 *   7. Valid file              → gpucompress_load_nn() succeeds
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include "gpucompress.h"

static const uint32_t NN_MAGIC   = 0x4E4E5754u;
static const uint32_t NN_VERSION = 2u;
static const int      INPUT_DIM  = 15;
static const int      HIDDEN_DIM = 128;
static const int      OUTPUT_DIM = 4;

static bool write_valid_nnwt(const char *path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t hdr[6] = { NN_MAGIC, NN_VERSION, 3,
                        (uint32_t)INPUT_DIM, (uint32_t)HIDDEN_DIM,
                        (uint32_t)OUTPUT_DIM };
    f.write(reinterpret_cast<char*>(hdr), sizeof(hdr));

    auto write_zeros = [&](int n) {
        float v = 0.0f;
        for (int i = 0; i < n; i++) f.write(reinterpret_cast<char*>(&v), 4);
    };
    auto write_ones = [&](int n) {
        float v = 1.0f;
        for (int i = 0; i < n; i++) f.write(reinterpret_cast<char*>(&v), 4);
    };

    write_zeros(INPUT_DIM);
    write_ones(INPUT_DIM);    /* x_stds != 0 */
    write_zeros(OUTPUT_DIM);
    write_ones(OUTPUT_DIM);   /* y_stds != 0 */

    write_zeros(HIDDEN_DIM * INPUT_DIM);
    write_zeros(HIDDEN_DIM);
    write_zeros(HIDDEN_DIM * HIDDEN_DIM);
    write_zeros(HIDDEN_DIM);
    write_zeros(OUTPUT_DIM * HIDDEN_DIM);
    write_zeros(OUTPUT_DIM);

    write_zeros(INPUT_DIM);   /* x_mins */
    write_ones(INPUT_DIM);    /* x_maxs */

    return f.good();
}

static bool truncate_file(const char *src, const char *dst, size_t keep_bytes)
{
    std::ifstream in(src, std::ios::binary);
    if (!in) return false;
    std::ofstream out(dst, std::ios::binary);
    if (!out) return false;
    char buf[4096];
    size_t written = 0;
    while (written < keep_bytes) {
        size_t want = keep_bytes - written;
        if (want > sizeof(buf)) want = sizeof(buf);
        in.read(buf, (std::streamsize)want);
        size_t got = (size_t)in.gcount();
        if (got == 0) break;
        out.write(buf, (std::streamsize)got);
        written += got;
    }
    return true;
}

static int run_test(const char *label, const char *path, bool expect_success)
{
    /* gpucompress_load_nn properly returns error when file is bad */
    gpucompress_error_t err = gpucompress_load_nn(path);
    bool got_success = (err == GPUCOMPRESS_SUCCESS);
    bool ok = (got_success == expect_success);
    printf("  %-48s %s (expected %s, got %s)\n",
           label,
           ok ? "PASS" : "FAIL",
           expect_success ? "SUCCESS" : "FAILURE",
           got_success    ? "SUCCESS" : "FAILURE");
    return ok ? 1 : 0;
}

int main(void)
{
    printf("=== BUG-5: loadNNFromBinary — Truncated File Detection ===\n\n");

    /* Library must be initialized before gpucompress_load_nn can be called */
    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n");
        return 1;
    }

    const char *valid_path = "/tmp/test_bug5_valid.nnwt";
    const char *trunc_path = "/tmp/test_bug5_trunc.nnwt";
    const char *bad_path   = "/tmp/test_bug5_bad.nnwt";

    if (!write_valid_nnwt(valid_path)) {
        fprintf(stderr, "Failed to write reference .nnwt\n");
        gpucompress_cleanup();
        return 1;
    }

    /* Size breakdown for truncation points:
     * header:  24 B
     * norm:    (15+15+4+4)*4 = 152 B
     * w1+b1:   (128*15+128)*4 = 8192 B
     * w2+b2:   (128*128+128)*4 = 66048 B
     * w3+b3:   (4*128+4)*4 = 2064 B  */
    size_t header_sz = 24;
    size_t norm_sz   = (INPUT_DIM*2 + OUTPUT_DIM*2) * 4;
    size_t w1b1_sz   = (HIDDEN_DIM*INPUT_DIM + HIDDEN_DIM) * 4;
    size_t w2b2_sz   = (HIDDEN_DIM*HIDDEN_DIM + HIDDEN_DIM) * 4;

    int passed = 0, total = 0;

    /* Case 1: empty file */
    total++;
    { std::ofstream f("/tmp/test_bug5_empty.nnwt"); }
    passed += run_test("Case 1: empty file (0 bytes)",
                       "/tmp/test_bug5_empty.nnwt", false);

    /* Case 2: truncated header (only 8 bytes) */
    total++;
    truncate_file(valid_path, trunc_path, 8);
    passed += run_test("Case 2: truncated header (8/24 bytes)",
                       trunc_path, false);

    /* Case 3: bad magic number */
    total++;
    {
        std::ofstream f(bad_path, std::ios::binary);
        uint32_t hdr[6] = { 0xDEADBEEFu, NN_VERSION, 3,
                            (uint32_t)INPUT_DIM, (uint32_t)HIDDEN_DIM,
                            (uint32_t)OUTPUT_DIM };
        f.write(reinterpret_cast<char*>(hdr), sizeof(hdr));
        char zeros[4096] = {};
        for (int i = 0; i < 20; i++) f.write(zeros, sizeof(zeros));
    }
    passed += run_test("Case 3: bad magic (0xDEADBEEF)",
                       bad_path, false);

    /* Case 4: correct magic, wrong hidden_dim */
    total++;
    {
        std::ofstream f(bad_path, std::ios::binary);
        uint32_t hdr[6] = { NN_MAGIC, NN_VERSION, 3,
                            (uint32_t)INPUT_DIM, 64u, (uint32_t)OUTPUT_DIM };
        f.write(reinterpret_cast<char*>(hdr), sizeof(hdr));
        char zeros[4096] = {};
        for (int i = 0; i < 20; i++) f.write(zeros, sizeof(zeros));
    }
    passed += run_test("Case 4: wrong architecture (hidden_dim=64)",
                       bad_path, false);

    /* Case 5: truncated in the middle of w2 */
    total++;
    truncate_file(valid_path, trunc_path, header_sz + norm_sz + w1b1_sz + w2b2_sz / 2);
    passed += run_test("Case 5: truncated mid-w2 (half of 66 KB weights)",
                       trunc_path, false);

    /* Case 6: truncated just 4 bytes into w3 */
    total++;
    truncate_file(valid_path, trunc_path, header_sz + norm_sz + w1b1_sz + w2b2_sz + 4);
    passed += run_test("Case 6: truncated at w3 start (4 bytes of w3)",
                       trunc_path, false);

    /* Case 7: valid complete file */
    total++;
    passed += run_test("Case 7: valid complete .nnwt",
                       valid_path, true);

    remove("/tmp/test_bug5_empty.nnwt");
    remove(trunc_path);
    remove(bad_path);
    remove(valid_path);

    gpucompress_cleanup();

    printf("\n=== BUG-5 Result: %d/%d passed ===\n", passed, total);
    if (passed == total) {
        printf("VERDICT: All truncated/corrupt files correctly rejected.\n");
        printf("         gcount() checks in loadNNFromBinary are working.\n");
        return 0;
    }
    printf("VERDICT: FAILURES — truncated files silently accepted.\n");
    return 1;
}
