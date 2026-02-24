/**
 * @file test_compression_core.cu
 * @brief Tests for src/compression/ fixes:
 *   1. CompressionHeader division-by-zero guard
 *   2. CompressionHeader validation and round-trip through device memory
 *   3. writeHeaderToDevice / readHeaderFromDevice error reporting
 *   4. DeviceChunkArrays RAII (no double-free)
 *   5. createDeviceChunkArrays stream-based sync
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include "compression/compression_header.h"
#include "compression/util.h"

static int g_pass = 0;
static int g_fail = 0;

#define TEST(name) \
    do { printf("  [TEST] %s ... ", name); } while (0)

#define PASS() \
    do { printf("PASS\n"); g_pass++; } while (0)

#define FAIL(msg) \
    do { printf("FAIL: %s\n", msg); g_fail++; } while (0)

#define ASSERT(cond, msg) \
    do { if (!(cond)) { FAIL(msg); return; } } while (0)

/* ============================================================
 * Test 1: CompressionHeader defaults and validation
 * ============================================================ */
static void test_header_defaults() {
    TEST("Header default construction");
    CompressionHeader h;
    ASSERT(h.magic == COMPRESSION_MAGIC, "magic mismatch");
    ASSERT(h.version == COMPRESSION_HEADER_VERSION, "version mismatch");
    ASSERT(h.isValid(), "default header should be valid");
    ASSERT(!h.hasShuffleApplied(), "default should have no shuffle");
    ASSERT(!h.hasQuantizationApplied(), "default should have no quantization");
    ASSERT(h.original_size == 0, "original_size should be 0");
    ASSERT(h.compressed_size == 0, "compressed_size should be 0");
    PASS();
}

/* ============================================================
 * Test 2: print() with compressed_size == 0 (division-by-zero fix)
 * ============================================================ */
static void test_header_print_zero_compressed() {
    TEST("Header print with compressed_size=0 (div-by-zero guard)");
    CompressionHeader h;
    h.original_size = 1024;
    h.compressed_size = 0;
    // Should not crash — this was the bug
    h.print();
    PASS();
}

/* ============================================================
 * Test 3: Quantization flags round-trip
 * ============================================================ */
static void test_header_quant_flags() {
    TEST("Header quantization flags round-trip");
    CompressionHeader h;
    h.setQuantizationFlags(1, 16, true);
    ASSERT(h.hasQuantizationApplied(), "quantization should be enabled");
    ASSERT(h.getQuantizationType() == 1, "type should be LINEAR (1)");
    ASSERT(h.getQuantizationPrecision() == 16, "precision should be 16");

    h.setQuantizationFlags(0, 8, false);
    ASSERT(!h.hasQuantizationApplied(), "quantization should be disabled");
    ASSERT(h.getQuantizationPrecision() == 8, "precision should be 8");
    PASS();
}

/* ============================================================
 * Test 4: Header GPU round-trip (write to device, read back)
 * ============================================================ */
static void test_header_device_roundtrip() {
    TEST("Header device write/read round-trip");

    CompressionHeader original;
    original.shuffle_element_size = 4;
    original.original_size = 1048576;
    original.compressed_size = 524288;
    original.setQuantizationFlags(1, 16, true);
    original.quant_error_bound = 0.001;
    original.quant_scale = 500.0;
    original.data_min = -1.5;
    original.data_max = 99.5;

    void* d_buf = nullptr;
    cudaError_t err = cudaMalloc(&d_buf, sizeof(CompressionHeader));
    ASSERT(err == cudaSuccess, "cudaMalloc failed");

    err = writeHeaderToDevice(d_buf, original);
    ASSERT(err == cudaSuccess, "writeHeaderToDevice failed");

    CompressionHeader readback;
    err = readHeaderFromDevice(d_buf, readback);
    ASSERT(err == cudaSuccess, "readHeaderFromDevice failed");

    cudaFree(d_buf);

    ASSERT(readback.magic == original.magic, "magic mismatch");
    ASSERT(readback.version == original.version, "version mismatch");
    ASSERT(readback.shuffle_element_size == 4, "shuffle mismatch");
    ASSERT(readback.original_size == 1048576, "original_size mismatch");
    ASSERT(readback.compressed_size == 524288, "compressed_size mismatch");
    ASSERT(readback.hasQuantizationApplied(), "quant flag mismatch");
    ASSERT(readback.getQuantizationPrecision() == 16, "quant precision mismatch");
    ASSERT(readback.quant_error_bound == 0.001, "error_bound mismatch");
    ASSERT(readback.data_min == -1.5, "data_min mismatch");
    ASSERT(readback.data_max == 99.5, "data_max mismatch");
    PASS();
}

/* ============================================================
 * Test 5: writeHeaderToDevice with nullptr (error return check)
 * ============================================================ */
static void test_header_write_null() {
    TEST("writeHeaderToDevice returns error on nullptr");
    CompressionHeader h;
    cudaError_t err = writeHeaderToDevice(nullptr, h);
    // Writing to nullptr should fail
    ASSERT(err != cudaSuccess, "expected error on nullptr write");
    // Clear the error so it doesn't pollute later tests
    cudaGetLastError();
    PASS();
}

/* ============================================================
 * Test 6: DeviceChunkArrays RAII — move semantics, no double-free
 * ============================================================ */
static void test_chunk_arrays_raii() {
    TEST("DeviceChunkArrays move semantics (no double-free, 16 MB)");

    // 16 MB buffer with 1 MB chunks = 16 chunks
    const size_t total = 16 * 1024 * 1024;
    const size_t chunk = 1024 * 1024;

    void* d_in = nullptr;
    void* d_out = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_in, total);
    ASSERT(err == cudaSuccess, "cudaMalloc d_in failed");
    err = cudaMalloc(&d_out, total);
    ASSERT(err == cudaSuccess, "cudaMalloc d_out failed");

    {
        DeviceChunkArrays a = createDeviceChunkArrays(d_in, d_out, total, chunk);
        ASSERT(a.num_chunks == 16, "expected 16 chunks");
        ASSERT(a.d_input_ptrs != nullptr, "d_input_ptrs null");

        // Move construct
        DeviceChunkArrays b(std::move(a));
        ASSERT(a.d_input_ptrs == nullptr, "source should be null after move");
        ASSERT(b.num_chunks == 16, "moved-to should have 16 chunks");

        // Move assign
        DeviceChunkArrays c;
        c = std::move(b);
        ASSERT(b.d_input_ptrs == nullptr, "source should be null after move-assign");
        ASSERT(c.num_chunks == 16, "move-assigned should have 16 chunks");

        // c's destructor should free without double-free
    }

    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

/* ============================================================
 * Test 7: createDeviceChunkArrays correctness — verify chunk pointers
 * ============================================================ */
static void test_chunk_arrays_correctness() {
    TEST("createDeviceChunkArrays correct chunk layout (10 MB, 1 MB chunks)");

    // 10 MB with 1 MB chunks = 10 exact chunks
    const size_t chunk = 1024 * 1024;
    const size_t total = 10 * chunk;

    void* d_in = nullptr;
    void* d_out = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_in, total);
    ASSERT(err == cudaSuccess, "cudaMalloc d_in failed");
    err = cudaMalloc(&d_out, total);
    ASSERT(err == cudaSuccess, "cudaMalloc d_out failed");

    DeviceChunkArrays arrays = createDeviceChunkArrays(d_in, d_out, total, chunk);
    ASSERT(arrays.num_chunks == 10, "expected 10 chunks");

    // Copy all chunk sizes back to host to verify
    size_t* h_sizes = (size_t*)malloc(10 * sizeof(size_t));
    err = cudaMemcpy(h_sizes, arrays.d_sizes, 10 * sizeof(size_t), cudaMemcpyDeviceToHost);
    ASSERT(err == cudaSuccess, "cudaMemcpy sizes failed");
    bool sizes_ok = true;
    for (int i = 0; i < 10; i++) {
        if (h_sizes[i] != chunk) { sizes_ok = false; break; }
    }
    ASSERT(sizes_ok, "all chunks should be 1 MB");

    // Verify pointer offsets
    uint8_t** h_ptrs = (uint8_t**)malloc(10 * sizeof(uint8_t*));
    err = cudaMemcpy(h_ptrs, arrays.d_input_ptrs, 10 * sizeof(uint8_t*), cudaMemcpyDeviceToHost);
    ASSERT(err == cudaSuccess, "cudaMemcpy ptrs failed");
    uint8_t* base = static_cast<uint8_t*>(d_in);
    bool ptrs_ok = true;
    for (int i = 0; i < 10; i++) {
        if (h_ptrs[i] != base + i * chunk) { ptrs_ok = false; break; }
    }
    ASSERT(ptrs_ok, "pointer offsets should be multiples of chunk size");

    free(h_sizes);
    free(h_ptrs);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

/* ============================================================
 * Test 7b: Chunk arrays with remainder (10 MB + 500 KB)
 * ============================================================ */
static void test_chunk_arrays_remainder() {
    TEST("createDeviceChunkArrays with remainder (10.5 MB, 1 MB chunks)");

    const size_t chunk = 1024 * 1024;
    const size_t total = 10 * chunk + 512 * 1024;  // 10.5 MB
    const size_t expected_chunks = 11;
    const size_t remainder = total - 10 * chunk;  // 512 KB

    void* d_in = nullptr;
    void* d_out = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_in, total);
    ASSERT(err == cudaSuccess, "cudaMalloc d_in failed");
    err = cudaMalloc(&d_out, total);
    ASSERT(err == cudaSuccess, "cudaMalloc d_out failed");

    DeviceChunkArrays arrays = createDeviceChunkArrays(d_in, d_out, total, chunk);
    ASSERT(arrays.num_chunks == expected_chunks, "expected 11 chunks");

    size_t* h_sizes = (size_t*)malloc(expected_chunks * sizeof(size_t));
    err = cudaMemcpy(h_sizes, arrays.d_sizes, expected_chunks * sizeof(size_t), cudaMemcpyDeviceToHost);
    ASSERT(err == cudaSuccess, "cudaMemcpy sizes failed");

    bool full_ok = true;
    for (size_t i = 0; i < expected_chunks - 1; i++) {
        if (h_sizes[i] != chunk) { full_ok = false; break; }
    }
    ASSERT(full_ok, "first 10 chunks should be 1 MB each");
    ASSERT(h_sizes[expected_chunks - 1] == remainder, "last chunk should be 512 KB");

    free(h_sizes);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

/* ============================================================
 * Test 8: createDeviceChunkArrays with stream
 * ============================================================ */
static void test_chunk_arrays_with_stream() {
    TEST("createDeviceChunkArrays with non-default stream (8 MB)");

    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    ASSERT(err == cudaSuccess, "cudaStreamCreate failed");

    const size_t chunk = 1024 * 1024;
    const size_t total = 8 * chunk;  // 8 MB

    void* d_in = nullptr;
    void* d_out = nullptr;
    err = cudaMalloc(&d_in, total);
    ASSERT(err == cudaSuccess, "cudaMalloc d_in failed");
    err = cudaMalloc(&d_out, total);
    ASSERT(err == cudaSuccess, "cudaMalloc d_out failed");

    DeviceChunkArrays arrays = createDeviceChunkArrays(d_in, d_out, total, chunk, stream);
    ASSERT(arrays.num_chunks == 8, "expected 8 chunks");

    size_t h_sizes[8];
    err = cudaMemcpy(h_sizes, arrays.d_sizes, 8 * sizeof(size_t), cudaMemcpyDeviceToHost);
    ASSERT(err == cudaSuccess, "cudaMemcpy failed");
    bool ok = true;
    for (int i = 0; i < 8; i++) {
        if (h_sizes[i] != chunk) { ok = false; break; }
    }
    ASSERT(ok, "all 8 chunks should be 1 MB");

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    PASS();
}

/* ============================================================
 * Test 9: createDeviceChunkArrays edge cases
 * ============================================================ */
static void test_chunk_arrays_edge_cases() {
    TEST("createDeviceChunkArrays edge cases");

    void* d_in = nullptr;
    void* d_out = nullptr;
    cudaMalloc(&d_in, 256);
    cudaMalloc(&d_out, 256);

    // total_bytes == 0 → empty result
    DeviceChunkArrays empty = createDeviceChunkArrays(d_in, d_out, 0, 1024);
    ASSERT(empty.num_chunks == 0, "zero bytes should give 0 chunks");

    // null pointers → exception
    bool caught = false;
    try {
        createDeviceChunkArrays(nullptr, d_out, 256, 64);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    ASSERT(caught, "null input should throw invalid_argument");

    caught = false;
    try {
        createDeviceChunkArrays(d_in, d_out, 256, 0);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    ASSERT(caught, "chunk_bytes=0 should throw invalid_argument");

    // Exact multiple — no remainder chunk
    DeviceChunkArrays exact = createDeviceChunkArrays(d_in, d_out, 256, 128);
    ASSERT(exact.num_chunks == 2, "256/128 should give exactly 2 chunks");

    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== test_compression_core ===\n");

    // Check for GPU
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found — skipping GPU tests\n");
        return 1;
    }

    test_header_defaults();
    test_header_print_zero_compressed();
    test_header_quant_flags();
    test_header_device_roundtrip();
    test_header_write_null();
    test_chunk_arrays_raii();
    test_chunk_arrays_correctness();
    test_chunk_arrays_remainder();
    test_chunk_arrays_with_stream();
    test_chunk_arrays_edge_cases();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
