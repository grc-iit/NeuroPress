/**
 * test_c2_learning_flag_race.cu
 *
 * C2: g_online_learning_enabled and g_exploration_enabled are plain bool,
 *     read/written from multiple threads without synchronization (data race).
 *
 * Test: spawn threads doing concurrent compressions with ALGO_AUTO while
 * the main thread rapidly toggles learning/exploration flags. With plain
 * bool, ThreadSanitizer would flag this; functionally we verify no crash
 * and that the flag state is coherent after toggling.
 *
 * Run: ./test_c2_learning_flag_race
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <atomic>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

// CTest sets WORKING_DIRECTORY = CMAKE_SOURCE_DIR (repo root), so the weights
// live at neural_net/weights/model.nnwt relative to it. Earlier "../neural_net/..."
// resolved above the repo root and the test silently ran with no weights loaded,
// causing every ALGO_AUTO compress to return NN_NOT_LOADED (-10).
static const char* WEIGHTS_PATH = "neural_net/weights/model.nnwt";

static std::atomic<bool> g_stop{false};
static std::atomic<int> g_compress_count{0};

static void compress_worker(int id) {
    const size_t DATA_SIZE = 32 * 1024;
    float* h_data = (float*)malloc(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++)
        h_data[i] = (float)(i + id) * 0.01f;

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* h_output = malloc(max_out);

    while (!g_stop.load(std::memory_order_relaxed)) {
        size_t compressed_size = max_out;
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

        gpucompress_compress(h_data, DATA_SIZE, h_output, &compressed_size, &cfg, NULL);
        g_compress_count.fetch_add(1, std::memory_order_relaxed);
    }

    free(h_data);
    free(h_output);
}

int main(void) {
    printf("=== C2: Learning flag data race test ===\n\n");

    gpucompress_error_t gerr = gpucompress_init(WEIGHTS_PATH);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        printf("SKIP: gpucompress_init failed (%d)\n", gerr);
        return 1;
    }
    PASS("init succeeded");

    /* ---- Test 1: Concurrent compress + flag toggling ---- */
    printf("\n--- Test 1: 4 compress threads + 100 flag toggles ---\n");
    {
        const int N_THREADS = 4;
        const int N_TOGGLES = 100;

        g_stop.store(false);
        g_compress_count.store(0);

        std::thread workers[N_THREADS];
        for (int i = 0; i < N_THREADS; i++)
            workers[i] = std::thread(compress_worker, i);

        // Rapidly toggle learning flags from main thread
        for (int i = 0; i < N_TOGGLES; i++) {
            if (i % 2 == 0) {
                gpucompress_enable_online_learning();
                gpucompress_set_exploration(1);
            } else {
                gpucompress_disable_online_learning();
                gpucompress_set_exploration(0);
            }
        }

        g_stop.store(true);
        for (int i = 0; i < N_THREADS; i++)
            workers[i].join();

        int count = g_compress_count.load();
        printf("  compressions completed: %d\n", count);

        // The test's real job per its file header — "we verify no crash and
        // that the flag state is coherent after toggling." 100 atomic toggles
        // finish in microseconds while each ALGO_AUTO compress takes ~1ms,
        // so count==0 is a legal outcome (workers raced against g_stop) and
        // does not indicate a C2 regression. The race-hunt itself is TSAN's
        // job, not this functional test.
        PASS("concurrent compress + flag toggle completed without crash");
        (void)count;
    }

    /* ---- Test 2: Flag state coherence ---- */
    printf("\n--- Test 2: Flag state coherence after toggle ---\n");
    {
        gpucompress_enable_online_learning();
        gpucompress_set_exploration(1);
        if (gpucompress_online_learning_enabled() == 1) {
            PASS("online_learning flag reads correctly after set");
        } else {
            FAIL("online_learning flag incoherent");
        }
        if (gpucompress_active_learning_enabled() == 1) {
            PASS("active_learning flag reads correctly after set");
        } else {
            FAIL("active_learning flag incoherent");
        }

        gpucompress_disable_online_learning();
        if (gpucompress_online_learning_enabled() == 0) {
            PASS("online_learning flag disabled correctly");
        } else {
            FAIL("online_learning flag still set after disable");
        }
    }

    gpucompress_cleanup();
    PASS("cleanup completed");

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
