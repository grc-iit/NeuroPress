/**
 * @file qtable_gpu.cu
 * @brief GPU Q-Table for RL-based compression algorithm selection
 *
 * Stores Q-Table in CUDA global device memory for inference.
 * Supports loading from binary or JSON format.
 *
 * Q-Table structure:
 *   States: 1024 (16 entropy x 4 error x 4 MAD x 4 derivative)
 *   Actions: 32 (2 quantization x 2 shuffle x 8 algorithms)
 *   Total: 32768 float values (~128KB)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace gpucompress {

/* ============================================================
 * Constants
 * ============================================================ */

/** Number of entropy bins (0.5-width bins, byte entropy range [0, 8)) */
constexpr int NUM_ENTROPY_BINS = 16;

/** Number of error bound levels (aggressive, balanced, precise, lossless) */
constexpr int NUM_ERROR_LEVELS = 4;

/** Number of MAD bins */
constexpr int NUM_MAD_BINS = 4;

/** Number of first derivative bins */
constexpr int NUM_DERIV_BINS = 4;

/** Total number of states */
constexpr int NUM_STATES = NUM_ENTROPY_BINS * NUM_ERROR_LEVELS * NUM_MAD_BINS * NUM_DERIV_BINS;

/** Total number of actions */
constexpr int NUM_ACTIONS = 32;

/** Total Q-Table size */
constexpr int QTABLE_SIZE = NUM_STATES * NUM_ACTIONS;

/* ============================================================
 * Q-Table in Global Device Memory
 * ============================================================ */

/**
 * Q-Table stored in global device memory.
 * Layout: q_values[state * NUM_ACTIONS + action]
 *
 * At 128KB, the table exceeds the 64KB CUDA constant memory limit,
 * so we use cudaMalloc'd global memory instead. Performance impact
 * is negligible since the lookup happens once per file, not per element.
 */
static float* d_qtable = nullptr;

/**
 * Flag indicating if Q-Table is loaded
 */
static bool g_qtable_loaded = false;

/**
 * Host copy of Q-Table for CPU access (heap-allocated, 128KB too large for stack)
 */
static float* g_qtable_host = nullptr;

/* ============================================================
 * Q-Table Loading
 * ============================================================ */

/**
 * Load Q-Table to GPU global memory.
 *
 * @param h_qtable Host array of Q-Table values (NUM_STATES * NUM_ACTIONS floats)
 * @return CUDA error code
 */
cudaError_t loadQTableToGPU(const float* h_qtable) {
    if (h_qtable == nullptr) {
        return cudaErrorInvalidValue;
    }

    // Allocate device memory if not yet allocated
    if (d_qtable == nullptr) {
        cudaError_t err = cudaMalloc(&d_qtable, QTABLE_SIZE * sizeof(float));
        if (err != cudaSuccess) {
            return err;
        }
    }

    // Copy to global device memory
    cudaError_t err = cudaMemcpy(
        d_qtable,
        h_qtable,
        QTABLE_SIZE * sizeof(float),
        cudaMemcpyHostToDevice
    );

    if (err == cudaSuccess) {
        // Allocate host copy if not yet allocated
        if (g_qtable_host == nullptr) {
            g_qtable_host = new float[QTABLE_SIZE]();
        }
        memcpy(g_qtable_host, h_qtable, QTABLE_SIZE * sizeof(float));
        g_qtable_loaded = true;
    }

    return err;
}

/**
 * Free GPU Q-Table memory.
 */
void cleanupQTable() {
    if (d_qtable != nullptr) {
        cudaFree(d_qtable);
        d_qtable = nullptr;
    }
    if (g_qtable_host != nullptr) {
        delete[] g_qtable_host;
        g_qtable_host = nullptr;
    }
    g_qtable_loaded = false;
}

/**
 * Load Q-Table from binary file.
 *
 * Binary format:
 *   - Header: 4 bytes magic (0x51544142 = "QTAB")
 *   - Header: 4 bytes version (1)
 *   - Header: 4 bytes num_states
 *   - Header: 4 bytes num_actions
 *   - Data: num_states * num_actions * 4 bytes (floats)
 *
 * @param filepath Path to binary Q-Table file
 * @return true on success
 */
bool loadQTableFromBinary(const char* filepath) {
    if (filepath == nullptr) {
        return false;
    }

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open Q-Table file: %s\n", filepath);
        return false;
    }

    // Read and validate header
    uint32_t magic, version, num_states, num_actions;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&num_states), 4);
    file.read(reinterpret_cast<char*>(&num_actions), 4);

    if (magic != 0x51544142) {  // "QTAB"
        fprintf(stderr, "Invalid Q-Table magic number\n");
        return false;
    }

    if (version != 1 && version != 2) {
        fprintf(stderr, "Unsupported Q-Table version: %u\n", version);
        return false;
    }

    if (num_states != NUM_STATES || num_actions != NUM_ACTIONS) {
        fprintf(stderr, "Q-Table size mismatch: expected %dx%d, got %ux%u\n",
                NUM_STATES, NUM_ACTIONS, num_states, num_actions);
        return false;
    }

    // Read Q-values
    std::vector<float> values(QTABLE_SIZE);
    file.read(reinterpret_cast<char*>(values.data()), QTABLE_SIZE * sizeof(float));

    if (!file) {
        fprintf(stderr, "Failed to read Q-Table data\n");
        return false;
    }

    // Load to GPU
    cudaError_t err = loadQTableToGPU(values.data());
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to load Q-Table to GPU: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

/**
 * Load Q-Table from JSON file.
 *
 * JSON format:
 *   {
 *     "version": 1,
 *     "n_states": 30,
 *     "n_actions": 32,
 *     "q_values": [[...], [...], ...]  // 30 arrays of 32 floats
 *   }
 *
 * This is a simple parser - for production, use a proper JSON library.
 *
 * @param filepath Path to JSON Q-Table file
 * @return true on success
 */
bool loadQTableFromJSON(const char* filepath) {
    if (filepath == nullptr) {
        return false;
    }

    std::ifstream file(filepath);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open Q-Table JSON file: %s\n", filepath);
        return false;
    }

    // Read entire file
    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    // Simple parsing - look for q_values array
    std::vector<float> values;
    values.reserve(QTABLE_SIZE);

    // Find "q_values" key
    size_t pos = content.find("\"q_values\"");
    if (pos == std::string::npos) {
        fprintf(stderr, "q_values not found in JSON\n");
        return false;
    }

    // Parse numbers after [ characters
    bool in_array = false;
    std::string num_str;

    for (size_t i = pos; i < content.size() && values.size() < QTABLE_SIZE; i++) {
        char c = content[i];

        if (c == '[') {
            in_array = true;
            continue;
        }

        if (!in_array) continue;

        if (c == ']' && !num_str.empty()) {
            // End of array segment, parse accumulated number
            values.push_back(std::stof(num_str));
            num_str.clear();
        } else if (c == ',' && !num_str.empty()) {
            values.push_back(std::stof(num_str));
            num_str.clear();
        } else if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+') {
            num_str += c;
        }
    }

    if (values.size() != QTABLE_SIZE) {
        fprintf(stderr, "Q-Table JSON has wrong number of values: %zu (expected %d)\n",
                values.size(), QTABLE_SIZE);
        return false;
    }

    // Load to GPU
    cudaError_t err = loadQTableToGPU(values.data());
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to load Q-Table to GPU: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

/**
 * Load Q-Table from file (auto-detect format).
 *
 * @param filepath Path to Q-Table file (.json or .bin)
 * @return true on success
 */
bool loadQTable(const char* filepath) {
    if (filepath == nullptr) {
        return false;
    }

    std::string path(filepath);

    // Detect format from extension
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
        return loadQTableFromJSON(filepath);
    } else if (path.size() >= 4 && path.substr(path.size() - 4) == ".bin") {
        return loadQTableFromBinary(filepath);
    } else {
        // Try binary first, then JSON
        if (loadQTableFromBinary(filepath)) {
            return true;
        }
        return loadQTableFromJSON(filepath);
    }
}

/**
 * Initialize Q-Table with default values.
 *
 * Default: all zeros (no prior knowledge).
 * Used when no Q-Table file is provided.
 */
void initializeDefaultQTable() {
    // Heap-allocate default values (128KB too large for stack)
    std::vector<float> default_values(QTABLE_SIZE, 0.0f);

    // Could set some heuristic defaults here
    // For now, just use zeros which will cause random selection

    loadQTableToGPU(default_values.data());
}

/**
 * Check if Q-Table is loaded.
 */
bool isQTableLoaded() {
    return g_qtable_loaded;
}

/**
 * Get device pointer to Q-Table in GPU global memory.
 *
 * Used by the auto-stats GPU pipeline to pass the Q-Table
 * directly to the finalizeAndLookupKernel.
 *
 * @return Device pointer to Q-Table, or nullptr if not loaded
 */
const float* getQTableDevicePtr() {
    return d_qtable;
}

/* ============================================================
 * Q-Table Inference Kernel
 * ============================================================ */

/**
 * Kernel to find best action for a state.
 *
 * @param qtable      Q-Table in global device memory
 * @param state       State index
 * @param best_action Output: best action index
 */
__global__ void qtableArgmaxKernel(const float* qtable, int state, int* best_action) {
    __shared__ float s_values[NUM_ACTIONS];
    __shared__ int s_indices[NUM_ACTIONS];

    int tid = threadIdx.x;

    // Load Q-values for this state from global memory
    if (tid < NUM_ACTIONS) {
        s_values[tid] = qtable[state * NUM_ACTIONS + tid];
        s_indices[tid] = tid;
    }
    __syncthreads();

    // Parallel reduction to find maximum
    for (int s = NUM_ACTIONS / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < NUM_ACTIONS) {
            if (s_values[tid + s] > s_values[tid]) {
                s_values[tid] = s_values[tid + s];
                s_indices[tid] = s_indices[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *best_action = s_indices[0];
    }
}

/**
 * Get best action from Q-Table (GPU implementation).
 *
 * @param state  Q-Table state index
 * @param stream CUDA stream
 * @return Best action ID
 */
int getBestActionGPU(int state, cudaStream_t stream) {
    if (state < 0 || state >= NUM_STATES) {
        return 0;  // Default action (LZ4, no preprocessing)
    }

    if (!g_qtable_loaded) {
        return 0;  // Default action
    }

    int* d_best_action = nullptr;
    int h_best_action = 0;

    cudaError_t err = cudaMalloc(&d_best_action, sizeof(int));
    if (err != cudaSuccess) {
        return 0;
    }

    qtableArgmaxKernel<<<1, 32, 0, stream>>>(d_qtable, state, d_best_action);

    err = cudaMemcpyAsync(&h_best_action, d_best_action, sizeof(int),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_best_action);
        return 0;
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(d_best_action);

    if (err != cudaSuccess) {
        return 0;
    }

    return h_best_action;
}

/**
 * Get best action from Q-Table (CPU implementation using host copy).
 *
 * Faster than GPU for single query since no kernel launch overhead.
 *
 * @param state Q-Table state index
 * @return Best action ID
 */
int getBestActionCPU(int state) {
    if (state < 0 || state >= NUM_STATES || !g_qtable_loaded) {
        return 0;
    }

    int best_action = 0;
    float best_value = g_qtable_host[state * NUM_ACTIONS];

    for (int a = 1; a < NUM_ACTIONS; a++) {
        float value = g_qtable_host[state * NUM_ACTIONS + a];
        if (value > best_value) {
            best_value = value;
            best_action = a;
        }
    }

    return best_action;
}

/**
 * Get Q-value for a state-action pair.
 *
 * @param state  State index
 * @param action Action index
 * @return Q-value
 */
float getQValue(int state, int action) {
    if (state < 0 || state >= NUM_STATES ||
        action < 0 || action >= NUM_ACTIONS ||
        !g_qtable_loaded) {
        return 0.0f;
    }

    return g_qtable_host[state * NUM_ACTIONS + action];
}

} // namespace gpucompress

/* ============================================================
 * C API Wrappers
 * ============================================================ */

extern "C" {

/**
 * Load Q-Table from file.
 */
int gpucompress_qtable_load_impl(const char* filepath) {
    return gpucompress::loadQTable(filepath) ? 0 : -1;
}

/**
 * Check if Q-Table is loaded.
 */
int gpucompress_qtable_is_loaded_impl(void) {
    return gpucompress::isQTableLoaded() ? 1 : 0;
}

/**
 * Get best action for state.
 */
int gpucompress_qtable_get_best_action_impl(int state) {
    return gpucompress::getBestActionCPU(state);
}

/**
 * Initialize default Q-Table.
 */
void gpucompress_qtable_init_default_impl(void) {
    gpucompress::initializeDefaultQTable();
}

/**
 * Cleanup Q-Table GPU memory.
 */
void gpucompress_qtable_cleanup_impl(void) {
    gpucompress::cleanupQTable();
}

} // extern "C"
