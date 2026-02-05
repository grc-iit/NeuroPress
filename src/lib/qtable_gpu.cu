/**
 * @file qtable_gpu.cu
 * @brief GPU Q-Table for RL-based compression algorithm selection
 *
 * Stores Q-Table in CUDA constant memory for fast inference.
 * Supports loading from binary or JSON format.
 *
 * Q-Table structure:
 *   States: 30 (10 entropy bins x 3 error levels)
 *   Actions: 32 (2 quantization x 2 shuffle x 8 algorithms)
 *   Total: 960 float values (~4KB)
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

/** Number of entropy bins */
constexpr int NUM_ENTROPY_BINS = 10;

/** Number of error bound levels */
constexpr int NUM_ERROR_LEVELS = 3;

/** Total number of states */
constexpr int NUM_STATES = NUM_ENTROPY_BINS * NUM_ERROR_LEVELS;

/** Total number of actions */
constexpr int NUM_ACTIONS = 32;

/** Total Q-Table size */
constexpr int QTABLE_SIZE = NUM_STATES * NUM_ACTIONS;

/* ============================================================
 * Q-Table in Constant Memory
 * ============================================================ */

/**
 * Q-Table stored in constant memory for fast read access.
 * Layout: q_values[state * NUM_ACTIONS + action]
 */
__constant__ float d_qtable[QTABLE_SIZE];

/**
 * Flag indicating if Q-Table is loaded
 */
static bool g_qtable_loaded = false;

/**
 * Host copy of Q-Table for CPU access
 */
static float g_qtable_host[QTABLE_SIZE] = {0};

/* ============================================================
 * Q-Table Loading
 * ============================================================ */

/**
 * Load Q-Table to GPU constant memory.
 *
 * @param h_qtable Host array of Q-Table values (NUM_STATES * NUM_ACTIONS floats)
 * @return CUDA error code
 */
cudaError_t loadQTableToGPU(const float* h_qtable) {
    if (h_qtable == nullptr) {
        return cudaErrorInvalidValue;
    }

    // Copy to constant memory
    cudaError_t err = cudaMemcpyToSymbol(
        d_qtable,
        h_qtable,
        QTABLE_SIZE * sizeof(float),
        0,
        cudaMemcpyHostToDevice
    );

    if (err == cudaSuccess) {
        // Also keep host copy
        memcpy(g_qtable_host, h_qtable, QTABLE_SIZE * sizeof(float));
        g_qtable_loaded = true;
    }

    return err;
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

    if (version != 1) {
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
    float default_values[QTABLE_SIZE] = {0};

    // Could set some heuristic defaults here
    // For now, just use zeros which will cause random selection

    loadQTableToGPU(default_values);
}

/**
 * Check if Q-Table is loaded.
 */
bool isQTableLoaded() {
    return g_qtable_loaded;
}

/* ============================================================
 * Q-Table Inference Kernel
 * ============================================================ */

/**
 * Kernel to find best action for a state.
 *
 * @param state       State index
 * @param best_action Output: best action index
 */
__global__ void qtableArgmaxKernel(int state, int* best_action) {
    __shared__ float s_values[NUM_ACTIONS];
    __shared__ int s_indices[NUM_ACTIONS];

    int tid = threadIdx.x;

    // Load Q-values for this state
    if (tid < NUM_ACTIONS) {
        s_values[tid] = d_qtable[state * NUM_ACTIONS + tid];
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

    qtableArgmaxKernel<<<1, 32, 0, stream>>>(state, d_best_action);

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

} // extern "C"
