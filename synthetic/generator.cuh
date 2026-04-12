#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

// ============================================================
// GPU Synthetic Data Generator
//
// Palette-based 32-bin system matching syntheticGeneration/generator.py
// Chunks are generated entirely on the GPU using cuRAND.
//
// Design:
//   Phase 1 (CPU): compute palette CDF and bin layouts (fast, N_BINS=32)
//   Phase 2 (GPU): Markov-chain kernel fills values using cuRAND
//                  Each thread handles a tile of consecutive elements.
//                  Spatial locality is controlled by `perturbation`:
//                    0.0 = all elements in same bin (long runs)
//                    1.0 = each element independently samples a bin (random)
// ============================================================

static const int N_BINS = 32;

enum Palette {
    PAL_UNIFORM     = 0,
    PAL_NORMAL      = 1,
    PAL_GAMMA       = 2,
    PAL_EXPONENTIAL = 3,
    PAL_BIMODAL     = 4,
    PAL_GRAYSCOTT   = 5,
    PAL_HIGH_ENTROPY= 6,
    PAL_COUNT       = 7
};

enum FillMode {
    FILL_CONSTANT   = 0,
    FILL_LINEAR     = 1,
    FILL_QUADRATIC  = 2,
    FILL_SINUSOIDAL = 3,
    FILL_RANDOM     = 4,
    FILL_COUNT      = 5
};

struct ChunkParams {
    Palette  palette;
    float    bin_width;    // absolute bin width in [0,1] value space
    float    perturbation; // 0.0 = long runs, 1.0 = fully random
    FillMode fill_mode;
    size_t   num_elements; // number of float32 elements
};

// Sampling tables matching Python TRAINING_* constants
static const float  TRAINING_BIN_WIDTHS[]    = {0.1f, 0.12f, 0.15f, 0.25f, 0.5f, 1.0f, 16.0f};
static const float  TRAINING_PERTURBATIONS[] = {0.0f, 0.1f, 0.2f, 0.325f, 0.5f, 0.75f, 0.95f, 1.0f};
static const size_t TRAINING_SIZES[]         = {64*1024/4, 256*1024/4, 1*1024*1024/4, 4*1024*1024/4};
// sizes in elements (float32 = 4 bytes): 64KB=16384, 256KB=65536, 1MB=262144, 4MB=1048576

static const int N_BIN_WIDTHS    = 7;
static const int N_PERTURBATIONS = 8;
static const int N_SIZES         = 4;

// Return human-readable name strings
const char* palette_name(Palette p);
const char* fillmode_name(FillMode f);

// ============================================================
// Main entry point:
//   Generates num_elements float32 values on the GPU.
//   Returns device pointer -- caller must cudaFree().
// ============================================================
float* generate_chunk_gpu(const ChunkParams& params, uint64_t seed,
                           cudaStream_t stream = nullptr);
