/**
 * @file heuristic.cu
 * @brief Entropy-based heuristic algorithm selector.
 *
 * A simple rule-based baseline for comparison against the NN selector.
 * Uses only the byte-level Shannon entropy (already computed by the stats
 * kernel) to pick a compression algorithm.
 *
 * Action encoding (same as NN):
 *   action = algo_index + quant*8 + shuffle*16
 *   algo_index: 0=lz4, 1=snappy, 2=deflate, 3=gdeflate, 4=zstd,
 *               5=ans, 6=cascaded, 7=bitcomp
 *
 * Heuristic rules:
 *   entropy < 3.5  →  zstd (index 4)       — low entropy, invest in ratio
 *   3.5 <= e < 5.5 →  gdeflate (index 3)   — medium entropy, GPU-optimized balance
 *   entropy >= 5.5 →  lz4 (index 0)        — high entropy, just be fast
 *
 * No preprocessing (shuffle/quantization) is applied — this is intentional.
 * The NN's advantage is partly in knowing when shuffle helps.
 */

#include "selection/heuristic.h"

int heuristic_select_action(double entropy)
{
    /* algo_index only, no shuffle or quantization */
    if (entropy < 3.5)
        return 4;   /* zstd */
    else if (entropy < 5.5)
        return 3;   /* gdeflate */
    else
        return 0;   /* lz4 */
}
