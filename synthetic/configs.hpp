#pragma once
#include <string>
#include <vector>

// ============================================================
// Compression configuration space (64 combos)
// 8 algorithms x 2 shuffle options x 4 quantization levels
// Matches neural_net/core/configs.py training space
// ============================================================

struct BenchConfig {
    std::string algo_name;
    int         algo_id;       // 1-8, matching gpucompress_algorithm_t
    int         shuffle_bytes; // 0 or 4
    unsigned    preprocessing; // bitmask: 0x02=shuffle4, 0x10=quantize
    bool        quantized;
    double      error_bound;   // 0.0 = lossless
};

inline std::vector<BenchConfig> make_bench_configs() {
    static const char*   ALGO_NAMES[]     = {"lz4","snappy","deflate","gdeflate",
                                              "zstd","ans","cascaded","bitcomp"};
    static const int     ALGO_IDS[]       = {1,2,3,4,5,6,7,8};
    static const int     SHUFFLE[]        = {0, 4};
    static const unsigned SHUFFLE_FLAGS[] = {0x00u, 0x02u};
    static const double  ERR_BOUNDS[]     = {0.0, 0.1, 0.01, 0.001};
    static const unsigned PREPROC_QUANT   = 0x10u;

    std::vector<BenchConfig> out;
    out.reserve(8 * 2 * 4);
    for (int a = 0; a < 8; ++a) {
        for (int s = 0; s < 2; ++s) {
            for (int q = 0; q < 4; ++q) {
                BenchConfig c;
                c.algo_name    = ALGO_NAMES[a];
                c.algo_id      = ALGO_IDS[a];
                c.shuffle_bytes= SHUFFLE[s];
                c.preprocessing= SHUFFLE_FLAGS[s];
                c.quantized    = (ERR_BOUNDS[q] > 0.0);
                c.error_bound  = ERR_BOUNDS[q];
                if (c.quantized) c.preprocessing |= PREPROC_QUANT;
                out.push_back(c);
            }
        }
    }
    return out;
}
