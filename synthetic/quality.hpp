#pragma once
#include <cmath>
#include <cstring>
#include <limits>

// ============================================================
// Quality metrics (CPU-side, after decompressed data download)
// Matches compute_quality() in scripts/benchmark.cpp
// ============================================================

struct QualityMetrics {
    double psnr;          // dB, inf = lossless
    double ssim;          // [-1, 1], 1.0 = perfect
    double max_abs_error; // maximum point-wise absolute error
    double rmse;          // root mean squared error
    double mean_abs_err;  // mean absolute error
    double bit_rate;      // bits per element
};

inline QualityMetrics compute_quality(const float* orig, const float* decomp,
                                      size_t n, size_t compressed_bytes)
{
    QualityMetrics qm{};
    qm.psnr         = std::numeric_limits<double>::infinity();
    qm.ssim         = 1.0;
    qm.max_abs_error= 0.0;
    qm.rmse         = 0.0;
    qm.mean_abs_err = 0.0;
    qm.bit_rate     = (n > 0) ? static_cast<double>(compressed_bytes) * 8.0 / n : 32.0;

    if (n == 0) return qm;

    double mse       = 0.0;
    double max_err   = 0.0;
    double sum_abs   = 0.0;
    double vmin      = orig[0];
    double vmax      = orig[0];
    double sum_x=0, sum_y=0, sum_x2=0, sum_y2=0, sum_xy=0;

    for (size_t i = 0; i < n; ++i) {
        double o = orig[i];
        double d = decomp[i];
        double diff = o - d;
        mse     += diff * diff;
        sum_abs += std::abs(diff);
        double adiff = std::abs(diff);
        if (adiff > max_err) max_err = adiff;
        if (o < vmin) vmin = o;
        if (o > vmax) vmax = o;
        sum_x  += o;   sum_y  += d;
        sum_x2 += o*o; sum_y2 += d*d;
        sum_xy += o*d;
    }

    double nd  = static_cast<double>(n);
    mse       /= nd;
    qm.max_abs_error = max_err;
    qm.rmse          = std::sqrt(mse);
    qm.mean_abs_err  = sum_abs / nd;

    // Global single-window SSIM approximation
    double mu_x  = sum_x / nd,   mu_y  = sum_y / nd;
    double var_x = sum_x2 / nd - mu_x * mu_x;
    double var_y = sum_y2 / nd - mu_y * mu_y;
    double cov   = sum_xy / nd - mu_x * mu_y;
    double L     = (vmax > vmin) ? (vmax - vmin) : 1.0;
    double c1    = (0.01 * L) * (0.01 * L);
    double c2    = (0.03 * L) * (0.03 * L);
    qm.ssim = ((2.0*mu_x*mu_y + c1) * (2.0*cov + c2))
            / ((mu_x*mu_x + mu_y*mu_y + c1) * (var_x + var_y + c2));

    if (mse == 0.0 || vmax == vmin) {
        qm.psnr = std::numeric_limits<double>::infinity();
        return qm;
    }
    double data_range = vmax - vmin;
    qm.psnr = 10.0 * std::log10((data_range * data_range) / mse);
    return qm;
}

// Lossless-perfect metrics (used when memcmp confirms exact match)
inline QualityMetrics perfect_quality(size_t n, size_t compressed_bytes) {
    QualityMetrics qm{};
    qm.psnr         = std::numeric_limits<double>::infinity();
    qm.ssim         = 1.0;
    qm.max_abs_error= 0.0;
    qm.rmse         = 0.0;
    qm.mean_abs_err = 0.0;
    qm.bit_rate     = (n > 0) ? static_cast<double>(compressed_bytes) * 8.0 / n : 32.0;
    return qm;
}
