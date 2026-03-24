/**
 * C-linkage wrapper for the Kendall tau profiler.
 * VPIC's deck is .cxx (compiled by host compiler), so CUDA-dependent
 * profiler code lives here and is linked separately.
 */
#include "../kendall_tau_profiler.cuh"

extern "C" {

int vpic_run_ranking_profiler(
    const void* d_data,
    size_t total_bytes,
    size_t chunk_bytes,
    double error_bound,
    float w0, float w1, float w2, float bw_bytes_per_ms,
    int n_repeats,
    FILE* csv,
    FILE* costs_csv,
    const char* phase_name,
    int timestep,
    RankingMilestoneResult* out)
{
    return run_ranking_profiler(d_data, total_bytes, chunk_bytes,
                                error_bound, w0, w1, w2, bw_bytes_per_ms,
                                n_repeats, csv, costs_csv, phase_name, timestep, out);
}

int vpic_is_ranking_milestone(int t, int total) {
    return is_ranking_milestone(t, total) ? 1 : 0;
}

void vpic_write_ranking_csv_header(FILE* csv) {
    write_ranking_csv_header(csv);
}

void vpic_write_ranking_costs_csv_header(FILE* csv) {
    write_ranking_costs_csv_header(csv);
}

} /* extern "C" */
