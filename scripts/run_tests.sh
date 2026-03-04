#!/bin/bash
#SBATCH --job-name=gpucompress-tests
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=test_results_%j.out
#SBATCH --error=test_results_%j.err

set -euo pipefail

LIBS_DIR=/u/imuradli/GPUCompress/libs
export LD_LIBRARY_PATH=${LIBS_DIR}:/tmp/lib:/tmp/hdf5-install/lib:${LD_LIBRARY_PATH:-}
WEIGHTS=/u/imuradli/GPUCompress/neural_net/weights/model.nnwt
cd /u/imuradli/GPUCompress

echo "========================================"
echo " GPUCompress Test Suite"
echo " $(date)"
echo " Node: $(hostname)"
echo "========================================"

nvidia-smi --query-gpu=gpu_name,compute_cap,driver_version,memory.total --format=csv,noheader
echo ""

PASS=0
FAIL=0
SKIP=0
RESULTS=""

run_test() {
    local name="$1"
    local cmd="$2"
    local timeout_sec="${3:-60}"

    echo "--- ${name} ---"
    START=$(date +%s%N)
    if timeout "${timeout_sec}" bash -c "${cmd}" > "/tmp/test_${name}.log" 2>&1; then
        END=$(date +%s%N)
        ELAPSED=$(( (END - START) / 1000000 ))
        echo "  PASS  (${ELAPSED} ms)"
        RESULTS="${RESULTS}\nPASS  ${name}  (${ELAPSED} ms)"
        PASS=$((PASS + 1))
    else
        EXIT_CODE=$?
        END=$(date +%s%N)
        ELAPSED=$(( (END - START) / 1000000 ))
        if [[ $EXIT_CODE -eq 124 ]]; then
            echo "  TIMEOUT after ${timeout_sec}s"
            RESULTS="${RESULTS}\nTIMEOUT  ${name}"
            SKIP=$((SKIP + 1))
        else
            echo "  FAIL  (exit ${EXIT_CODE}, ${ELAPSED} ms)"
            tail -5 "/tmp/test_${name}.log" 2>/dev/null | sed 's/^/    /'
            RESULTS="${RESULTS}\nFAIL  ${name}  (exit ${EXIT_CODE})"
            FAIL=$((FAIL + 1))
        fi
    fi
    echo ""
}

echo ""
echo "========================================"
echo " Unit / Regression Tests"
echo "========================================"
echo ""

run_test "quantization_roundtrip" "./build/test_quantization_roundtrip"
run_test "nn"                     "./build/test_nn"
run_test "nn_pipeline"            "./build/test_nn_pipeline"
run_test "nn_reinforce"           "./build/test_nn_reinforce"
run_test "bug3_sgd_gradients"     "./build/test_bug3_sgd_gradients"
run_test "bug4_format_string"     "./build/test_bug4_format_string"
run_test "bug5_truncated_nnwt"    "./build/test_bug5_truncated_nnwt"
run_test "bug7_concurrent_quant"  "./build/test_bug7_concurrent_quantize"
run_test "bug8_sgd_concurrent"    "./build/test_bug8_sgd_concurrent"
run_test "sgd_weight_update"      "./build/test_sgd_weight_update"
run_test "perf2_sort_speedup"     "./build/test_perf2_sort_speedup"
run_test "perf4_batched_dh"       "./build/test_perf4_batched_dh"
run_test "perf14_atomic_double"   "./build/test_perf14_atomic_double"
run_test "perf16_gather_stream"   "./build/test_perf16_gather_stream"
run_test "design6_chunk_tracker"  "./build/test_design6_chunk_tracker"
run_test "qw4_atomic_counters"    "./build/test_qw4_atomic_counters"

echo ""
echo "========================================"
echo " HDF5 Tests"
echo "========================================"
echo ""

run_test "hdf5_configs"           "./build/test_hdf5_configs"
run_test "f9_transfers"           "./build/test_f9_transfers"
run_test "h5z_8mb"                "./build/test_h5z_8mb"

echo ""
echo "========================================"
echo " VOL Connector Tests"
echo "========================================"
echo ""

run_test "vol_gpu_write"          "./build/test_vol_gpu_write"
run_test "vol2_gpu_fallback"      "./build/test_vol2_gpu_fallback"
run_test "vol_8mb"                "./build/test_vol_8mb"
run_test "vol_comprehensive"      "./build/test_vol_comprehensive"
run_test "correctness_vol"        "./build/test_correctness_vol"

echo ""
echo "========================================"
echo " Benchmarks (60s timeout each)"
echo "========================================"
echo ""

run_test "benchmark_hdf5"         "./build/benchmark_hdf5 ${WEIGHTS}" 180
run_test "benchmark_lz4_vs_nocomp" "./build/benchmark_lz4_vs_nocomp" 60
run_test "benchmark_gpu_resident" "./build/benchmark_gpu_resident ${WEIGHTS}" 60
run_test "benchmark_vol_gpu"      "./build/benchmark_vol_gpu ${WEIGHTS}" 60
run_test "benchmark_algo_sweep"   "./build/benchmark_algo_sweep" 120
run_test "quick_algo_sweep"       "./build/quick_algo_sweep" 60

echo ""
echo "========================================"
echo " SUMMARY"
echo "========================================"
echo ""
echo -e "${RESULTS}"
echo ""
echo "----------------------------------------"
echo "PASS: ${PASS}  FAIL: ${FAIL}  TIMEOUT/SKIP: ${SKIP}  TOTAL: $((PASS + FAIL + SKIP))"
echo "----------------------------------------"
echo ""
echo "Done at $(date)"
