# ============================================================
# Helper: register a test target with CTest
#   gpucompress_add_test(<target> <label> [timeout_seconds])
# Working directory is always the project root so tests can
# find neural_net/weights/model.nnwt via relative path.
# ============================================================
macro(gpucompress_add_test target label)
    set(_timeout 60)
    if(${ARGC} GREATER 2)
        set(_timeout ${ARGV2})
    endif()
    add_test(NAME ${target}
             COMMAND ${target}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(${target} PROPERTIES
        LABELS   "${label}"
        TIMEOUT  ${_timeout})
endmacro()

# ============================================================
# Unit Tests (no HDF5 dependency)
# ============================================================

# (Removed: test_quantization_roundtrip — uses host-path stub gpucompress_compress())

# (Removed: test_grayscott_gpu — host-path stub)

# VPIC adapter test
add_executable(test_vpic_adapter tests/unit/test_vpic_adapter.cu)
set_source_files_properties(tests/unit/test_vpic_adapter.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_vpic_adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_vpic_adapter PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_vpic_adapter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# WarpX adapter test
add_executable(test_warpx_adapter tests/unit/test_warpx_adapter.cu)
set_source_files_properties(tests/unit/test_warpx_adapter.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_warpx_adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_warpx_adapter PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_warpx_adapter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# WarpX policy benchmark (fixed vs NN, ratio vs balanced, lossless vs lossy)
add_executable(bench_warpx_policies tests/unit/bench_warpx_policies.cu)
set_source_files_properties(tests/unit/bench_warpx_policies.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(bench_warpx_policies PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(bench_warpx_policies PRIVATE gpucompress CUDA::cudart m)
set_target_properties(bench_warpx_policies PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# WarpX hyperparameter study: uses generic_benchmark via run_warpx_benchmark.sh
# (see benchmarks/warpx/run_warpx_benchmark.sh)

# EMA gradient buffer reset test (verifies CRITICAL-2 fix)
# NOTE: linked to HDF5 VOL in cmake/HDF5Vol.cmake (needs H5VLgpucompress target)

# Compression core test suite
add_executable(test_compression_core
    tests/unit/test_compression_core.cu
)
target_include_directories(test_compression_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_compression_core PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_compression_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Preprocessing test
add_executable(test_preprocessing
    tests/unit/test_preprocessing.cu
)
target_include_directories(test_preprocessing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_preprocessing PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_preprocessing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Stats test
add_executable(test_stats tests/unit/test_stats.cu)
target_include_directories(test_stats PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_stats PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_stats PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Transfer tracker + on-device stats init test
add_executable(test_xfer_stats_init tests/unit/test_xfer_stats_init.cu)
set_source_files_properties(tests/unit/test_xfer_stats_init.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_xfer_stats_init PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_xfer_stats_init PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_xfer_stats_init PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_compress_gpu_delegation tests/unit/test_compress_gpu_delegation.cu)
set_source_files_properties(tests/unit/test_compress_gpu_delegation.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_compress_gpu_delegation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_compress_gpu_delegation PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_compress_gpu_delegation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_cli, test_api — use host-path stub gpucompress_compress())

# Shuffle test
add_executable(test_shuffle tests/unit/test_shuffle.cu)
target_include_directories(test_shuffle PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_shuffle PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_shuffle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Quantization suite
add_executable(test_quantization
    tests/unit/quantization/test_quantization_suite.cu
)
target_include_directories(test_quantization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_quantization PRIVATE gpucompress CUDA::cudart CUDA::curand)
set_target_properties(test_quantization PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Quantization CUB min/max verification
add_executable(test_quantization_cub
    tests/unit/test_quantization_cub.cu
)
set_source_files_properties(tests/unit/test_quantization_cub.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_quantization_cub PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_quantization_cub PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_quantization_cub PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================
# Neural Network Tests
# ============================================================

add_executable(test_nn tests/nn/test_nn.cu)
target_include_directories(test_nn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_nn PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_nn_reinforce tests/nn/test_nn_reinforce.cpp)
set_source_files_properties(tests/nn/test_nn_reinforce.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(test_nn_reinforce PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_nn_reinforce PRIVATE gpucompress CUDA::cudart pthread)

# (Removed: test_nn_pipeline — host-path stub)
# (Removed: test_nn_shuffle — missing input file)

add_executable(test_nn_cost_ranking tests/nn/test_nn_cost_ranking.cu)
set_source_files_properties(tests/nn/test_nn_cost_ranking.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_nn_cost_ranking PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_nn_cost_ranking PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_nn_cost_ranking PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_sgd_weight_update tests/nn/test_sgd_weight_update.cu)
set_source_files_properties(tests/nn/test_sgd_weight_update.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_sgd_weight_update PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_sgd_weight_update PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_sgd_weight_update PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================
# Bug Regression Tests (no HDF5 dependency)
# ============================================================

add_executable(test_bug3_sgd_gradients tests/regression/test_bug3_sgd_gradients.cu)
set_source_files_properties(tests/regression/test_bug3_sgd_gradients.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_bug3_sgd_gradients PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_bug3_sgd_gradients PRIVATE gpucompress CUDA::cudart pthread)
set_target_properties(test_bug3_sgd_gradients PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_bug4_format_string — uses host-path stub)

add_executable(test_bug5_truncated_nnwt tests/regression/test_bug5_truncated_nnwt.cu)
set_source_files_properties(tests/regression/test_bug5_truncated_nnwt.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_bug5_truncated_nnwt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_bug5_truncated_nnwt PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_bug5_truncated_nnwt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_bug8_sgd_concurrent tests/regression/test_bug8_sgd_concurrent.cu)
set_source_files_properties(tests/regression/test_bug8_sgd_concurrent.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_bug8_sgd_concurrent PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_bug8_sgd_concurrent PRIVATE gpucompress CUDA::cudart pthread)
set_target_properties(test_bug8_sgd_concurrent PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================
# Performance Regression Tests (no HDF5 dependency)
# ============================================================

add_executable(test_m1m2_global_races tests/regression/test_m1m2_global_races.cu)
set_source_files_properties(tests/regression/test_m1m2_global_races.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m1m2_global_races PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m1m2_global_races PRIVATE gpucompress CUDA::cudart pthread)
set_target_properties(test_m1m2_global_races PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_m4_chunk_history_realloc tests/regression/test_m4_chunk_history_realloc.cu)
set_source_files_properties(tests/regression/test_m4_chunk_history_realloc.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m4_chunk_history_realloc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m4_chunk_history_realloc PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_m4_chunk_history_realloc PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_m5_header_overread — uses host-path stub)

add_executable(test_m6_integer_overflow tests/regression/test_m6_integer_overflow.cu)
set_source_files_properties(tests/regression/test_m6_integer_overflow.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m6_integer_overflow PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m6_integer_overflow PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_m6_integer_overflow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_m7_header_async tests/regression/test_m7_header_async.cu)
set_source_files_properties(tests/regression/test_m7_header_async.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m7_header_async PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m7_header_async PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_m7_header_async PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_m13_compute_range_errors — uses host-path stub)

# (Removed: test_m3_pool_init_failure — host-path stub)

add_executable(test_h8_pool_partial_leak tests/regression/test_h8_pool_partial_leak.cu)
set_source_files_properties(tests/regression/test_h8_pool_partial_leak.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h8_pool_partial_leak PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_h8_pool_partial_leak PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_h8_pool_partial_leak PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_c6c7_init_error_checking — uses host-path stub)

# (Removed: test_c1_nn_reload_race — host-path stub)

# (Removed: test_h5_configure_compression_check — uses host-path stub)

add_executable(test_c2_learning_flag_race tests/regression/test_c2_learning_flag_race.cu)
set_source_files_properties(tests/regression/test_c2_learning_flag_race.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_c2_learning_flag_race PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_c2_learning_flag_race PRIVATE gpucompress CUDA::cudart pthread)
set_target_properties(test_c2_learning_flag_race PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_h14_cleanup_order tests/regression/test_h14_cleanup_order.cu)
set_source_files_properties(tests/regression/test_h14_cleanup_order.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h14_cleanup_order PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_h14_cleanup_order PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_h14_cleanup_order PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_h1_sgd_inference_race tests/regression/test_h1_sgd_inference_race.cu)
set_source_files_properties(tests/regression/test_h1_sgd_inference_race.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h1_sgd_inference_race PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_h1_sgd_inference_race PRIVATE gpucompress CUDA::cudart pthread m)
set_target_properties(test_h1_sgd_inference_race PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_c4_int32_clamp tests/regression/test_c4_int32_clamp.cu)
set_source_files_properties(tests/regression/test_c4_int32_clamp.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_c4_int32_clamp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_c4_int32_clamp PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_c4_int32_clamp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_c2_timing_race — uses host-path stub)

add_executable(test_c1_exploration_header tests/regression/test_c1_exploration_header.cu)
set_source_files_properties(tests/regression/test_c1_exploration_header.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_c1_exploration_header PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_c1_exploration_header PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_c1_exploration_header PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_error_handling_fixes tests/regression/test_error_handling_fixes.cu)
set_source_files_properties(tests/regression/test_error_handling_fixes.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_error_handling_fixes PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_error_handling_fixes PRIVATE gpucompress CUDA::cudart pthread m)
set_target_properties(test_error_handling_fixes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_exploration_preproc — API_CHANGE: assertions don't match current NN behavior)

add_executable(test_nn_preproc_debug tests/regression/test_nn_preproc_debug.cu)
set_source_files_properties(tests/regression/test_nn_preproc_debug.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_nn_preproc_debug PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_nn_preproc_debug PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_nn_preproc_debug PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_nn_action_diversity — API_CHANGE: NN never picks shuffle/quant)

# (Removed: test_h3_auto_global_buffers — uses host-path stub)

add_executable(test_h9_int8_error_bound tests/regression/test_h9_int8_error_bound.cu)
set_source_files_properties(tests/regression/test_h9_int8_error_bound.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h9_int8_error_bound PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_h9_int8_error_bound PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_h9_int8_error_bound PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_h8_zero_compressed_size, test_h7_version_check — use host-path stub)

add_executable(test_h6_cub_int_overflow tests/regression/test_h6_cub_int_overflow.cu)
set_source_files_properties(tests/regression/test_h6_cub_int_overflow.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h6_cub_int_overflow PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_h6_cub_int_overflow PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_h6_cub_int_overflow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_explore_lossless_skip tests/regression/test_explore_lossless_skip_roundtrip.cu)
set_source_files_properties(tests/regression/test_explore_lossless_skip_roundtrip.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_explore_lossless_skip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_explore_lossless_skip PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_explore_lossless_skip PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_sgd_target_clamping tests/regression/test_sgd_target_clamping.cu)
set_source_files_properties(tests/regression/test_sgd_target_clamping.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_sgd_target_clamping PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_sgd_target_clamping PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_sgd_target_clamping PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_h2_unnecessary_stream_sync — uses host-path stub)

add_executable(test_h6_size_overflow tests/regression/test_h6_size_overflow.cu)
set_source_files_properties(tests/regression/test_h6_size_overflow.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h6_size_overflow PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_h6_size_overflow PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_h6_size_overflow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# (Removed: test_h4_sgd_mutex — uses host-path stub)

# ============================================================
# Performance Regression Tests (no HDF5 dependency)
# ============================================================

add_executable(test_perf2_sort_speedup tests/perf/test_perf2_sort_speedup.cu)
set_source_files_properties(tests/perf/test_perf2_sort_speedup.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_perf2_sort_speedup PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_perf2_sort_speedup PRIVATE CUDA::cudart)
set_target_properties(test_perf2_sort_speedup PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/perf/test_perf4_batched_dh.cu")
add_executable(test_perf4_batched_dh tests/perf/test_perf4_batched_dh.cu)
set_source_files_properties(tests/perf/test_perf4_batched_dh.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_perf4_batched_dh PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_perf4_batched_dh PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_perf4_batched_dh PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(test_perf14_atomic_double tests/perf/test_perf14_atomic_double.cu)
set_source_files_properties(tests/perf/test_perf14_atomic_double.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_perf14_atomic_double PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_perf14_atomic_double PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_perf14_atomic_double PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_nn_ratio_prediction tests/regression/test_nn_ratio_prediction.cu)
set_source_files_properties(tests/regression/test_nn_ratio_prediction.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_nn_ratio_prediction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_nn_ratio_prediction PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_nn_ratio_prediction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_nn_timing_inflation tests/regression/test_nn_timing_inflation.cu)
set_source_files_properties(tests/regression/test_nn_timing_inflation.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_nn_timing_inflation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_nn_timing_inflation PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_nn_timing_inflation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_nn_inference_profile tests/perf/test_nn_inference_profile.cu)
set_source_files_properties(tests/perf/test_nn_inference_profile.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_nn_inference_profile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_nn_inference_profile PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_nn_inference_profile PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_p1_preproc_correctness tests/perf/test_p1_preproc_correctness.cu)
set_source_files_properties(tests/perf/test_p1_preproc_correctness.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_p1_preproc_correctness PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_p1_preproc_correctness PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_p1_preproc_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_p4_diag_overhead tests/perf/test_p4_diag_overhead.cu)
set_source_files_properties(tests/perf/test_p4_diag_overhead.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_p4_diag_overhead PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_p4_diag_overhead PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_p4_diag_overhead PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_perf_optimizations tests/perf/test_perf_optimizations.cu)
set_source_files_properties(tests/perf/test_perf_optimizations.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_perf_optimizations PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_perf_optimizations PRIVATE gpucompress CUDA::cudart m pthread)
set_target_properties(test_perf_optimizations PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_k4_shuffle_throughput tests/perf/test_k4_shuffle_throughput.cu)
set_source_files_properties(tests/perf/test_k4_shuffle_throughput.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_k4_shuffle_throughput PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_k4_shuffle_throughput PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_k4_shuffle_throughput PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_shuffle_quant_effect tests/test_shuffle_quant_effect.cu)
target_include_directories(test_shuffle_quant_effect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_shuffle_quant_effect PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_shuffle_quant_effect PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(explore_algo_patterns tests/explore_algo_patterns.cu)
set_source_files_properties(tests/explore_algo_patterns.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(explore_algo_patterns PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(explore_algo_patterns PRIVATE gpucompress CUDA::cudart m)
set_target_properties(explore_algo_patterns PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ============================================================
# CTest registrations
# ============================================================

# Unit
gpucompress_add_test(test_vpic_adapter          unit)
gpucompress_add_test(test_warpx_adapter         unit)
gpucompress_add_test(test_compression_core      unit)
gpucompress_add_test(test_preprocessing         unit)
gpucompress_add_test(test_stats                 unit)
gpucompress_add_test(test_xfer_stats_init       unit)
gpucompress_add_test(test_compress_gpu_delegation unit)
gpucompress_add_test(test_shuffle               unit)
gpucompress_add_test(test_quantization          unit)
gpucompress_add_test(test_quantization_cub      unit)
gpucompress_add_test(test_shuffle_quant_effect  unit)

# NN
gpucompress_add_test(test_nn                    nn  120)
gpucompress_add_test(test_nn_reinforce          nn  120)
gpucompress_add_test(test_nn_cost_ranking       nn  120)
gpucompress_add_test(test_sgd_weight_update     nn  120)

# Regression (no HDF5)
gpucompress_add_test(test_bug3_sgd_gradients    regression)
gpucompress_add_test(test_bug5_truncated_nnwt   regression)
gpucompress_add_test(test_bug8_sgd_concurrent   regression)
gpucompress_add_test(test_m1m2_global_races     regression)
gpucompress_add_test(test_m4_chunk_history_realloc regression)
gpucompress_add_test(test_m6_integer_overflow   regression)
gpucompress_add_test(test_m7_header_async       regression)
gpucompress_add_test(test_h8_pool_partial_leak  regression)
gpucompress_add_test(test_c2_learning_flag_race regression)
gpucompress_add_test(test_h14_cleanup_order     regression)
gpucompress_add_test(test_h1_sgd_inference_race regression)
gpucompress_add_test(test_c4_int32_clamp        regression)
gpucompress_add_test(test_c1_exploration_header regression)
gpucompress_add_test(test_error_handling_fixes  regression)
gpucompress_add_test(test_nn_preproc_debug      regression)
gpucompress_add_test(test_h9_int8_error_bound   regression)
gpucompress_add_test(test_h6_cub_int_overflow   regression)
gpucompress_add_test(test_explore_lossless_skip regression)
gpucompress_add_test(test_sgd_target_clamping   regression)
gpucompress_add_test(test_h6_size_overflow      regression)

# Perf
gpucompress_add_test(test_perf2_sort_speedup    perf  120)
if(TARGET test_perf4_batched_dh)
    gpucompress_add_test(test_perf4_batched_dh  perf  120)
endif()
gpucompress_add_test(test_perf14_atomic_double  perf  120)
gpucompress_add_test(test_nn_ratio_prediction   perf  120)
gpucompress_add_test(test_nn_timing_inflation   perf  120)
gpucompress_add_test(test_nn_inference_profile  perf  120)
gpucompress_add_test(test_p1_preproc_correctness perf 120)
gpucompress_add_test(test_p4_diag_overhead      perf  120)
gpucompress_add_test(test_perf_optimizations    perf  120)
gpucompress_add_test(test_k4_shuffle_throughput perf  120)

