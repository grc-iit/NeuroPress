# ============================================================
# Unit Tests (no HDF5 dependency)
# ============================================================

# Quantization round-trip verification
add_executable(test_quantization_roundtrip tests/unit/test_quantization_roundtrip.c)
target_include_directories(test_quantization_roundtrip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_quantization_roundtrip PRIVATE gpucompress m)

# Gray-Scott GPU simulation test
add_executable(test_grayscott_gpu tests/unit/test_grayscott_gpu.cu)
set_source_files_properties(tests/unit/test_grayscott_gpu.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_grayscott_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_grayscott_gpu PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_grayscott_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# VPIC adapter test
add_executable(test_vpic_adapter tests/unit/test_vpic_adapter.cu)
set_source_files_properties(tests/unit/test_vpic_adapter.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_vpic_adapter PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_vpic_adapter PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_vpic_adapter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

# CLI test
add_executable(test_cli tests/unit/test_cli.cu)
target_include_directories(test_cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_cli PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_cli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# API test
add_executable(test_api tests/unit/test_api.cu)
target_include_directories(test_api PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_api PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_api PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

add_executable(test_nn_pipeline tests/nn/test_nn_pipeline.cpp)
target_include_directories(test_nn_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_nn_pipeline PRIVATE gpucompress pthread)

add_executable(test_nn_shuffle tests/nn/test_nn_shuffle.cu)
target_include_directories(test_nn_shuffle PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_nn_shuffle PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_nn_shuffle PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

add_executable(test_bug4_format_string tests/regression/test_bug4_format_string.cu)
set_source_files_properties(tests/regression/test_bug4_format_string.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_bug4_format_string PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_bug4_format_string PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_bug4_format_string PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

add_executable(test_m5_header_overread tests/regression/test_m5_header_overread.cu)
set_source_files_properties(tests/regression/test_m5_header_overread.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m5_header_overread PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m5_header_overread PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_m5_header_overread PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

add_executable(test_m8m9_warp_mask tests/regression/test_m8m9_warp_mask.cu)
set_source_files_properties(tests/regression/test_m8m9_warp_mask.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m8m9_warp_mask PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m8m9_warp_mask PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_m8m9_warp_mask PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_m10_stats_workspace_race tests/regression/test_m10_stats_workspace_race.cu)
set_source_files_properties(tests/regression/test_m10_stats_workspace_race.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m10_stats_workspace_race PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m10_stats_workspace_race PRIVATE gpucompress CUDA::cudart pthread)
set_target_properties(test_m10_stats_workspace_race PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_m11_chunk_arrays_raii tests/regression/test_m11_chunk_arrays_raii.cu)
set_source_files_properties(tests/regression/test_m11_chunk_arrays_raii.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m11_chunk_arrays_raii PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m11_chunk_arrays_raii PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_m11_chunk_arrays_raii PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_m12_range_bufs_race tests/regression/test_m12_range_bufs_race.cu)
set_source_files_properties(tests/regression/test_m12_range_bufs_race.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m12_range_bufs_race PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m12_range_bufs_race PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_m12_range_bufs_race PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_m13_compute_range_errors tests/regression/test_m13_compute_range_errors.cu)
set_source_files_properties(tests/regression/test_m13_compute_range_errors.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m13_compute_range_errors PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m13_compute_range_errors PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_m13_compute_range_errors PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_m3_pool_init_failure tests/regression/test_m3_pool_init_failure.cu)
set_source_files_properties(tests/regression/test_m3_pool_init_failure.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_m3_pool_init_failure PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_m3_pool_init_failure PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_m3_pool_init_failure PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

add_executable(test_c2_timing_race tests/regression/test_c2_timing_race.cu)
set_source_files_properties(tests/regression/test_c2_timing_race.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_c2_timing_race PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_c2_timing_race PRIVATE gpucompress CUDA::cudart pthread m)
set_target_properties(test_c2_timing_race PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_c1_exploration_header tests/regression/test_c1_exploration_header.cu)
set_source_files_properties(tests/regression/test_c1_exploration_header.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_c1_exploration_header PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_c1_exploration_header PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_c1_exploration_header PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_h3_auto_global_buffers tests/regression/test_h3_auto_global_buffers.cu)
set_source_files_properties(tests/regression/test_h3_auto_global_buffers.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h3_auto_global_buffers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_h3_auto_global_buffers PRIVATE gpucompress CUDA::cudart pthread m)
set_target_properties(test_h3_auto_global_buffers PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_h9_int8_error_bound tests/regression/test_h9_int8_error_bound.cu)
set_source_files_properties(tests/regression/test_h9_int8_error_bound.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h9_int8_error_bound PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_h9_int8_error_bound PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_h9_int8_error_bound PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_h8_zero_compressed_size tests/regression/test_h8_zero_compressed_size.cu)
set_source_files_properties(tests/regression/test_h8_zero_compressed_size.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h8_zero_compressed_size PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_h8_zero_compressed_size PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_h8_zero_compressed_size PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_h7_version_check tests/regression/test_h7_version_check.cu)
set_source_files_properties(tests/regression/test_h7_version_check.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h7_version_check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_h7_version_check PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_h7_version_check PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

add_executable(test_h4_sgd_mutex tests/regression/test_h4_sgd_mutex.cu)
set_source_files_properties(tests/regression/test_h4_sgd_mutex.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_h4_sgd_mutex PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_h4_sgd_mutex PRIVATE gpucompress CUDA::cudart pthread m)
set_target_properties(test_h4_sgd_mutex PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

add_executable(test_perf4_batched_dh tests/perf/test_perf4_batched_dh.cu)
set_source_files_properties(tests/perf/test_perf4_batched_dh.cu PROPERTIES LANGUAGE CUDA)
target_include_directories(test_perf4_batched_dh PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_perf4_batched_dh PRIVATE gpucompress CUDA::cudart m)
set_target_properties(test_perf4_batched_dh PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

add_executable(test_shuffle_quant_effect tests/test_shuffle_quant_effect.cu)
target_include_directories(test_shuffle_quant_effect PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_shuffle_quant_effect PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_shuffle_quant_effect PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

