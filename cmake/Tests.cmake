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
    ${PREPROCESSING_SOURCES}
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
    ${PREPROCESSING_SOURCES}
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

# CPU stats test
add_executable(test_cpu_stats tests/unit/test_cpu_stats.cu)
target_include_directories(test_cpu_stats PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_cpu_stats PRIVATE gpucompress CUDA::cudart)
set_target_properties(test_cpu_stats PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Quantization suite
add_executable(test_quantization
    tests/unit/quantization/test_quantization_suite.cu
    ${PREPROCESSING_SOURCES}
)
target_include_directories(test_quantization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(test_quantization PRIVATE gpucompress CUDA::cudart CUDA::curand)
set_target_properties(test_quantization PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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

