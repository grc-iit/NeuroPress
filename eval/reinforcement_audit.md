# Reinforcement Pipeline Audit

Audit of the online reinforcement (SGD) pipeline in GPUCompress.

## How Reinforcement Works

1. NN predicts best compression config + expected ratio and comp time
2. Actual compression runs, measures real ratio and time
3. If MAPE > threshold, SGD fires: backpropagate error, update weights in GPU memory
4. Exploration optionally tries up to 31 alternative configs for ground truth
5. Weights accumulate corrections over time but are never saved to disk

## When SGD Does NOT Fire

- Ratio MAPE and comp_time MAPE are both below their thresholds
- Active learning is not enabled (`gpucompress_enable_active_learning()` not called)
- `--reinforce` flag is not passed
- NN weights are not loaded
- Algorithm is not `ALGO_AUTO`
- Predicted comp_time is negative (silently skipped)

## Bugs and Issues

### Critical

#### 1. Explored samples never fed to SGD

**Files:** `src/api/gpucompress_api.cpp` lines 565-788

The exploration loop compresses up to 31 alternative configs and collects their ground-truth ratios and comp times into `explored_samples`. However, the reinforcement block only trains on the primary config. All that expensive exploration data is discarded.

The reinforcement code builds input features and calls `nn_reinforce_add_sample()` only once for `nn_action`, then immediately calls `nn_reinforce_apply()`. The `explored_samples` vector goes out of scope unused.

**Impact:** SGD learns from 1 sample per file instead of up to 32. Much slower adaptation and noisier gradients.

#### 2. Reinforcement requires active learning enabled (undocumented)

**File:** `src/api/gpucompress_api.cpp` line 531

The entire reinforcement block is inside `if (cfg.algorithm == GPUCOMPRESS_ALGO_AUTO && nn_was_used && g_active_learning_enabled)`. If a user calls `gpucompress_set_reinforcement(1, ...)` without also calling `gpucompress_enable_active_learning()`, SGD never fires. This dependency is not documented in the public API.

**Impact:** Silent failure — user thinks reinforcement is enabled but nothing happens.

### Medium

#### 3. Single-sample SGD; batching is dead code

**File:** `src/api/gpucompress_api.cpp` lines 785-788

Every trigger adds exactly 1 sample and immediately calls `nn_reinforce_apply()`. The `nn_reinforce` module supports gradient accumulation and averaging (it divides by `sample_count` in `apply()`), but batching is never utilized because `apply()` is called right after a single `add_sample()`.

**Impact:** Noisier gradients, gradient clipping (max norm 1.0) is less effective on single samples.

#### 4. `actual_ratio` is stale after exploration swaps output

**File:** `src/api/gpucompress_api.cpp` lines 533, 719-720, 785

`actual_ratio` is computed from the primary compression result at line 533. During exploration, if an alternative config achieves a better ratio, the output is swapped (line 718-720) and `compressed_size` is updated. But `actual_ratio` is NOT recalculated. SGD trains on the primary config's ratio even though the function returns a different config's compressed data.

**Impact:** Training signal is inconsistent with the delivered output.

#### 5. Experience buffer records `comp_time_ms = 0.0` despite measurement

**File:** `src/api/gpucompress_api.cpp` lines 545 and 677

The primary experience sample (line 545) and alternative exploration samples (line 677) both hard-code `actual_comp_time_ms = 0.0` even though `primary_comp_time_ms` and `alt_ct_ms` are measured with CUDA events. The experience CSV loses all timing data, making offline retraining from experience miss comp_time information.

**Impact:** Experience buffer is useless for offline comp_time retraining.

#### 6. No thread safety on reinforcement state

**File:** `src/nn/nn_reinforce.cpp` lines 21-37

All state is in `static` global variables with no mutex protection: `h_weights`, gradient accumulators `dw1..dw3`, `db1..db3`, `sample_count`, `g_initialized`. The `gpucompress_api.cpp` also uses bare globals (`g_reinforce_enabled`, `g_reinforce_initialized`, etc.). Concurrent `gpucompress_compress()` calls would cause data races.

**Impact:** Undefined behavior in multi-threaded usage.

#### 7. Host weights never re-synced after `gpucompress_reload_nn()`

**Files:** `src/nn/nn_reinforce.cpp` lines 57-71, `src/api/gpucompress_api.cpp` lines 755-758

`nn_reinforce_init()` copies GPU weights to host once (guarded by `g_reinforce_initialized`). If weights are reloaded via `gpucompress_reload_nn()`, the host copy stays stale. The next `nn_reinforce_apply()` overwrites the fresh GPU weights with the old host copy.

**Impact:** Reloading weights while reinforcement is active silently reverts to old weights.

### Low

#### 8. Negative predicted_comp_time silently disables ct threshold

**File:** `src/api/gpucompress_api.cpp` lines 743-747

The check `predicted_comp_time > 0.0f` skips ct_error_pct computation if the NN predicts negative comp_time (possible when `expm1f(output)` returns a value in `(-1, 0)`). The ct threshold never triggers, so bad comp_time predictions go uncorrected.

#### 9. CUDA event leak on partial creation failure

**File:** `src/api/gpucompress_api.cpp` lines 439-440, 646-647

If `cudaEventCreate(&t_start)` succeeds but `cudaEventCreate(&t_stop)` fails, `t_start` is created but never destroyed. Same pattern in the exploration timing block.

#### 10. MSE factor-of-2 absorbed into learning rate

**File:** `src/nn/nn_reinforce.cpp` lines 130-137

The derivative of MSE `(y - t)^2` is `2(y - t)`, but the code uses `(y - t)` and absorbs the factor of 2 into the learning rate. This means the effective learning rate is 2x what the user sets. Also, both ratio and comp_time losses are equally weighted despite potentially different scales.

#### 11. Float `== 0.0f` sentinel for inactive outputs

**File:** `src/nn/nn_reinforce.cpp` lines 140-141

The backward pass skips outputs where `d3_out[out] == 0.0f` to detect inactive outputs. This works because inactive outputs are explicitly zeroed via `memset`, but is fragile if a legitimate gradient happens to be exactly 0.0f (rare but possible).

## Recommended Fix Priority

1. Feed explored samples to SGD (Critical #1) — biggest performance win
2. Write measured comp_time to experience buffer (Medium #5) — trivial fix
3. Fix CUDA event leak (Low #9) — trivial fix
4. Document active learning dependency (Critical #2) — or decouple them
5. Handle negative predicted_comp_time (Low #8) — clamp to 0
