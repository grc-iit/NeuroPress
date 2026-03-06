# VPIC-Kokkos Integration Plan

## Context

GPUCompress needs vpic-kokkos as a second simulation data source (alongside Gray-Scott) for benchmarking GPU-accelerated compression. VPIC-Kokkos is a large external PIC simulation that keeps field/particle/hydro data GPU-resident in Kokkos::Views. Unlike Gray-Scott (self-contained CUDA kernels), VPIC builds separately. The adapter is a **thin data-access layer** — it borrows device pointers from Kokkos::Views and passes them to `gpucompress_compress_gpu()`. No simulation logic lives in GPUCompress.

## Approach

Follow the Gray-Scott pattern exactly: public C header → opaque handle → lifecycle API → convenience function. Key difference: the adapter does **not own** GPU memory — it borrows pointers via `attach()` instead of `create()` allocating buffers.

Kokkos stays out of GPUCompress's headers entirely. `Kokkos::View::data()` returns a raw CUDA device pointer — the caller extracts it and passes `float*` to the adapter. An optional header-only Kokkos helper is provided for convenience.

## Files to Create

### 1. `include/gpucompress_vpic.h` — Public C API
- `vpic_data_type_t` enum: `VPIC_DATA_FIELDS` (16 vars), `VPIC_DATA_HYDRO` (14 vars), `VPIC_DATA_PARTICLES` (7 floats + 1 int)
- `VpicSettings` struct: `data_type`, `n_cells`, `n_particles` + `vpic_default_settings()` helper
- Opaque handle: `typedef struct gpucompress_vpic* gpucompress_vpic_t`
- Lifecycle: `create(handle, settings)`, `destroy(handle)`
- `attach(handle, d_data, d_data_i, n_elements)` — borrows device pointers, no ownership transfer
- Data access: `get_device_ptrs(handle, &d_data, &d_data_i, &nbytes_float, &nbytes_int)`, `get_nbytes(handle, &nbytes)`
- Convenience: `gpucompress_compress_vpic(d_data, nbytes, d_output, &output_size, config, stats)`

### 2. `src/vpic/vpic_adapter.cu` — Driver (~120 lines)
- Internal struct stores: `VpicSettings`, borrowed `float* d_data`, `int* d_data_i`, `n_elements`, `n_vars` (16/14/7), computed `nbytes`
- `create()`: allocates handle, sets `n_vars` from `data_type`
- `attach()`: stores pointer + element count, computes `nbytes = n_elements * n_vars * sizeof(float)`
- `get_device_ptrs()`: returns stored pointers
- `gpucompress_compress_vpic()`: calls `gpucompress_compress_gpu()` directly
- Template: `src/gray-scott/gray_scott_sim.cu` (same error handling pattern, same `extern "C"` wrapping)

### 3. `examples/vpic_vol_demo.cu` — End-to-end demo
- Works **without** vpic-kokkos installed — uses synthetic GPU data matching VPIC layouts
- CUDA kernel fills buffers with synthetic EM field patterns (sinusoidal, to mimic real physics)
- Pipeline: allocate → fill synthetic → attach → compress via VOL → write HDF5 → read back → verify
- CLI args: `--data_type {fields|hydro|particles}`, `--n_cells N`, `--chunk_mb N`, `[model.nnwt]`
- Follows `examples/grayscott_vol_demo.cu` structure

### 4. `tests/test_vpic_adapter.cu` — Correctness tests
- Lifecycle test (create/attach/destroy)
- `get_nbytes` correctness for each data type
- Compression round-trip for fields (16-var), hydro (14-var), particles (7-var float + int)
- Follows `tests/test_grayscott_gpu.cu` structure

### 5. `examples/vpic_kokkos_bridge.hpp` — Optional Kokkos helper (header-only, not compiled)
- `vpic_attach_fields(handle, k_f_d)` — calls `attach(handle, k_f_d.data(), NULL, k_f_d.extent(0))`
- `vpic_attach_hydro(handle, k_h_d)` — same pattern
- `vpic_attach_particles(handle, k_p_d, k_p_i_d)` — both float and int views
- Requires `#include <Kokkos_Core.hpp>` — only usable from vpic deck code

## Files to Modify

### 6. `CMakeLists.txt`
- Add `src/vpic/vpic_adapter.cu` to `LIB_SOURCES` (alongside gray-scott sources at ~line 72)
- Add `test_vpic_adapter` executable (alongside `test_grayscott_gpu` at ~line 295)
- Add `vpic_vol_demo` executable inside the HDF5 VOL block (alongside `grayscott_vol_demo` at ~line 822)

## Implementation Order

1. Create `include/gpucompress_vpic.h`
2. Create `src/vpic/vpic_adapter.cu`
3. Update `CMakeLists.txt` — add source + test + demo targets
4. Create `tests/test_vpic_adapter.cu`
5. Create `examples/vpic_vol_demo.cu`
6. Create `examples/vpic_kokkos_bridge.hpp`

## Verification

1. `cmake --build build` — compiles without vpic-kokkos dependency
2. `./build/test_vpic_adapter` — passes all lifecycle + round-trip tests with synthetic data
3. `./build/vpic_vol_demo` — runs full HDF5 VOL pipeline, prints compression stats, verifies lossless round-trip
4. Existing Gray-Scott tests still pass (no regressions)
