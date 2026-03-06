/**
 * @file vpic_kokkos_bridge.hpp
 * @brief Header-only Kokkos helper for VPIC adapter
 *
 * Convenience functions to attach Kokkos::Views to the VPIC adapter.
 * This file requires Kokkos and is only usable from vpic deck code.
 * It is NOT compiled by the GPUCompress build system.
 *
 * Usage (in a vpic deck or Kokkos-enabled translation unit):
 *
 *   #include <Kokkos_Core.hpp>
 *   #include "gpucompress_vpic.h"
 *   #include "vpic_kokkos_bridge.hpp"
 *
 *   // k_f_d is Kokkos::View<float*, Kokkos::CudaSpace>
 *   vpic_attach_fields(handle, k_f_d);
 */

#ifndef GPUCOMPRESS_VPIC_KOKKOS_BRIDGE_HPP
#define GPUCOMPRESS_VPIC_KOKKOS_BRIDGE_HPP

#include "gpucompress_vpic.h"
#include <Kokkos_Core.hpp>

/**
 * Attach a Kokkos field view (16 vars × n_cells) to the adapter.
 * View layout must be contiguous (LayoutLeft or LayoutRight).
 */
template <typename ViewType>
inline gpucompress_error_t vpic_attach_fields(
    gpucompress_vpic_t handle,
    const ViewType& k_f_d)
{
    return gpucompress_vpic_attach(
        handle,
        k_f_d.data(),
        NULL,
        k_f_d.extent(0));
}

/**
 * Attach a Kokkos hydro view (14 vars × n_cells) to the adapter.
 */
template <typename ViewType>
inline gpucompress_error_t vpic_attach_hydro(
    gpucompress_vpic_t handle,
    const ViewType& k_h_d)
{
    return gpucompress_vpic_attach(
        handle,
        k_h_d.data(),
        NULL,
        k_h_d.extent(0));
}

/**
 * Attach Kokkos particle views (7 float vars + 1 int species per particle).
 *
 * @param k_p_d   Float view: n_particles × 7
 * @param k_p_i_d Int view:   n_particles × 1
 */
template <typename FloatViewType, typename IntViewType>
inline gpucompress_error_t vpic_attach_particles(
    gpucompress_vpic_t handle,
    const FloatViewType& k_p_d,
    const IntViewType&   k_p_i_d)
{
    return gpucompress_vpic_attach(
        handle,
        k_p_d.data(),
        k_p_i_d.data(),
        k_p_d.extent(0));
}

#endif /* GPUCOMPRESS_VPIC_KOKKOS_BRIDGE_HPP */
