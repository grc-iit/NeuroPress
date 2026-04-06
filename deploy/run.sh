#!/usr/bin/env bash
# GPUCompress Deployment Driver
# Reads gpucompress-build.yaml and simulations.yaml to automate build & benchmark.
#
# Usage:
#   ./run.sh build                  Build GPUCompress core library
#   ./run.sh bridge <name>          Build a bridge library (lammps_udf, nekrs_udf)
#   ./run.sh setup <sim>            Clone, patch, and build a simulation
#   ./run.sh bench <sim>            Run benchmark for a simulation
#   ./run.sh env                    Print the runtime environment block
#   ./run.sh list                   List available simulations
#   ./run.sh all                    Build everything, run all benchmarks
#
# Options (env vars or flags):
#   CUDA_ARCH=80            GPU architecture (default: 80 = A100)
#   BUILD_TYPE=Release      CMake build type
#   HDF5_ROOT=/tmp/hdf5-install
#   NVCOMP_PREFIX=/tmp
#   SIMS_DIR=$HOME/sims     Where simulation sources are cloned
#   RESULTS_BASE=results    Where benchmark results are collected

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GPUC_DIR
export CUDA_ARCH="${CUDA_ARCH:-80}"
export BUILD_TYPE="${BUILD_TYPE:-Release}"
export HDF5_ROOT="${HDF5_ROOT:-/tmp/hdf5-install}"
export NVCOMP_PREFIX="${NVCOMP_PREFIX:-/tmp}"
export SIMS_DIR="${SIMS_DIR:-$HOME/sims}"
export RESULTS_BASE="${RESULTS_BASE:-${GPUC_DIR}/results}"

# Simulations list (order matters for 'all')
SIMULATIONS=(lammps nyx warpx nekrs vpic)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }
die()   { err "$@"; exit 1; }

require_cmd() {
    command -v "$1" &>/dev/null || die "Required command not found: $1"
}

# Simple YAML value reader (handles flat key: value lines)
# For the structured configs we just hard-code the logic in functions below;
# this helper is for quick single-value lookups.
yaml_val() {
    local file="$1" key="$2"
    grep -E "^\s*${key}:" "$file" | head -1 | sed 's/^[^:]*:\s*//' | sed 's/^"\(.*\)"$/\1/' | sed "s/^'\(.*\)'$/\1/"
}

setup_env() {
    export LD_LIBRARY_PATH="${GPUC_DIR}/build:${GPUC_DIR}/examples:${HDF5_ROOT}/lib:${NVCOMP_PREFIX}/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export GPUCOMPRESS_WEIGHTS="${GPUC_DIR}/neural_net/weights/model.nnwt"
}

# ---------------------------------------------------------------------------
# build — Build GPUCompress core
# ---------------------------------------------------------------------------
cmd_build() {
    info "Building GPUCompress core library"
    require_cmd cmake
    require_cmd nvcc

    cd "$GPUC_DIR"

    info "Configure: CUDA_ARCH=${CUDA_ARCH}, BUILD_TYPE=${BUILD_TYPE}"
    cmake -B build -S . \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DNVCOMP_PREFIX="${NVCOMP_PREFIX}" \
        -DHDF5_ROOT="${HDF5_ROOT}"

    info "Compile ($(nproc) jobs)"
    cmake --build build -j"$(nproc)"

    # Verify key outputs
    for lib in libgpucompress.so libH5Zgpucompress.so libH5VLgpucompress.so; do
        if [ -f "build/${lib}" ]; then
            ok "Built ${lib}"
        else
            warn "Missing ${lib} (may be expected if HDF5 not found)"
        fi
    done

    if [ -f "build/generic_benchmark" ]; then
        ok "Built generic_benchmark"
    fi

    ok "GPUCompress core build complete"
}

# ---------------------------------------------------------------------------
# bridge — Build bridge libraries
# ---------------------------------------------------------------------------
cmd_bridge() {
    local name="${1:-}"
    [ -z "$name" ] && die "Usage: $0 bridge <lammps_udf|nekrs_udf|all>"

    cd "$GPUC_DIR"
    setup_env

    build_bridge() {
        local bname="$1" src="$2" out="$3"
        info "Building bridge: ${bname}"
        g++ -shared -fPIC -o "$out" "$src" \
            -Iinclude -I"${HDF5_ROOT}/include" -I/usr/local/cuda/include \
            -Lbuild -lgpucompress -lH5VLgpucompress -lH5Zgpucompress \
            -L"${HDF5_ROOT}/lib" -lhdf5 \
            -L/usr/local/cuda/lib64 -lcudart \
            -Wl,-rpath,"${GPUC_DIR}/build" -Wl,-rpath,"${HDF5_ROOT}/lib"
        ok "Built ${out}"
    }

    case "$name" in
        lammps_udf)
            build_bridge lammps_udf examples/lammps_gpucompress_udf.cpp examples/liblammps_gpucompress_udf.so
            ;;
        nekrs_udf)
            build_bridge nekrs_udf examples/nekrs_gpucompress_udf.cpp examples/libnekrs_gpucompress_udf.so
            ;;
        all)
            build_bridge lammps_udf examples/lammps_gpucompress_udf.cpp examples/liblammps_gpucompress_udf.so
            build_bridge nekrs_udf examples/nekrs_gpucompress_udf.cpp examples/libnekrs_gpucompress_udf.so
            ;;
        *)
            die "Unknown bridge: $name (available: lammps_udf, nekrs_udf, all)"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# setup — Clone, patch, build a simulation
# ---------------------------------------------------------------------------
cmd_setup() {
    local sim="${1:-}"
    [ -z "$sim" ] && die "Usage: $0 setup <lammps|nyx|warpx|nekrs|vpic|all>"

    if [ "$sim" = "all" ]; then
        for s in "${SIMULATIONS[@]}"; do
            cmd_setup "$s"
        done
        return
    fi

    mkdir -p "$SIMS_DIR"
    setup_env

    case "$sim" in
        lammps) setup_lammps ;;
        nyx)    setup_nyx ;;
        warpx)  setup_warpx ;;
        nekrs)  setup_nekrs ;;
        vpic)   setup_vpic ;;
        *)      die "Unknown simulation: $sim" ;;
    esac
}

setup_lammps() {
    local src="${SIMS_DIR}/lammps"
    info "=== Setting up LAMMPS ==="

    # Bridge library required first
    if [ ! -f "${GPUC_DIR}/examples/liblammps_gpucompress_udf.so" ]; then
        info "Building LAMMPS bridge library first..."
        cmd_bridge lammps_udf
    fi

    # Clone
    if [ ! -d "$src" ]; then
        info "Cloning LAMMPS..."
        git clone https://github.com/lammps/lammps.git "$src"
    else
        info "LAMMPS source exists at $src"
    fi

    # Patch
    cd "$src"
    if [ ! -f src/KOKKOS/fix_gpucompress_kokkos.h ]; then
        info "Applying GPUCompress patch..."
        cp "${GPUC_DIR}/benchmarks/lammps/patches/fix_gpucompress_kokkos.h" src/KOKKOS/
        cp "${GPUC_DIR}/benchmarks/lammps/patches/fix_gpucompress_kokkos.cpp" src/KOKKOS/
        git apply "${GPUC_DIR}/benchmarks/lammps/patches/lammps-gpucompress.patch" || \
            warn "Patch may already be applied or needs manual resolution"
    else
        info "Patch already applied"
    fi

    local kokkos_arch="${LAMMPS_KOKKOS_ARCH:-AMPERE80}"

    # Build
    info "Building LAMMPS (Kokkos arch: ${kokkos_arch})..."
    mkdir -p build && cd build
    cmake ../cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DPKG_KOKKOS=ON \
        -DKokkos_ENABLE_CUDA=ON \
        -D"Kokkos_ARCH_${kokkos_arch}=ON" \
        -DBUILD_MPI=ON \
        -DCMAKE_CXX_FLAGS="-I${GPUC_DIR}/include -I${GPUC_DIR}/examples -I${HDF5_ROOT}/include -I/usr/local/cuda/include" \
        -DCMAKE_EXE_LINKER_FLAGS="-L${GPUC_DIR}/examples -llammps_gpucompress_udf -L${GPUC_DIR}/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L${HDF5_ROOT}/lib -lhdf5 -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,${GPUC_DIR}/build -Wl,-rpath,${GPUC_DIR}/examples -Wl,-rpath,${HDF5_ROOT}/lib"
    make -j"$(nproc)"

    ok "LAMMPS built: ${src}/build/lmp"
}

setup_nyx() {
    local src="${SIMS_DIR}/Nyx"
    info "=== Setting up Nyx ==="

    # Clone
    if [ ! -d "$src" ]; then
        info "Cloning Nyx (recursive)..."
        git clone --recursive https://github.com/AMReX-Astro/Nyx.git "$src"
        cd "$src"
        cd subprojects/amrex && git checkout development && cd ../..
    else
        info "Nyx source exists at $src"
        cd "$src"
    fi

    # Patch
    if ! grep -q "use_gpucompress" Source/Driver/Nyx.H 2>/dev/null; then
        info "Applying GPUCompress patch..."
        git apply "${GPUC_DIR}/benchmarks/nyx/patches/nyx-gpucompress.patch" || \
            warn "Patch may already be applied or needs manual resolution"
    else
        info "Patch already applied"
    fi

    # Build
    info "Building Nyx (Sedov blast wave target)..."
    mkdir -p build-gpucompress && cd build-gpucompress
    cmake -S .. -B . \
        -DCMAKE_BUILD_TYPE=Release \
        -DNyx_MPI=YES \
        -DNyx_OMP=NO \
        -DNyx_HYDRO=YES \
        -DNyx_HEATCOOL=NO \
        -DNyx_GPU_BACKEND=CUDA \
        -DAMReX_CUDA_ARCH=Ampere \
        -DAMReX_PRECISION=SINGLE \
        -DAMReX_PARTICLES_PRECISION=SINGLE \
        -DCMAKE_C_COMPILER="$(which gcc)" \
        -DCMAKE_CXX_COMPILER="$(which g++)" \
        -DCMAKE_CUDA_HOST_COMPILER="$(which g++)" \
        "-DCMAKE_CXX_FLAGS=-DAMREX_USE_GPUCOMPRESS -I${GPUC_DIR}/include -I${GPUC_DIR}/examples -I${HDF5_ROOT}/include" \
        "-DCMAKE_CUDA_FLAGS=-DAMREX_USE_GPUCOMPRESS -I${GPUC_DIR}/include -I${GPUC_DIR}/examples -I${HDF5_ROOT}/include" \
        "-DCMAKE_EXE_LINKER_FLAGS=-L${GPUC_DIR}/build -L${HDF5_ROOT}/lib -L${NVCOMP_PREFIX}/lib -Wl,--no-as-needed -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -lhdf5 -lnvcomp -Wl,--as-needed" \
        "-DCMAKE_SHARED_LINKER_FLAGS=-L${HDF5_ROOT}/lib -lhdf5"
    cmake --build . --target nyx_HydroTests -j"$(nproc)"

    ok "Nyx built: ${src}/build-gpucompress/Exec/HydroTests/nyx_HydroTests"
}

setup_warpx() {
    local src="${SIMS_DIR}/warpx"
    info "=== Setting up WarpX ==="

    # Clone
    if [ ! -d "$src" ]; then
        info "Cloning WarpX..."
        git clone https://github.com/BLAST-WarpX/warpx.git "$src"
    else
        info "WarpX source exists at $src"
    fi

    cd "$src"

    # Patch
    if ! grep -q "WarpX_GPUCOMPRESS" CMakeLists.txt 2>/dev/null; then
        info "Applying GPUCompress patch..."
        git apply "${GPUC_DIR}/benchmarks/warpx/patches/warpx-gpucompress.patch" || \
            warn "Patch may already be applied or needs manual resolution"
    else
        info "Patch already applied"
    fi

    # Build
    info "Building WarpX (3D CUDA)..."
    export CC=$(which gcc)
    export CXX=$(which g++)
    export CUDACXX=$(which nvcc)
    export CUDAHOSTCXX=$(which g++)

    mkdir -p build-gpucompress && cd build-gpucompress
    cmake -S .. -B . \
        -DCMAKE_BUILD_TYPE=Release \
        -DWarpX_COMPUTE=CUDA \
        -DWarpX_MPI=ON \
        -DWarpX_DIMS=3 \
        -DWarpX_GPUCOMPRESS=ON \
        "-DCMAKE_CXX_FLAGS=-I${GPUC_DIR}/include -I${GPUC_DIR}/examples -I${HDF5_ROOT}/include" \
        "-DCMAKE_EXE_LINKER_FLAGS=-L${GPUC_DIR}/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L${HDF5_ROOT}/lib -lhdf5 -L/usr/local/cuda/lib64 -lcudart"
    cmake --build . -j"$(nproc)"

    ok "WarpX built: ${src}/build-gpucompress/bin/warpx.3d"
}

setup_nekrs() {
    local src="${SIMS_DIR}/nekRS"
    local nekrs_home="${NEKRS_HOME:-$HOME/.local/nekrs}"
    local case_dir="${NEKRS_CASE_DIR:-$HOME/tgv_gpucompress}"
    info "=== Setting up nekRS ==="

    # Bridge library required first
    if [ ! -f "${GPUC_DIR}/examples/libnekrs_gpucompress_udf.so" ]; then
        info "Building nekRS bridge library first..."
        cmd_bridge nekrs_udf
    fi

    # Clone
    if [ ! -d "$src" ]; then
        info "Cloning nekRS..."
        git clone https://github.com/Nek5000/nekRS.git "$src"
    else
        info "nekRS source exists at $src"
    fi

    # Build nekRS (no source patches needed)
    cd "$src"
    info "Building nekRS (install to ${nekrs_home})..."
    mkdir -p build && cd build
    CC=mpicc CXX=mpic++ FC=mpif77 cmake -S .. -B . \
        -DCMAKE_INSTALL_PREFIX="${nekrs_home}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
    cmake --build . --target install -j"$(nproc)"

    # Copy case files
    info "Setting up TGV case directory at ${case_dir}..."
    mkdir -p "$case_dir"
    cp "${GPUC_DIR}/benchmarks/nekrs/patches/tgv.udf" "$case_dir/"
    cp "${GPUC_DIR}/benchmarks/nekrs/patches/tgv.par" "$case_dir/"
    cp "${GPUC_DIR}/benchmarks/nekrs/patches/udf.cmake" "$case_dir/"

    ok "nekRS installed: ${nekrs_home}/bin/nekrs-fp32"
    ok "Case dir ready: ${case_dir}"
    warn "First run JIT-compiles OCCA kernels (~5-10 min), cached thereafter"
}

setup_vpic() {
    local src="${SIMS_DIR}/vpic-kokkos"
    info "=== Setting up VPIC-Kokkos ==="

    # Clone
    if [ ! -d "$src" ]; then
        info "Cloning VPIC-Kokkos..."
        git clone https://github.com/lanl/vpic-kokkos.git "$src"
    else
        info "VPIC-Kokkos source exists at $src"
    fi

    # Build VPIC itself
    cd "$src"
    if [ ! -f build-compress/libvpic.so ] && [ ! -f build-compress/libvpic.a ]; then
        info "Building VPIC core with Kokkos CUDA..."
        cmake -S . -B build-compress \
            -DCMAKE_BUILD_TYPE=Release \
            -DENABLE_KOKKOS_CUDA=ON \
            -DKokkos_ARCH_AMPERE80=ON \
            -DCMAKE_CXX_COMPILER="$(pwd)/kokkos/bin/nvcc_wrapper"
        cmake --build build-compress -j"$(nproc)"
    fi

    # Build benchmark deck
    info "Building VPIC benchmark deck..."
    export VPIC_DIR="$src"
    export GPU_DIR="$GPUC_DIR"
    export VPIC_BUILD="${src}/build-compress"
    export HDF5_PREFIX="${HDF5_ROOT}"
    export NVCOMP_LIB="${NVCOMP_PREFIX}/lib"
    export CUDA_ARCH="${CUDA_ARCH}"
    bash "${GPUC_DIR}/benchmarks/vpic-kokkos/build_vpic_pm.sh"

    ok "VPIC deck built: ${GPUC_DIR}/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux"
}

# ---------------------------------------------------------------------------
# bench — Run benchmark for a simulation
# ---------------------------------------------------------------------------
cmd_bench() {
    local sim="${1:-}"
    [ -z "$sim" ] && die "Usage: $0 bench <lammps|nyx|warpx|nekrs|vpic|all>"

    if [ "$sim" = "all" ]; then
        for s in "${SIMULATIONS[@]}"; do
            cmd_bench "$s"
        done
        return
    fi

    setup_env
    mkdir -p "$RESULTS_BASE"

    case "$sim" in
        lammps) bench_lammps ;;
        nyx)    bench_nyx ;;
        warpx)  bench_warpx ;;
        nekrs)  bench_nekrs ;;
        vpic)   bench_vpic ;;
        *)      die "Unknown simulation: $sim" ;;
    esac
}

bench_lammps() {
    info "=== Running LAMMPS benchmark ==="
    local lmp_bin="${LMP_BIN:-${SIMS_DIR}/lammps/build/lmp}"
    [ -x "$lmp_bin" ] || die "LAMMPS binary not found: $lmp_bin (run: $0 setup lammps)"

    export LMP_BIN="$lmp_bin"
    export LMP_ATOMS="${LMP_ATOMS:-80}"
    export TIMESTEPS="${TIMESTEPS:-10}"
    export SIM_INTERVAL="${SIM_INTERVAL:-50}"
    export WARMUP_STEPS="${WARMUP_STEPS:-100}"
    export CHUNK_MB="${CHUNK_MB:-4}"
    export ERROR_BOUND="${ERROR_BOUND:-0.0}"
    export POLICIES="${POLICIES:-balanced,ratio,speed}"
    export RESULTS_DIR="${RESULTS_BASE}/lammps"
    export GPUC_DIR

    mkdir -p "$RESULTS_DIR"
    bash "${GPUC_DIR}/benchmarks/lammps/run_lammps_vpic_benchmark.sh"

    ok "LAMMPS results in ${RESULTS_DIR}/"
}

bench_nyx() {
    info "=== Running Nyx benchmark ==="
    local nyx_bin="${NYX_BIN:-${SIMS_DIR}/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests}"
    [ -x "$nyx_bin" ] || die "Nyx binary not found: $nyx_bin (run: $0 setup nyx)"

    export NYX_BIN="$nyx_bin"
    export NYX_NCELL="${NYX_NCELL:-128}"
    export NYX_MAX_STEP="${NYX_MAX_STEP:-200}"
    export NYX_PLOT_INT="${NYX_PLOT_INT:-10}"
    export CHUNK_MB="${CHUNK_MB:-4}"
    export ERROR_BOUND="${ERROR_BOUND:-0.0}"
    export POLICIES="${POLICIES:-balanced,ratio,speed}"
    export NO_RANKING="${NO_RANKING:-1}"
    export RESULTS_DIR="${RESULTS_BASE}/nyx"
    export GPUC_DIR

    mkdir -p "$RESULTS_DIR"
    bash "${GPUC_DIR}/benchmarks/nyx/run_nyx_benchmark.sh"

    ok "Nyx results in ${RESULTS_DIR}/"
}

bench_warpx() {
    info "=== Running WarpX benchmark ==="
    local warpx_bin="${WARPX_BIN:-${SIMS_DIR}/warpx/build-gpucompress/bin/warpx.3d}"
    [ -x "$warpx_bin" ] || die "WarpX binary not found: $warpx_bin (run: $0 setup warpx)"

    export WARPX_BIN="$warpx_bin"
    export WARPX_INPUTS="${WARPX_INPUTS:-${SIMS_DIR}/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}"
    export WARPX_NCELL="${WARPX_NCELL:-32 32 256}"
    export WARPX_MAX_STEP="${WARPX_MAX_STEP:-200}"
    export WARPX_DIAG_INT="${WARPX_DIAG_INT:-10}"
    export CHUNK_MB="${CHUNK_MB:-4}"
    export ERROR_BOUND="${ERROR_BOUND:-0.01}"
    export POLICIES="${POLICIES:-balanced,ratio}"
    export RESULTS_DIR="${RESULTS_BASE}/warpx"
    export GPUC_DIR

    mkdir -p "$RESULTS_DIR"
    bash "${GPUC_DIR}/benchmarks/warpx/run_warpx_benchmark.sh"

    ok "WarpX results in ${RESULTS_DIR}/"
}

bench_nekrs() {
    info "=== Running nekRS benchmark ==="
    local nekrs_home="${NEKRS_HOME:-$HOME/.local/nekrs}"
    local nekrs_bin="${nekrs_home}/bin/nekrs-fp32"
    [ -x "$nekrs_bin" ] || die "nekRS binary not found: $nekrs_bin (run: $0 setup nekrs)"

    export NEKRS_HOME="$nekrs_home"
    export CASE_DIR="${NEKRS_CASE_DIR:-$HOME/tgv_gpucompress}"
    export NP="${NP:-2}"
    export NUM_STEPS="${NUM_STEPS:-20}"
    export CHECKPOINT_INT="${CHECKPOINT_INT:-10}"
    export USE_FP32="${USE_FP32:-1}"
    export POLICIES="${POLICIES:-balanced,ratio,speed}"
    export RESULTS_DIR="${RESULTS_BASE}/nekrs"
    export GPUC_DIR

    mkdir -p "$RESULTS_DIR"
    bash "${GPUC_DIR}/benchmarks/nekrs/run_nekrs_benchmark.sh"

    ok "nekRS results in ${RESULTS_DIR}/"
}

bench_vpic() {
    info "=== Running VPIC benchmark ==="
    local vpic_deck="${GPUC_DIR}/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux"
    [ -x "$vpic_deck" ] || die "VPIC deck not found: $vpic_deck (run: $0 setup vpic)"

    export VPIC_NX="${VPIC_NX:-200}"
    export VPIC_TIMESTEPS="${VPIC_TIMESTEPS:-10}"
    export VPIC_WARMUP_STEPS="${VPIC_WARMUP_STEPS:-100}"
    export VPIC_SIM_INTERVAL="${VPIC_SIM_INTERVAL:-10}"
    export VPIC_CHUNK_MB="${VPIC_CHUNK_MB:-${CHUNK_MB:-4}}"
    export VPIC_POLICIES="${VPIC_POLICIES:-${POLICIES:-balanced,ratio,speed}}"
    export RESULTS_DIR="${RESULTS_BASE}/vpic"

    mkdir -p "$RESULTS_DIR"
    cd "$RESULTS_DIR"
    mpirun -np 1 "$vpic_deck"

    ok "VPIC results in ${RESULTS_DIR}/"
}

# ---------------------------------------------------------------------------
# env — Print runtime environment
# ---------------------------------------------------------------------------
cmd_env() {
    setup_env
    echo "# GPUCompress Runtime Environment"
    echo "# Source this file: eval \$($0 env)"
    echo "export GPUC_DIR=\"${GPUC_DIR}\""
    echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}\""
    echo "export GPUCOMPRESS_WEIGHTS=\"${GPUCOMPRESS_WEIGHTS}\""
    echo "export GPUCOMPRESS_ALGO=\"${GPUCOMPRESS_ALGO:-auto}\""
    echo "export GPUCOMPRESS_POLICY=\"${GPUCOMPRESS_POLICY:-ratio}\""
    echo "export GPUCOMPRESS_VERIFY=\"${GPUCOMPRESS_VERIFY:-0}\""
    echo "export SIMS_DIR=\"${SIMS_DIR}\""
}

# ---------------------------------------------------------------------------
# list — List available simulations
# ---------------------------------------------------------------------------
cmd_list() {
    echo "Available simulations:"
    echo ""
    echo "  lammps    Molecular dynamics (Kokkos GPU, ~2M atoms)"
    echo "  nyx       Cosmological hydro (AMReX GPU, Sedov blast)"
    echo "  warpx     Laser wakefield PIC (AMReX GPU, E/B fields)"
    echo "  nekrs     Spectral element CFD (OCCA GPU, Taylor-Green vortex)"
    echo "  vpic      Plasma PIC (Kokkos GPU, custom deck)"
    echo ""
    echo "Usage:"
    echo "  $0 setup <sim>    Clone, patch, and build"
    echo "  $0 bench <sim>    Run benchmark"
    echo "  $0 setup all      Build all simulations"
    echo "  $0 bench all      Run all benchmarks"
    echo "  $0 all            Build core + all sims + run all benchmarks"
}

# ---------------------------------------------------------------------------
# all — Full pipeline
# ---------------------------------------------------------------------------
cmd_all() {
    info "=== Full GPUCompress deployment pipeline ==="
    cmd_build
    cmd_bridge all
    cmd_setup all
    cmd_bench all
    ok "All done. Results in ${RESULTS_BASE}/"
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
main() {
    local cmd="${1:-help}"
    shift || true

    case "$cmd" in
        build)  cmd_build "$@" ;;
        bridge) cmd_bridge "$@" ;;
        setup)  cmd_setup "$@" ;;
        bench)  cmd_bench "$@" ;;
        env)    cmd_env "$@" ;;
        list)   cmd_list "$@" ;;
        all)    cmd_all "$@" ;;
        help|-h|--help)
            echo "GPUCompress Deployment Driver"
            echo ""
            echo "Commands:"
            echo "  build              Build GPUCompress core library"
            echo "  bridge <name>      Build bridge library (lammps_udf|nekrs_udf|all)"
            echo "  setup <sim>        Clone, patch, build a simulation"
            echo "  bench <sim>        Run benchmark for a simulation"
            echo "  env                Print runtime environment (eval-able)"
            echo "  list               List available simulations"
            echo "  all                Full pipeline: build + setup all + bench all"
            echo ""
            echo "Environment variables:"
            echo "  CUDA_ARCH=80           GPU arch (80=A100, 70=V100, 90=H100)"
            echo "  BUILD_TYPE=Release     CMake build type"
            echo "  HDF5_ROOT              HDF5 install prefix [/tmp/hdf5-install]"
            echo "  NVCOMP_PREFIX          nvcomp install prefix [/tmp]"
            echo "  SIMS_DIR              Where sims are cloned [\$HOME/sims]"
            echo "  RESULTS_BASE          Results output dir [results/]"
            echo "  POLICIES              Comma-sep policies [balanced,ratio,speed]"
            echo ""
            echo "Examples:"
            echo "  $0 build                          # Build GPUCompress"
            echo "  $0 setup lammps                   # Clone+patch+build LAMMPS"
            echo "  $0 bench lammps                   # Run LAMMPS benchmark"
            echo "  CUDA_ARCH=90 $0 all               # Full pipeline on H100"
            echo "  LMP_ATOMS=100 $0 bench lammps     # Larger LAMMPS problem"
            ;;
        *)
            die "Unknown command: $cmd (try: $0 help)"
            ;;
    esac
}

main "$@"
