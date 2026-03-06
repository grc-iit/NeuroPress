# GPUCompress Requirements and Installation Guide

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Linux x86_64 (Ubuntu 20.04+, RHEL 9+) | Ubuntu 22.04/24.04 x86_64 |
| GPU | NVIDIA GPU (Compute Capability >= 7.0) | NVIDIA A100 (sm_80) / H100 (sm_90) |
| NVIDIA Driver | >= 525.60.13 | Latest stable |
| CUDA Toolkit | >= 12.0 | 12.6+ |
| RAM | 8 GB | 16 GB+ |
| Disk | 2 GB free | 5 GB+ free |

## Dependencies

### Required

| Package | Version | Install Location | Purpose |
|---------|---------|-----------------|---------|
| CUDA Toolkit | >= 12.0 | system | GPU compilation and runtime |
| nvcomp | 5.1.0 | `/tmp/include`, `/tmp/lib` | GPU compression library |
| cuFile | (included with CUDA) | CUDA install dir | GPUDirect Storage |
| HDF5 | >= 2.0.0 | `/tmp/hdf5-install` | HDF5 VOL connector (GPU-native I/O) |
| cmake | >= 3.18 | system | Build system |
| g++ | >= 9.0 | system | C++ compiler |

### Optional

| Package | Purpose |
|---------|---------|
| HDF5 (system, any version) | HDF5 filter plugin (`libH5Zgpucompress.so`) |
| NVIDIA Nsight Systems | Performance profiling |
| GDS drivers | Direct GPU-to-storage I/O |

### Dependency Notes

- **nvcomp 5.1.0** is the GPU compression backend. Headers go to `/tmp/include/`, libraries to `/tmp/lib/`.
- **HDF5 >= 2.0.0** is required for the VOL connector (`libH5VLgpucompress.so`), which provides GPU-native `H5Dwrite`/`H5Dread` interception. The VOL API uses types (`H5VL_attr_get_args_t`, `H5VL_optional_args_t`, `H5VL_native_dataset_optional_args_t`) and the 6-argument `H5Dread_chunk` signature that are only available in HDF5 2.x. System HDF5 1.12/1.14 is **not sufficient** for the VOL connector.
- A **system HDF5** (any version with `find_package(HDF5)` support) is enough to build the filter plugin (`libH5Zgpucompress.so`) and the C-based test/benchmark targets.
- **cuFile** is typically located under the CUDA install directory (e.g., `/opt/nvidia/cuda-12.8/targets/x86_64-linux/lib/`). If cmake cannot find it, add its directory to `link_directories` in `CMakeLists.txt` or pass `-DCMAKE_LIBRARY_PATH=<path>`.

## Quick Installation

Run the automated installation script:

```bash
cd GPUCompress
chmod +x scripts/install_dependencies.sh
./scripts/install_dependencies.sh
```

This script will:
1. Verify system requirements (CUDA, GPU, compiler)
2. Install cmake and build tools (if needed)
3. Download and install nvcomp 5.1.0 to `/tmp/`
4. Download, build, and install HDF5 2.0.0 to `/tmp/hdf5-install/`
5. Build the project with the correct CUDA architecture flags

## Manual Installation

### Step 1: Install CUDA Toolkit

If not already installed, download from:
https://developer.nvidia.com/cuda-downloads

Verify installation:
```bash
nvcc --version
nvidia-smi
```

### Step 2: Install Build Tools

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential curl xz-utils
```

RHEL/CentOS:
```bash
sudo yum install -y cmake gcc-c++ make curl xz
```

### Step 3: Install nvcomp

Download and extract nvcomp 5.1.0:

```bash
# Create directories
mkdir -p /tmp/include /tmp/lib

# Download nvcomp
curl -L -o /tmp/nvcomp.tar.xz \
  "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.1.0.21_cuda12-archive.tar.xz"

# Extract
cd /tmp
tar -xf nvcomp.tar.xz

# Install
cp -r nvcomp-linux-x86_64-5.1.0.21_cuda12-archive/include/* /tmp/include/
cp -r nvcomp-linux-x86_64-5.1.0.21_cuda12-archive/lib/* /tmp/lib/
```

### Step 4: Build and Install HDF5 2.0.0

The HDF5 VOL connector requires HDF5 >= 2.0.0. Build from source:

```bash
# Download HDF5 2.0.0
curl -L -o /tmp/hdf5-2.0.0.tar.gz \
  "https://github.com/HDFGroup/hdf5/releases/download/2.0.0/hdf5-2.0.0.tar.gz"

# Extract
cd /tmp
tar xzf hdf5-2.0.0.tar.gz

# Build (C library only, minimal config for speed)
mkdir -p /tmp/hdf5-build && cd /tmp/hdf5-build
cmake /tmp/hdf5-2.0.0 \
  -DCMAKE_INSTALL_PREFIX=/tmp/hdf5-install \
  -DCMAKE_BUILD_TYPE=Release \
  -DHDF5_BUILD_TOOLS=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF \
  -DBUILD_TESTING=OFF \
  -DHDF5_BUILD_CPP_LIB=OFF \
  -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_BUILD_JAVA=OFF \
  -DHDF5_BUILD_HL_LIB=ON

make -j$(nproc)
make install
```

Verify:
```bash
ls /tmp/hdf5-install/lib/libhdf5.so /tmp/hdf5-install/include/hdf5.h
```

### Step 5: Build the Project

```bash
cd GPUCompress
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make -j$(nproc)
```

**Important**: The `-DCMAKE_CUDA_ARCHITECTURES=80` flag is required when building
on systems with Cray Programming Environment (PE) wrappers. The Cray PE can
override the default CUDA architecture to sm_52, which does not support
`atomicAdd(double*, double)` (requires sm_60+). Set this to match your target GPU:
- A100: `80`
- H100: `90`
- V100: `70`

If cuFile is not found during linking, locate it and add the path:
```bash
# Find cuFile
find /opt/nvidia /usr/local/cuda -name "libcufile.so" 2>/dev/null

# If needed, add to CMakeLists.txt link_directories or pass:
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 \
         -DCMAKE_LIBRARY_PATH="/opt/nvidia/cuda-12.8/targets/x86_64-linux/lib"
```

### Step 6: Set Up Environment

Before running executables, set the library path:

```bash
# Option A: Source the setup script
source scripts/setup_env.sh

# Option B: Set manually
export LD_LIBRARY_PATH=/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
```

For permanent setup, add to `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH=/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Build Outputs

A successful build produces these targets:

| Target | Type | Description |
|--------|------|-------------|
| `libgpucompress.so` | Shared library | Core C API (compression, NN, stats, preprocessing) |
| `gpu_compress` | Executable | CLI compression tool with GDS support |
| `gpu_decompress` | Executable | CLI decompression tool with GDS support |
| `libH5Zgpucompress.so` | Shared library | HDF5 filter plugin (requires system HDF5) |
| `libH5VLgpucompress.so` | Shared library | HDF5 VOL connector for GPU-native I/O (requires HDF5 2.x) |
| `test_*` / `benchmark_*` | Executables | Test suites and benchmarks |

## Verification

Test the installation:

```bash
# Check core executables exist
ls -la build/gpu_compress build/gpu_decompress

# Check libraries
ls -la build/libgpucompress.so build/libH5Zgpucompress.so build/libH5VLgpucompress.so

# Show help
./build/gpu_compress --help

# Run quantization round-trip test
./build/test_quantization_roundtrip

# Run NN pipeline test
./build/test_nn_pipeline
```

## Troubleshooting

### nvcomp library not found at runtime

```
error while loading shared libraries: libnvcomp.so.5: cannot open shared object file
```

**Solution**: Set LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=/tmp/lib:$LD_LIBRARY_PATH
```

### HDF5 2.x library not found at runtime

```
error while loading shared libraries: libhdf5.so.320: cannot open shared object file
```

**Solution**: Add the HDF5 2.0 install path:
```bash
export LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
```

### CUDA not found during cmake

```
Could not find CUDAToolkit
```

**Solution**: Ensure CUDA is in PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### atomicAdd(double*, double) errors on Cray PE systems

```
error: no instance of overloaded function "atomicAdd" matches the argument list
  argument types are: (double *, double)
```

**Cause**: The Cray Programming Environment compiler wrapper overrides
`CMAKE_CUDA_ARCHITECTURES` to sm_52, which does not support native
`atomicAdd(double*, double)` (available on sm_60+).

**Solution**: Explicitly pass the architecture at configure time:
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
```

### cuFile (libcufile.so) not found during linking

```
cannot find -lcufile: No such file or directory
```

**Cause**: cuFile is bundled with CUDA but may not be in the default library search path.

**Solution**: Find the library and add its directory to CMakeLists.txt `link_directories`:
```bash
find /opt/nvidia /usr/local/cuda -name "libcufile.so" 2>/dev/null
# e.g., /opt/nvidia/cuda-12.8/targets/x86_64-linux/lib/
```
Then add that path to `link_directories(...)` in `CMakeLists.txt`, or pass
`-DCMAKE_LIBRARY_PATH=<path>` at configure time.

### GDS/cuFile runtime errors

If GPUDirect Storage is not available, the application will fall back to standard I/O.
For full GDS support:
1. Ensure filesystem supports O_DIRECT (ext4, xfs)
2. Install GDS drivers: https://docs.nvidia.com/gpudirect-storage/

Check GDS status:
```bash
/usr/local/cuda/gds/tools/gdscheck -p
```

### Build errors with nvcomp API

The code is tested with nvcomp 5.1.0. If using a different version, API changes
may cause compilation errors. Check `src/compression/compression_factory.cpp` for
the compression manager initialization code.

### HDF5 version conflicts (libhdf5.so.200 vs libhdf5.so.320)

```
warning: libhdf5.so.200, needed by libH5Zgpucompress.so, may conflict with libhdf5.so.320
```

This warning occurs when the H5Z filter plugin links against the system HDF5
(e.g., 1.12) while VOL targets link against HDF5 2.0. This is expected and
generally harmless at link time. At runtime, ensure `LD_LIBRARY_PATH` lists
`/tmp/hdf5-install/lib` **before** system library directories so that VOL-based
executables pick up HDF5 2.x.

## nvcomp Supported Algorithms

| Algorithm | Best For | Speed | Ratio |
|-----------|----------|-------|-------|
| lz4 | General purpose | Very Fast | Medium |
| snappy | Speed critical | Fastest | Low |
| deflate | Compatibility | Slow | High |
| gdeflate | GPU-optimized gzip | Slow | High |
| zstd | Best ratio | Medium | Very High |
| ans | Numerical data | Medium | High |
| cascaded | Floating-point | Medium | Very High |
| bitcomp | Scientific data | Fast | High |

## Project Structure

```
GPUCompress/
├── include/
│   ├── gpucompress.h                # Public C API header
│   └── gpucompress_hdf5_vol.h       # VOL connector public header
├── src/
│   ├── cli/
│   │   ├── compress.cpp             # gpu_compress entry point
│   │   └── decompress.cpp           # gpu_decompress entry point
│   ├── compression/
│   │   ├── compression_factory.cpp  # nvcomp algorithm factory
│   │   ├── compression_factory.hpp
│   │   └── compression_header.h     # 64-byte header format
│   ├── api/
│   │   └── gpucompress_api.cpp      # C API implementation
│   ├── nn/
│   │   └── nn_gpu.cu               # NN inference on GPU
│   ├── stats/
│   │   ├── entropy_kernel.cu        # GPU entropy calculation
│   │   ├── stats_kernel.cu          # Stats pipeline kernels
│   │   └── stats_cpu.cpp            # CPU stats fallback
│   ├── preprocessing/
│   │   ├── byte_shuffle_kernels.cu  # Byte shuffle CUDA kernels
│   │   └── quantization_kernels.cu  # Linear quantization kernels
│   └── hdf5/
│       ├── H5Zgpucompress.c         # HDF5 filter plugin
│       └── H5VLgpucompress.cu       # HDF5 VOL connector (GPU-native I/O)
├── examples/
│   ├── sample_gpu_compress_roundtrip.cu  # Minimal lossless round-trip
│   ├── demo_gpu_pipeline.cu              # Full write/inspect/read demo
│   ├── nn_vol_demo.cu                    # NN + VOL demo
│   └── hdf5_chunk_verify.cu             # Chunk/header verification
├── neural_net/                      # NN training and inference
├── eval/                            # Evaluation scripts and tools
├── syntheticGeneration/             # Synthetic data generator
├── scripts/
│   ├── install_dependencies.sh      # Automated setup
│   └── setup_env.sh                 # Environment setup
├── tests/                           # Test suites and benchmarks
├── build/                           # Build output (created by cmake)
├── CMakeLists.txt
├── REQUIREMENTS.md                  # This file
└── README.md
```

## References

- [nvcomp Documentation](https://docs.nvidia.com/cuda/nvcomp/)
- [nvcomp Downloads](https://developer.nvidia.com/nvcomp-download)
- [HDF5 Releases (GitHub)](https://github.com/HDFGroup/hdf5/releases)
- [HDF5 VOL Connector Guide](https://docs.hdfgroup.org/hdf5/develop/group___h5_v_l.html)
- [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
