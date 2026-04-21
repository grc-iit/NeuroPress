"""
gpucompress_base — installs HDF5 + nvcomp and builds the GPUCompress
library inside a Jarvis build container. Workload packages
(gpucompress_nyx, gpucompress_vpic, ...) chain after this one; they
inherit the committed build image (HDF5 at /usr/local, nvcomp at
/opt/nvcomp, GPUCompress at /opt/GPUCompress) and add their own binaries.

Build-only — no start/stop semantics. Must precede workload packages in
the pipeline YAML.
"""
from jarvis_cd.core.pkg import Library


class GpucompressBase(Library):

    def _configure_menu(self):
        return [
            {'name': 'hdf5_version',
             'msg': 'HDF5 release version',
             'type': str, 'default': '2.0.0'},
            {'name': 'nvcomp_version',
             'msg': 'nvcomp redist version (matches NVIDIA download URL)',
             'type': str, 'default': '5.1.0.21_cuda12'},
            {'name': 'cuda_arch',
             'msg': 'CUDA compute capability (80=A100, 89=RTX4070, 90=H100)',
             'type': int, 'default': 80},
            {'name': 'gpucompress_repo',
             'msg': 'GPUCompress git clone URL',
             'type': str,
             'default': 'https://github.com/grc-iit/GPUCompress.git'},
            {'name': 'gpucompress_ref',
             'msg': 'GPUCompress git ref (branch | tag | commit)',
             'type': str, 'default': 'main'},
            {'name': 'deploy_base',
             'msg': 'Base image for the deploy stage',
             'type': str,
             'default': 'nvidia/cuda:12.8.0-runtime-ubuntu24.04'},
        ]

    def _build_phase(self):
        if self.config.get('deploy_mode') != 'container':
            return None
        content = self._read_build_script('build.sh', {
            'HDF5_VERSION':     self.config['hdf5_version'],
            'NVCOMP_VERSION':   self.config['nvcomp_version'],
            'CUDA_ARCH':        str(self.config['cuda_arch']),
            'GPUCOMPRESS_REPO': self.config['gpucompress_repo'],
            'GPUCOMPRESS_REF':  self.config['gpucompress_ref'],
        })
        suffix = (f"cuda-{self.config['cuda_arch']}-"
                  f"hdf5-{self.config['hdf5_version']}")
        return content, suffix

    def _build_deploy_phase(self):
        if self.config.get('deploy_mode') != 'container':
            return None
        suffix = getattr(self, '_build_suffix', '')
        content = self._read_dockerfile('Dockerfile.deploy', {
            'BUILD_IMAGE': self.build_image_name(),
            'DEPLOY_BASE': self.config['deploy_base'],
        })
        return content, suffix
