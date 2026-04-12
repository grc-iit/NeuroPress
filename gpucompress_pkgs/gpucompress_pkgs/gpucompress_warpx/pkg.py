"""
WarpX benchmark package for GPUCompress.
Runs a laser wakefield acceleration simulation with GPU-accelerated
compression via the HDF5 VOL connector.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo, LocalExecInfo
from jarvis_cd.shell.process import Rm
import os


class GpucompressWarpx(Application):
    """
    Deploy and run the WarpX GPUCompress benchmark.
    """

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {
                'name': 'nprocs',
                'msg': 'Number of MPI processes',
                'type': int,
                'default': 1,
            },
            {
                'name': 'ppn',
                'msg': 'Processes per node',
                'type': int,
                'default': 1,
            },
            {
                'name': 'warpx_bin',
                'msg': 'Path to WarpX binary',
                'type': str,
                'default': '/opt/sims/warpx/build-gpucompress/bin/warpx.3d',
            },
            {
                'name': 'inputs',
                'msg': 'Path to WarpX inputs file',
                'type': str,
                'default': '/opt/sims/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d',
            },
            {
                'name': 'gpucompress_dir',
                'msg': 'Path to GPUCompress installation',
                'type': str,
                'default': '/opt/GPUCompress',
            },
            {
                'name': 'nvcomp_lib',
                'msg': 'Path to nvcomp lib directory',
                'type': str,
                'default': '/tmp/lib',
            },
            {
                'name': 'hdf5_lib',
                'msg': 'Path to HDF5 lib directory',
                'type': str,
                'default': '/tmp/hdf5-install/lib',
            },
            {
                'name': 'weights',
                'msg': 'Path to NN weights file (.nnwt)',
                'type': str,
                'default': '/opt/GPUCompress/neural_net/weights/model.nnwt',
            },
            {
                'name': 'ncell',
                'msg': 'Grid cells (space-separated, e.g. "32 32 256")',
                'type': str,
                'default': '32 32 256',
            },
            {
                'name': 'max_step',
                'msg': 'Maximum simulation steps',
                'type': int,
                'default': 200,
            },
            {
                'name': 'diag_int',
                'msg': 'Steps between diagnostic outputs',
                'type': int,
                'default': 10,
            },
            {
                'name': 'chunk_mb',
                'msg': 'HDF5 chunk size in MB',
                'type': int,
                'default': 4,
            },
            {
                'name': 'policies',
                'msg': 'Comma-separated NN cost model policies',
                'type': str,
                'default': 'balanced,ratio',
            },
            {
                'name': 'error_bound',
                'msg': 'Lossy error bound (0.0 = lossless)',
                'type': float,
                'default': 0.01,
            },
            {
                'name': 'results_dir',
                'msg': 'Output directory for results',
                'type': str,
                'default': '/tmp/gpucompress_warpx_results',
            },
        ]

    def _configure(self, **kwargs):
        pass

    def start(self):
        gpuc = self.config['gpucompress_dir']
        os.makedirs(self.config['results_dir'], exist_ok=True)

        env = dict(os.environ)
        lib_paths = [
            f'{gpuc}/build',
            self.config['nvcomp_lib'],
            self.config['hdf5_lib'],
        ]
        ld = ':'.join(lib_paths)
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f'{ld}:{env["LD_LIBRARY_PATH"]}'
        else:
            env['LD_LIBRARY_PATH'] = ld
        env['GPUCOMPRESS_WEIGHTS'] = self.config['weights']
        env['HDF5_PLUGIN_PATH'] = f'{gpuc}/build'

        cmd = (f'{self.config["warpx_bin"]} {self.config["inputs"]}'
               f' amr.n_cell="{self.config["ncell"]}"'
               f' max_step={self.config["max_step"]}'
               f' diagnostics.diag1.intervals={self.config["diag_int"]}')

        if self.config['nprocs'] > 1:
            Exec(cmd,
                 MpiExecInfo(nprocs=self.config['nprocs'],
                             ppn=self.config['ppn'],
                             hostfile=self.hostfile,
                             env=env,
                             cwd=self.config['results_dir'])).run()
        else:
            Exec(cmd,
                 LocalExecInfo(env=env,
                               cwd=self.config['results_dir'])).run()

    def stop(self):
        pass

    def clean(self):
        if self.config['results_dir'] and os.path.isdir(self.config['results_dir']):
            Rm(self.config['results_dir']).run()
