"""
Gray-Scott benchmark package for GPUCompress.
Runs a reaction-diffusion PDE simulation on GPU with
compression via the HDF5 VOL connector.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo, LocalExecInfo
from jarvis_cd.shell.process import Rm
import os


class GpucompressGrayscott(Application):
    """
    Deploy and run the Gray-Scott GPUCompress benchmark.
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
                'name': 'grid_size',
                'msg': 'Grid dimension L (L^3 floats per field)',
                'type': int,
                'default': 512,
            },
            {
                'name': 'steps',
                'msg': 'Simulation steps per timestep',
                'type': int,
                'default': 1000,
            },
            {
                'name': 'chunk_mb',
                'msg': 'HDF5 chunk size in MB',
                'type': int,
                'default': 8,
            },
            {
                'name': 'feed_rate',
                'msg': 'Gray-Scott feed rate F',
                'type': float,
                'default': 0.04,
            },
            {
                'name': 'kill_rate',
                'msg': 'Gray-Scott kill rate k',
                'type': float,
                'default': 0.06075,
            },
            {
                'name': 'phase',
                'msg': 'Benchmark phase (e.g. nn-rl+exp50)',
                'type': str,
                'default': 'nn-rl+exp50',
            },
            {
                'name': 'policies',
                'msg': 'Comma-separated NN cost model policies',
                'type': str,
                'default': 'balanced,ratio,speed',
            },
            {
                'name': 'results_dir',
                'msg': 'Output directory for results',
                'type': str,
                'default': '/tmp/gpucompress_grayscott_results',
            },
        ]

    def _configure(self, **kwargs):
        pass

    def start(self):
        gpuc = self.config['gpucompress_dir']
        benchmark_bin = f'{gpuc}/build/grayscott_benchmark_pm'
        os.makedirs(self.config['results_dir'], exist_ok=True)

        # Build env from config paths — portable across systems
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

        cmd = (f'{benchmark_bin} {self.config["weights"]}'
               f' --L {self.config["grid_size"]}'
               f' --steps {self.config["steps"]}'
               f' --chunk-mb {self.config["chunk_mb"]}'
               f' --F {self.config["feed_rate"]}'
               f' --k {self.config["kill_rate"]}'
               f' --phase {self.config["phase"]}')

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
