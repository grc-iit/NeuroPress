"""
VPIC-Kokkos benchmark package for GPUCompress.
Runs a Harris sheet reconnection PIC simulation with GPU-accelerated
compression via the HDF5 VOL connector.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo, LocalExecInfo
from jarvis_cd.shell.process import Rm
import os


class GpucompressVpic(Application):
    """
    Deploy and run the VPIC-Kokkos GPUCompress benchmark.
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
                'name': 'vpic_deck',
                'msg': 'Path to vpic_benchmark_deck.Linux binary',
                'type': str,
                'default': '/opt/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux',
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
                'name': 'nx',
                'msg': 'Grid cells per dimension',
                'type': int,
                'default': 200,
            },
            {
                'name': 'chunk_mb',
                'msg': 'HDF5 chunk size in MB',
                'type': int,
                'default': 4,
            },
            {
                'name': 'timesteps',
                'msg': 'Number of benchmark write cycles',
                'type': int,
                'default': 20,
            },
            {
                'name': 'warmup_steps',
                'msg': 'Physics warmup steps before first write',
                'type': int,
                'default': 500,
            },
            {
                'name': 'sim_interval',
                'msg': 'Physics steps between writes',
                'type': int,
                'default': 190,
            },
            {
                'name': 'policies',
                'msg': 'Comma-separated NN cost model policies',
                'type': str,
                'default': 'balanced,ratio,speed',
            },
            {
                'name': 'error_bound',
                'msg': 'Lossy error bound (0.0 = lossless)',
                'type': float,
                'default': 0.0,
            },
            {
                'name': 'sgd_lr',
                'msg': 'SGD learning rate',
                'type': float,
                'default': 0.2,
            },
            {
                'name': 'sgd_mape',
                'msg': 'MAPE threshold for SGD firing',
                'type': float,
                'default': 0.10,
            },
            {
                'name': 'explore_k',
                'msg': 'Number of exploration alternatives',
                'type': int,
                'default': 4,
            },
            {
                'name': 'explore_thresh',
                'msg': 'Exploration error threshold',
                'type': float,
                'default': 0.20,
            },
            {
                'name': 'results_dir',
                'msg': 'Output directory for results',
                'type': str,
                'default': '/tmp/gpucompress_vpic_results',
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
        env['VPIC_NX'] = str(self.config['nx'])
        env['VPIC_CHUNK_MB'] = str(self.config['chunk_mb'])
        env['VPIC_TIMESTEPS'] = str(self.config['timesteps'])
        env['VPIC_WARMUP_STEPS'] = str(self.config['warmup_steps'])
        env['VPIC_SIM_INTERVAL'] = str(self.config['sim_interval'])
        env['VPIC_POLICIES'] = self.config['policies']
        env['VPIC_ERROR_BOUND'] = str(self.config['error_bound'])
        env['VPIC_LR'] = str(self.config['sgd_lr'])
        env['VPIC_MAPE_THRESHOLD'] = str(self.config['sgd_mape'])
        env['VPIC_EXPLORE_K'] = str(self.config['explore_k'])
        env['VPIC_EXPLORE_THRESH'] = str(self.config['explore_thresh'])
        env['VPIC_RESULTS_DIR'] = self.config['results_dir']

        cmd = self.config['vpic_deck']

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
