"""
Nyx benchmark package for GPUCompress.
Runs a Sedov blast wave cosmological hydro simulation with
GPU-accelerated compression via the HDF5 VOL connector.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo, LocalExecInfo
from jarvis_cd.shell.process import Rm
import os


class GpucompressNyx(Application):
    """
    Deploy and run the Nyx GPUCompress benchmark.
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
                'name': 'nyx_bin',
                'msg': 'Path to nyx_HydroTests binary',
                'type': str,
                'default': '/opt/sims/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests',
            },
            {
                'name': 'inputs',
                'msg': 'Path to Nyx inputs file',
                'type': str,
                'default': '/opt/sims/Nyx/Exec/HydroTests/inputs.3d.sph.sedov',
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
                'msg': 'Grid cells per dimension',
                'type': int,
                'default': 128,
            },
            {
                'name': 'max_step',
                'msg': 'Maximum simulation steps',
                'type': int,
                'default': 200,
            },
            {
                'name': 'plot_int',
                'msg': 'Steps between plot file outputs',
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
                'default': 'balanced,ratio,speed',
            },
            {
                'name': 'error_bound',
                'msg': 'Lossy error bound (0.0 = lossless)',
                'type': float,
                'default': 0.0,
            },
            {
                'name': 'results_dir',
                'msg': 'Output directory for results',
                'type': str,
                'default': '/tmp/gpucompress_nyx_results',
            },
        ]

    def _configure(self, **kwargs):
        pass

    def start(self):
        gpuc = self.config['gpucompress_dir']
        ncell = self.config['ncell']
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

        cmd = (f'{self.config["nyx_bin"]} {self.config["inputs"]}'
               f' amr.n_cell={ncell} {ncell} {ncell}'
               f' max_step={self.config["max_step"]}'
               f' amr.plot_int={self.config["plot_int"]}'
               f' nyx.write_hdf5=1'
               f' nyx.use_gpucompress=1'
               f' nyx.gpucompress_weights={self.config["weights"]}')

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
