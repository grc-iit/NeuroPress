"""
nekRS benchmark package for GPUCompress.
Runs a Taylor-Green vortex spectral-element CFD simulation with
GPU-accelerated compression via the HDF5 VOL connector.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo, LocalExecInfo
from jarvis_cd.shell.process import Rm
import os


class GpucompressNekrs(Application):
    """
    Deploy and run the nekRS GPUCompress benchmark.
    """

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {
                'name': 'nprocs',
                'msg': 'Number of MPI processes',
                'type': int,
                'default': 2,
            },
            {
                'name': 'ppn',
                'msg': 'Processes per node',
                'type': int,
                'default': 2,
            },
            {
                'name': 'nekrs_home',
                'msg': 'Path to nekRS installation prefix',
                'type': str,
                'default': '/opt/nekrs',
            },
            {
                'name': 'case_dir',
                'msg': 'Path to TGV case directory',
                'type': str,
                'default': '/opt/nekrs/cases/tgv',
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
                'name': 'num_steps',
                'msg': 'Number of simulation steps',
                'type': int,
                'default': 20,
            },
            {
                'name': 'checkpoint_int',
                'msg': 'Steps between checkpoints',
                'type': int,
                'default': 10,
            },
            {
                'name': 'use_fp32',
                'msg': 'Use single precision (1=yes, 0=no)',
                'type': int,
                'default': 1,
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
                'default': '/tmp/gpucompress_nekrs_results',
            },
        ]

    def _configure(self, **kwargs):
        pass

    def start(self):
        gpuc = self.config['gpucompress_dir']
        nekrs_home = self.config['nekrs_home']
        os.makedirs(self.config['results_dir'], exist_ok=True)

        env = dict(os.environ)
        lib_paths = [
            f'{gpuc}/build',
            self.config['nvcomp_lib'],
            self.config['hdf5_lib'],
            f'{nekrs_home}/lib',
        ]
        ld = ':'.join(lib_paths)
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f'{ld}:{env["LD_LIBRARY_PATH"]}'
        else:
            env['LD_LIBRARY_PATH'] = ld
        env['NEKRS_HOME'] = nekrs_home
        env['GPUCOMPRESS_WEIGHTS'] = self.config['weights']
        env['HDF5_PLUGIN_PATH'] = f'{gpuc}/build'
        env['USE_FP32'] = str(self.config['use_fp32'])

        nekrs_bin = f'{nekrs_home}/bin/nekrs'
        cmd = (f'{nekrs_bin}'
               f' --setup {self.config["case_dir"]}/tgv.par'
               f' --backend CUDA')

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
