"""
LAMMPS benchmark package for GPUCompress.
Runs a Lennard-Jones melt with hot sphere expansion, dumping
GPU-resident fields through the HDF5 VOL connector.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo, LocalExecInfo
from jarvis_cd.shell.process import Rm
import os


class GpucompressLammps(Application):
    """
    Deploy and run the LAMMPS GPUCompress benchmark.
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
                'name': 'lmp_bin',
                'msg': 'Path to LAMMPS binary (lmp)',
                'type': str,
                'default': '/opt/sims/lammps/build/lmp',
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
                'name': 'atoms',
                'msg': 'Box size per dimension (total atoms ~ 4*N^3)',
                'type': int,
                'default': 80,
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
                'default': 10,
            },
            {
                'name': 'sim_interval',
                'msg': 'Physics steps between dumps',
                'type': int,
                'default': 50,
            },
            {
                'name': 'warmup_steps',
                'msg': 'Physics steps before first dump',
                'type': int,
                'default': 100,
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
                'default': '/tmp/gpucompress_lammps_results',
            },
        ]

    def _configure(self, **kwargs):
        pass

    def start(self):
        gpuc = self.config['gpucompress_dir']
        atoms = self.config['atoms']
        interval = self.config['sim_interval']
        total_steps = self.config['warmup_steps'] + self.config['timesteps'] * interval
        half = atoms // 2
        quarter = atoms // 4

        os.makedirs(self.config['results_dir'], exist_ok=True)

        # Generate LAMMPS input script
        input_file = os.path.join(self.config['results_dir'], 'input.lmp')
        with open(input_file, 'w') as f:
            f.write(f"""units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 {atoms} 0 {atoms} 0 {atoms}
create_box      1 box
create_atoms    1 box
mass            1 1.0

region          hot sphere {half} {half} {half} {quarter}
group           hot region hot
group           cold subtract all hot
velocity        cold create 0.01 87287 loop geom
velocity        hot create 10.0 12345 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    every 10 delay 0 check no

fix             1 all nve
fix             gpuc all gpucompress {interval} positions velocities forces
thermo          {interval}
timestep        0.003
run             {total_steps}
""")

        # Build env
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
        env['GPUCOMPRESS_VERIFY'] = '0'
        env['LAMMPS_DUMP_FIELDS'] = '1'
        env['LAMMPS_DUMP_DIR'] = os.path.join(self.config['results_dir'], 'raw_fields')
        os.makedirs(env['LAMMPS_DUMP_DIR'], exist_ok=True)

        cmd = f'{self.config["lmp_bin"]} -k on g 1 -sf kk -in {input_file}'

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
