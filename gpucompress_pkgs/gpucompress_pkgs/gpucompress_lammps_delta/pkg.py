"""
gpucompress_lammps_delta — LAMMPS Lennard-Jones FCC + hot-sphere benchmark
with GPUCompress Kokkos fix.

Jarvis Path B (install_manager: container, container_engine: apptainer).
Requires `gpucompress_base` to precede this package in the pipeline YAML
so HDF5, nvcomp, GPUCompress libs/weights/patches, and the two LAMMPS
bridge shared libraries (built by this package's build.sh on top of the
base image) are all present in the shared build image.

Workload mirrors bench_tests/lammps.sh from the nn-feature-engineering
tree.  Unlike the Nyx/VPIC/WarpX ports, LAMMPS is a single-phase workflow:
the `fix gpucompress` Kokkos fix writes compressed HDF5 via GPUCompress's
VOL connector in-situ during the MD run, producing gpuc_step_* output
directories.  After the sim finishes, this package parses those directories
host-side to produce a per-timestep CSV (no separate generic_benchmark
invocation — LAMMPS's own fix covers both phases).
"""
import os
import glob

from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo
from jarvis_cd.shell.process import Mkdir, Rm


VALID_PHASES = {
    'lz4', 'snappy', 'deflate', 'gdeflate', 'zstd',
    'ans', 'cascaded', 'bitcomp',
    'nn', 'nn-rl', 'nn-rl+exp50',
}

VALID_POLICIES = {'balanced', 'ratio', 'speed'}

# Absolute paths inside the built image. Must match build.sh / Dockerfile.deploy.
LMP_BIN = '/opt/sims/lammps/build/lmp'
WEIGHTS = '/opt/GPUCompress/neural_net/weights/model.nnwt'

# Runtime LD_LIBRARY_PATH — Jarvis's auto-generated %environment sets
# /opt/<pkg>/install/lib (doesn't exist in our SIF), so we prefix each
# Exec command with `env LD_LIBRARY_PATH=…` to override at exec time.
# Mirrors the gpucompress_nyx_delta pattern. Includes /usr/local/lib for
# the two LAMMPS bridge shared libraries (liblammps_gpucompress_udf.so
# and liblammps_ranking_profiler.so) installed there by build.sh.
LD_LIBRARY_PATH = (
    '/.singularity.d/libs'            # host libcuda.so.1 bound via --nv
    ':/usr/local/cuda/lib64'          # CUDA runtime libs
    ':/opt/hdf5-install/lib'          # HDF5 2.0.0
    ':/opt/nvcomp/lib'                # nvcomp
    ':/opt/GPUCompress/build'         # libgpucompress + VOL/Filter .so
    ':/usr/local/lib'                 # liblammps_gpucompress_udf + ranking_profiler
)


class GpucompressLammpsDelta(Application):

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            # HDF5 mode / algorithm / policy
            {'name': 'hdf5_mode',
             'msg': "HDF5 mode: 'default' (no-comp baseline: algo=lz4, "
                    "ratio reported as 1.0) or 'vol' (GPUCompress VOL)",
             'type': str, 'default': 'default'},
            {'name': 'phase',
             'msg': 'Compression phase (vol mode only): '
                    'lz4|snappy|deflate|gdeflate|zstd|ans|cascaded|bitcomp'
                    '|nn|nn-rl|nn-rl+exp50',
             'type': str, 'default': 'lz4'},
            {'name': 'policy',
             'msg': 'NN cost-model policy: balanced | ratio | speed',
             'type': str, 'default': 'balanced'},
            {'name': 'error_bound',
             'msg': 'Lossy error bound (0.0 = lossless)',
             'type': float, 'default': 0.0},

            # LAMMPS I/O volume parameters
            {'name': 'atoms',
             'msg': 'Box edge in FCC unit cells (total atoms = 4 * N^3). '
                    '20 ~ 3 MB/dump, 40 ~ 27 MB, 80 ~ 216 MB, 120 ~ 730 MB',
             'type': int, 'default': 40},
            {'name': 'timesteps',
             'msg': 'Number of field dumps',
             'type': int, 'default': 3},
            {'name': 'sim_interval',
             'msg': 'MD steps between dumps',
             'type': int, 'default': 50},
            {'name': 'warmup_steps',
             'msg': 'MD steps before first dump (also used to filter '
                    'gpuc_step_* directories during CSV post-processing)',
             'type': int, 'default': 20},
            {'name': 'chunk_mb',
             'msg': 'HDF5 chunk size (MB)',
             'type': int, 'default': 4},
            {'name': 'verify',
             'msg': 'Bitwise readback verify: 1 = on, 0 = off',
             'type': int, 'default': 1},

            # LAMMPS physics parameters (affect data variety / compressibility)
            {'name': 't_hot',
             'msg': 'Hot-sphere temperature (LJ units)',
             'type': float, 'default': 10.0},
            {'name': 't_cold',
             'msg': 'Cold-lattice temperature (LJ units)',
             'type': float, 'default': 0.01},
            {'name': 'hot_radius_frac',
             'msg': 'Hot-sphere radius as fraction of box half-width',
             'type': float, 'default': 0.25},

            # MPI / GPU launch
            {'name': 'nprocs',
             'msg': 'Number of MPI processes',
             'type': int, 'default': 1},
            {'name': 'ppn',
             'msg': 'Processes per node',
             'type': int, 'default': 1},
            {'name': 'num_gpus',
             'msg': 'Number of GPUs per rank (passed to -k on g N)',
             'type': int, 'default': 1},

            # Container build options
            {'name': 'cuda_arch',
             'msg': 'CUDA compute capability (must match gpucompress_base)',
             'type': int, 'default': 80},
            {'name': 'kokkos_arch',
             'msg': "Kokkos arch flag suffix (AMPERE80 for A100, "
                    "AMPERE86 for A40, HOPPER90 for H100). Default: "
                    "derived from cuda_arch as AMPERE<cuda_arch>",
             'type': str, 'default': ''},
            {'name': 'deploy_base',
             'msg': 'Base image for the deploy stage',
             'type': str,
             'default': 'nvidia/cuda:12.6.0-runtime-ubuntu24.04'},
            {'name': 'use_gpu',
             'msg': 'Pass --nv to apptainer at run time',
             'type': bool, 'default': True},

            # Runtime output path — a /tmp directory is bind-mounted into
            # the apptainer instance by default, so host and container see
            # the same files there.
            {'name': 'results_dir',
             'msg': 'Output root (empty = /tmp/gpucompress_lammps_<pkg_id>...)',
             'type': str, 'default': ''},
        ]

    def _configure(self, **kwargs):
        pass

    # ------------------------------------------------------------------
    # Container build hooks (Jarvis Path B)
    # ------------------------------------------------------------------

    def _build_phase(self):
        if self.config.get('deploy_mode') != 'container':
            return None
        cuda_arch = self.config['cuda_arch']
        kokkos_arch = self.config.get('kokkos_arch') or f'AMPERE{cuda_arch}'
        content = self._read_build_script('build.sh', {
            'CUDA_ARCH': str(cuda_arch),
            'KOKKOS_ARCH': kokkos_arch,
        })
        suffix = f"cuda-{cuda_arch}"
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        cfg = self.config
        self._validate(cfg)

        atoms         = int(cfg['atoms'])
        timesteps     = int(cfg['timesteps'])
        sim_interval  = int(cfg['sim_interval'])
        warmup_steps  = int(cfg['warmup_steps'])
        chunk_mb      = int(cfg['chunk_mb'])
        verify        = int(cfg['verify'])
        error_bound   = float(cfg['error_bound'])
        t_hot         = float(cfg['t_hot'])
        t_cold        = float(cfg['t_cold'])
        hot_radius_frac = float(cfg['hot_radius_frac'])
        num_gpus      = int(cfg['num_gpus'])

        # Derived (mirrors bench_tests/lammps.sh)
        natoms       = 4 * atoms ** 3
        total_steps  = warmup_steps + timesteps * sim_interval
        hot_radius   = max(1, int(atoms / 2 * hot_radius_frac))
        orig_bytes   = natoms * 36                     # 3 fields × 3 xyz × 4 bytes
        orig_mb      = orig_bytes / 1048576

        # Map hdf5_mode → GPUCOMPRESS_ALGO + phase_label
        if cfg['hdf5_mode'] == 'default':
            gpuc_algo = 'lz4'
            phase_label = 'no-comp'
        else:
            gpuc_algo = cfg['phase']
            phase_label = cfg['phase']

        verify_tag = '_noverify' if verify == 0 else ''

        results_dir = cfg['results_dir'] or (
            f"/tmp/gpucompress_lammps_{self.pkg_id}"
            f"_box{atoms}_ts{timesteps}_{cfg['hdf5_mode']}{verify_tag}"
        )
        workdir = f'{results_dir}/work_{phase_label}'
        for d in (results_dir, workdir):
            Mkdir(d).run()

        input_file = f'{workdir}/input.lmp'
        self._write_input_deck(
            input_file, atoms, sim_interval, total_steps,
            hot_radius, t_hot, t_cold,
        )

        lammps_log = f'{workdir}/lammps.log'

        # ── Run LAMMPS with the GPUCompress Kokkos fix ──────────────────
        lmp_env = dict(self.mod_env)
        lmp_env['GPUCOMPRESS_ALGO']            = gpuc_algo
        lmp_env['GPUCOMPRESS_VERIFY']          = str(verify)
        lmp_env['GPUCOMPRESS_WEIGHTS']         = WEIGHTS
        lmp_env['GPUCOMPRESS_POLICY']          = cfg['policy']
        lmp_env['GPUCOMPRESS_ERROR_BOUND']     = str(error_bound)
        lmp_env['GPUCOMPRESS_CHUNK_MB']        = str(chunk_mb)
        lmp_env['GPUCOMPRESS_DETAILED_TIMING'] = '1'
        lmp_env['HDF5_PLUGIN_PATH']            = '/opt/GPUCompress/build'

        cmd = (
            f'env LD_LIBRARY_PATH={LD_LIBRARY_PATH} '
            f'{LMP_BIN} -k on g {num_gpus} -sf kk '
            f'-in {input_file}'
        )

        phase1 = Exec(cmd, MpiExecInfo(
            nprocs=int(cfg['nprocs']),
            ppn=int(cfg['ppn']),
            hostfile=self.hostfile,
            port=self.ssh_port,
            container=self._container_engine,
            container_image=self.deploy_image_name(),
            shared_dir=self.shared_dir,
            private_dir=self.private_dir,
            env=lmp_env,
            gpu=cfg.get('use_gpu', True),
            cwd=workdir,
            pipe_stdout=lammps_log,
            pipe_stderr=lammps_log,
        ))
        phase1.run()
        p1_rc = max(phase1.exit_code.values()) if getattr(phase1, 'exit_code', None) else 0
        if p1_rc != 0:
            print(f"[gpucompress_lammps_delta] WARNING: LAMMPS exited {p1_rc} "
                  f"— see {lammps_log}")

        # ── Host-side: parse gpuc_step_* dirs → CSV ──────────────────────
        # Mirrors the parsing loop at the tail of bench_tests/lammps.sh.
        step_dirs = sorted(glob.glob(f'{workdir}/gpuc_step_*'))
        if not step_dirs:
            raise RuntimeError(
                f"No gpuc_step_* directories in {workdir} — the fix "
                f"produced no output. See {lammps_log}"
            )

        verify_failed = False
        try:
            with open(lammps_log) as f:
                verify_failed = 'VERIFY FAILED' in f.read()
        except OSError:
            pass

        csv_path = f'{results_dir}/benchmark_lammps_timesteps.csv'
        with open(csv_path, 'w') as csv:
            csv.write('rank,phase,timestep,write_ms,ratio,orig_mb,comp_mb,verify\n')
            ts = 0
            for d in step_dirs:
                name = os.path.basename(d)
                step_str = name[len('gpuc_step_'):].lstrip('0') or '0'
                try:
                    step_num = int(step_str)
                except ValueError:
                    continue
                if step_num < warmup_steps:
                    continue
                comp_bytes = self._dir_bytes(d)
                if cfg['hdf5_mode'] == 'default':
                    ratio = 1.00
                    comp_mb_report = orig_mb
                else:
                    ratio = (orig_bytes / comp_bytes) if comp_bytes > 0 else 0.0
                    comp_mb_report = comp_bytes / 1048576
                verify_ok = 0 if verify_failed else 1
                csv.write(
                    f'0,{phase_label},{ts},0,'
                    f'{ratio:.2f},{orig_mb:.1f},{comp_mb_report:.2f},{verify_ok}\n'
                )
                ts += 1

        if p1_rc != 0:
            raise RuntimeError(
                f"LAMMPS failed (exit {p1_rc}) — see {lammps_log}"
            )
        if ts == 0:
            raise RuntimeError(
                f"No gpuc_step_* directories past warmup_steps={warmup_steps} "
                f"in {workdir}. Either warmup_steps is too large, sim didn't "
                f"run long enough, or the fix never fired."
            )

    def stop(self):
        pass

    def clean(self):
        results_dir = self.config.get('results_dir')
        if results_dir and os.path.isdir(results_dir):
            Rm(results_dir).run()
        # Also sweep the auto-generated default-path variants created by start()
        Rm(f'/tmp/gpucompress_lammps_{self.pkg_id}_*').run()

    # ------------------------------------------------------------------
    def _validate(self, cfg):
        if cfg['hdf5_mode'] not in ('default', 'vol'):
            raise ValueError(
                f"hdf5_mode must be 'default' or 'vol', got {cfg['hdf5_mode']!r}"
            )
        if cfg['hdf5_mode'] == 'vol' and cfg['phase'] not in VALID_PHASES:
            raise ValueError(
                f"phase {cfg['phase']!r} not in {sorted(VALID_PHASES)}"
            )
        if cfg['policy'] not in VALID_POLICIES:
            raise ValueError(
                f"policy must be one of {sorted(VALID_POLICIES)}, "
                f"got {cfg['policy']!r}"
            )
        if int(cfg['atoms']) < 2:
            raise ValueError(
                f"atoms must be >= 2 (4*N^3 total atoms), got {cfg['atoms']}"
            )

    def _write_input_deck(self, path, atoms, sim_interval, total_steps,
                          hot_radius, t_hot, t_cold):
        """Mirrors the heredoc in bench_tests/lammps.sh."""
        half = atoms // 2
        deck = f"""units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 {atoms} 0 {atoms} 0 {atoms}
create_box      1 box
create_atoms    1 box
mass            1 1.0

region          hot sphere {half} {half} {half} {hot_radius}
group           hot region hot
group           cold subtract all hot
velocity        cold create {t_cold} 87287 loop geom
velocity        hot  create {t_hot}  12345 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    every 10 delay 0 check no

fix             1 all nve
fix             gpuc all gpucompress {sim_interval} positions velocities forces
thermo          {sim_interval}
timestep        0.003
run             {total_steps}
"""
        with open(path, 'w') as f:
            f.write(deck)

    @staticmethod
    def _dir_bytes(d):
        """Recursive apparent-size (bytes) of a directory — analogue of
        `du -sb` used in bench_tests/lammps.sh."""
        total = 0
        for root, _, files in os.walk(d):
            for fname in files:
                try:
                    total += os.path.getsize(os.path.join(root, fname))
                except OSError:
                    pass
        return total
