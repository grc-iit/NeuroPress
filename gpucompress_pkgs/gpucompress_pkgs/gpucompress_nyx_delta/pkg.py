"""
gpucompress_nyx — Nyx Sedov-blast benchmark with GPUCompress integration.

Jarvis Path B (install_manager: container, container_engine: apptainer).
Requires `gpucompress_base` to precede this package in the pipeline YAML
so that HDF5, nvcomp and the GPUCompress library/weights/patches are
present in the shared build image.

Workload mirrors bench_tests/nyx.sh from the nn-feature-engineering merge:
  Phase 1 — run nyx_HydroTests with NYX_DUMP_FIELDS=1 to dump raw .f32
            files per FAB per component per plt* timestep directory.
  Phase 2 — flatten the dumps into a single directory and run
            generic_benchmark against them.
"""
import os
import glob

from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo
from jarvis_cd.shell.process import Mkdir, Rm


POLICY_WEIGHTS = {
    'ratio':    (0.0, 0.0, 1.0),
    'speed':    (1.0, 1.0, 0.0),
    'balanced': (1.0, 1.0, 1.0),
}

VALID_PHASES = {
    'lz4', 'snappy', 'deflate', 'gdeflate', 'zstd',
    'ans', 'cascaded', 'bitcomp',
    'nn', 'nn-rl', 'nn-rl+exp50',
}

# Absolute paths inside the built image. Must match build.sh / Dockerfile.deploy.
NYX_BIN     = '/opt/sims/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests'
GENERIC_BIN = '/opt/GPUCompress/build/generic_benchmark'
WEIGHTS     = '/opt/GPUCompress/neural_net/weights/model.nnwt'


class GpucompressNyxDelta(Application):

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            # HDF5 mode / algorithm / policy
            {'name': 'hdf5_mode',
             'msg': "HDF5 mode: 'default' (no-comp baseline) or 'vol' "
                    "(GPUCompress VOL)",
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

            # Nyx I/O volume
            {'name': 'ncell',
             'msg': 'Grid cells per dimension (NYX_NCELL)',
             'type': int, 'default': 64},
            {'name': 'max_step',
             'msg': 'Total simulation steps (NYX_MAX_STEP)',
             'type': int, 'default': 30},
            {'name': 'plot_int',
             'msg': 'Steps between plot-file dumps (NYX_PLOT_INT)',
             'type': int, 'default': 10},
            {'name': 'chunk_mb',
             'msg': 'HDF5 chunk size (MB)',
             'type': int, 'default': 4},
            {'name': 'verify',
             'msg': 'Bitwise readback verify: 1 = on, 0 = off',
             'type': int, 'default': 1},

            # NN online learning knobs (passed to generic_benchmark)
            {'name': 'sgd_lr',
             'msg': 'SGD learning rate (--lr)',
             'type': float, 'default': 0.2},
            {'name': 'sgd_mape',
             'msg': 'MAPE threshold for SGD firing (--mape)',
             'type': float, 'default': 0.10},
            {'name': 'explore_k',
             'msg': 'Top-K exploration alternatives (--explore-k)',
             'type': int, 'default': 4},
            {'name': 'explore_thresh',
             'msg': 'Exploration error threshold (--explore-thresh)',
             'type': float, 'default': 0.20},

            # Container build options
            {'name': 'cuda_arch',
             'msg': 'CUDA compute capability (must match gpucompress_base)',
             'type': int, 'default': 80},
            {'name': 'deploy_base',
             'msg': 'Base image for the deploy stage',
             'type': str,
             'default': 'nvidia/cuda:12.8.0-runtime-ubuntu24.04'},
            {'name': 'use_gpu',
             'msg': 'Pass --nv to apptainer at run time',
             'type': bool, 'default': True},

            # Runtime output path — a /tmp directory is bind-mounted into
            # the apptainer instance by default, so host and container see
            # the same files there.
            {'name': 'results_dir',
             'msg': 'Output root (empty = /tmp/gpucompress_nyx_<pkg_id>)',
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
        content = self._read_build_script('build.sh', {
            'CUDA_ARCH': str(self.config['cuda_arch']),
        })
        suffix = f"cuda-{self.config['cuda_arch']}"
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

        verify = int(cfg['verify'])
        error_bound = float(cfg['error_bound'])
        lossy = error_bound not in (0, 0.0)
        verify_tag = '_noverify' if verify == 0 else ''
        eb_tag = f'_lossy{error_bound}' if lossy else ''

        results_dir = cfg['results_dir'] or (
            f"/tmp/gpucompress_nyx_{self.pkg_id}"
            f"_n{cfg['ncell']}_ms{cfg['max_step']}_{cfg['hdf5_mode']}"
            f"{eb_tag}{verify_tag}"
        )
        raw_dir  = f'{results_dir}/raw_fields'
        flat_dir = f'{results_dir}/flat_fields'
        bench_phase = 'no-comp' if cfg['hdf5_mode'] == 'default' else cfg['phase']
        bench_dir = f'{results_dir}/{cfg["hdf5_mode"]}_{bench_phase}'
        for d in (results_dir, raw_dir, flat_dir, bench_dir):
            Mkdir(d).run()

        input_file = f'{results_dir}/inputs.sedov'
        self._write_inputs_sedov(input_file, cfg, WEIGHTS)

        nyx_log = f'{results_dir}/nyx_sim.log'
        bench_log = f'{bench_dir}/nyx_bench.log'

        w0, w1, w2 = POLICY_WEIGHTS[cfg['policy']]

        # ── Phase 1: Nyx Sedov sim with raw-field dump ───────────────────
        nyx_env = dict(self.mod_env)
        nyx_env['NYX_DUMP_FIELDS'] = '1'
        nyx_env['NYX_DUMP_DIR'] = raw_dir
        Exec(f'{NYX_BIN} {input_file}', MpiExecInfo(
            nprocs=1,
            ppn=1,
            hostfile=self.hostfile,
            port=self.ssh_port,
            container=self._container_engine,
            container_image=self.deploy_image_name(),
            shared_dir=self.shared_dir,
            private_dir=self.private_dir,
            env=nyx_env,
            gpu=cfg.get('use_gpu', True),
            pipe_stdout=nyx_log,
            pipe_stderr=nyx_log,
        )).run()

        # ── Host-side: gather plt* dirs, derive dims, flatten symlinks ───
        plt_dirs = sorted(glob.glob(f'{raw_dir}/plt*'))
        if not plt_dirs:
            raise RuntimeError(
                f"No plt* directories in {raw_dir} — check {nyx_log}"
            )

        first_f32 = None
        for plt in plt_dirs:
            matches = sorted(glob.glob(f'{plt}/*.f32'))
            if matches:
                first_f32 = matches[0]
                break
        if not first_f32:
            raise RuntimeError(f"No .f32 files found under {raw_dir}")

        n_floats = os.path.getsize(first_f32) // 4
        dims = f'{n_floats},1'

        for plt in plt_dirs:
            ts_name = os.path.basename(plt)
            for f in sorted(glob.glob(f'{plt}/*.f32')):
                link = f'{flat_dir}/{ts_name}_{os.path.basename(f)}'
                if os.path.islink(link) or os.path.exists(link):
                    os.remove(link)
                os.symlink(f, link)

        # ── Phase 2: generic_benchmark ───────────────────────────────────
        bench_parts = [
            GENERIC_BIN,
            WEIGHTS,
            f'--data-dir {flat_dir}',
            f'--dims {dims}',
            '--ext .f32',
            f'--chunk-mb {cfg["chunk_mb"]}',
            f'--name nyx_n{cfg["ncell"]}',
        ]
        if lossy:
            bench_parts.append(f'--error-bound {error_bound}')
        if verify == 0:
            bench_parts.append('--no-verify')
        bench_parts += [
            f'--phase {bench_phase}',
            f'--w0 {w0} --w1 {w1} --w2 {w2}',
            f'--lr {cfg["sgd_lr"]} --mape {cfg["sgd_mape"]}',
            f'--explore-k {cfg["explore_k"]} --explore-thresh {cfg["explore_thresh"]}',
            f'--out-dir {bench_dir}',
        ]
        bench_cmd = ' '.join(bench_parts)

        bench_env = dict(self.mod_env)
        bench_env['GPUCOMPRESS_DETAILED_TIMING'] = '1'
        Exec(bench_cmd, MpiExecInfo(
            nprocs=1,
            ppn=1,
            hostfile=self.hostfile,
            port=self.ssh_port,
            container=self._container_engine,
            container_image=self.deploy_image_name(),
            shared_dir=self.shared_dir,
            private_dir=self.private_dir,
            env=bench_env,
            gpu=cfg.get('use_gpu', True),
            pipe_stdout=bench_log,
            pipe_stderr=bench_log,
        )).run()

    def stop(self):
        pass

    def clean(self):
        results_dir = self.config.get('results_dir')
        if results_dir and os.path.isdir(results_dir):
            Rm(results_dir).run()
        # Also sweep the auto-generated default-path variants created by start()
        Rm(f'/tmp/gpucompress_nyx_{self.pkg_id}_*').run()

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
        if cfg['policy'] not in POLICY_WEIGHTS:
            raise ValueError(
                f"policy must be one of {sorted(POLICY_WEIGHTS)}, "
                f"got {cfg['policy']!r}"
            )

    def _write_inputs_sedov(self, path, cfg, weights):
        ncell = cfg['ncell']
        max_grid = min(ncell, 128)
        deck = f"""# Sedov blast wave — based on Nyx/Exec/HydroTests/inputs.regtest.sedov
amr.n_cell         = {ncell} {ncell} {ncell}
amr.max_level      = 0
amr.max_grid_size  = {max_grid}
amr.ref_ratio      = 2 2 2 2
amr.regrid_int     = 2
amr.blocking_factor = 4
amr.plot_int       = {cfg['plot_int']}
amr.check_int      = 0

max_step       = {cfg['max_step']}
stop_time      = -1

geometry.coord_sys   = 0
geometry.prob_lo     = 0.0 0.0 0.0
geometry.prob_hi     = 1.0 1.0 1.0
geometry.is_periodic = 0 0 0

# Outflow BCs on all faces (2 = outflow)
nyx.lo_bc          = 2 2 2
nyx.hi_bc          = 2 2 2

# Hydro
nyx.do_hydro       = 1
nyx.do_grav        = 0
nyx.do_santa_barbara = 0
nyx.ppm_type       = 0
nyx.init_shrink    = 0.01
nyx.cfl            = 0.5
nyx.dt_cutoff      = 5.e-20
nyx.change_max     = 1.1

# Comoving (required by Nyx, set for non-cosmological use)
nyx.comoving_OmM   = 1.0
nyx.comoving_OmB   = 1.0
nyx.comoving_h     = 0.0
nyx.initial_z      = 0.0

# Species
nyx.h_species      = 0.76
nyx.he_species     = 0.24

# Problem setup: Sedov blast (prob_type=33)
prob.prob_type     = 33
prob.r_init        = 0.01
prob.p_ambient     = 1.e-5
prob.dens_ambient  = 1.0
prob.exp_energy    = 1.0
prob.nsub          = 10

# GPUCompress integration
nyx.use_gpucompress       = 1
nyx.gpucompress_weights   = {weights}
nyx.gpucompress_algorithm = auto
nyx.gpucompress_policy    = ratio
nyx.gpucompress_verify    = 0
nyx.gpucompress_chunk_mb  = {cfg['chunk_mb']}

# Prevent AMReX from pre-allocating nearly all GPU memory at startup
amrex.the_arena_init_size = 0
amrex.the_async_arena_init_size = 0
"""
        with open(path, 'w') as f:
            f.write(deck)
