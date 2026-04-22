"""
gpucompress_warpx_delta — WarpX laser-wakefield (LWFA) benchmark with
GPUCompress integration.

Jarvis Path B (install_manager: container, container_engine: apptainer).
Requires `gpucompress_base` to precede this package in the pipeline YAML
so that HDF5, nvcomp and the GPUCompress library/weights/patches are
present in the shared build image.

Workload mirrors bench_tests/warpx.sh (retrieved from git 3c66e01):
  Phase 1 — run warpx.3d with WARPX_DUMP_FIELDS=1 to dump raw .f32
            files per FAB per component per diag* timestep directory.
  Phase 2 — flatten the dumps into a single directory and run
            generic_benchmark against them.

Unlike Nyx (which consumes an input deck), WarpX takes a base inputs
file and a long list of command-line `amr.*` / `diag1.*` /
`gpucompress.*` overrides.
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
WARPX_BIN    = ('/opt/sims/warpx/build-gpucompress/bin/'
                'warpx.3d.MPI.CUDA.SP.PSP.OPMD.EB.QED')
WARPX_INPUTS = ('/opt/sims/warpx/Examples/Physics_applications/'
                'laser_acceleration/inputs_base_3d')
GENERIC_BIN  = '/opt/GPUCompress/build/generic_benchmark'
WEIGHTS      = '/opt/GPUCompress/neural_net/weights/model.nnwt'

# Runtime LD_LIBRARY_PATH — Jarvis's auto-generated %environment sets
# /opt/<pkg>/install/lib (doesn't exist in our SIF), so we prefix each
# Exec command with `env LD_LIBRARY_PATH=…` to override at exec time.
# Mirrors the gpucompress_nyx_delta pattern.
LD_LIBRARY_PATH = (
    '/.singularity.d/libs'            # host libcuda.so.1 bound via --nv
    ':/usr/local/cuda/lib64'          # CUDA runtime libs
    ':/opt/hdf5-install/lib'          # HDF5 2.0.0
    ':/opt/nvcomp/lib'                # nvcomp
    ':/opt/GPUCompress/build'         # libgpucompress + VOL/Filter .so
)


class GpucompressWarpxDelta(Application):

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
             'msg': 'NN cost-model policy: balanced | ratio | speed '
                    '(ratio recommended for LWFA)',
             'type': str, 'default': 'ratio'},
            {'name': 'error_bound',
             'msg': 'Lossy error bound (0.0 = lossless; LWFA tolerates 1e-3 – 1e-2)',
             'type': float, 'default': 0.0},

            # WarpX I/O volume
            {'name': 'ncell',
             'msg': 'Grid cells as space-separated "nx ny nz" (WARPX_NCELL)',
             'type': str, 'default': '32 32 256'},
            {'name': 'max_step',
             'msg': 'Total simulation steps (WARPX_MAX_STEP)',
             'type': int, 'default': 30},
            {'name': 'diag_int',
             'msg': 'Steps between diag1 dumps (WARPX_DIAG_INT)',
             'type': int, 'default': 10},
            {'name': 'max_grid_size',
             'msg': 'AMR max grid size (empty = WarpX default, '
                    'sets amr.max_grid_size)',
             'type': str, 'default': ''},
            {'name': 'blocking_factor',
             'msg': 'AMR blocking factor (empty = WarpX default, '
                    'sets amr.blocking_factor)',
             'type': str, 'default': ''},
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
             'default': 'nvidia/cuda:12.6.0-runtime-ubuntu24.04'},
            {'name': 'use_gpu',
             'msg': 'Pass --nv to apptainer at run time',
             'type': bool, 'default': True},

            # Runtime output path — a /tmp directory is bind-mounted into
            # the apptainer instance by default, so host and container see
            # the same files there.
            {'name': 'results_dir',
             'msg': 'Output root (empty = /tmp/gpucompress_warpx_<pkg_id>_…)',
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

        ncell = str(cfg['ncell']).strip()
        ncell_compact = ncell.replace(' ', 'x')

        results_dir = cfg['results_dir'] or (
            f"/tmp/gpucompress_warpx_{self.pkg_id}"
            f"_{ncell_compact}_ms{cfg['max_step']}_{cfg['hdf5_mode']}"
            f"{eb_tag}{verify_tag}"
        )
        raw_dir  = f'{results_dir}/raw_fields'
        flat_dir = f'{results_dir}/flat_fields'
        bench_phase = 'no-comp' if cfg['hdf5_mode'] == 'default' else cfg['phase']
        bench_dir = f'{results_dir}/{cfg["hdf5_mode"]}_{bench_phase}'
        for d in (results_dir, raw_dir, flat_dir, bench_dir):
            Mkdir(d).run()

        warpx_log = f'{results_dir}/warpx_sim.log'
        bench_log = f'{bench_dir}/warpx_bench.log'

        w0, w1, w2 = POLICY_WEIGHTS[cfg['policy']]

        chunk_bytes = int(cfg['chunk_mb']) * 1024 * 1024

        # ── Phase 1: WarpX LWFA sim with raw-field dump ──────────────────
        # WarpX takes inputs_base_3d followed by key=value overrides.
        warpx_parts = [
            WARPX_BIN,
            WARPX_INPUTS,
            f'max_step={cfg["max_step"]}',
            f'amr.n_cell="{ncell}"',
        ]
        mgs = str(cfg.get('max_grid_size', '')).strip()
        if mgs:
            warpx_parts.append(f'amr.max_grid_size="{mgs}"')
        bf = str(cfg.get('blocking_factor', '')).strip()
        if bf:
            warpx_parts.append(f'amr.blocking_factor="{bf}"')
        warpx_parts += [
            'diagnostics.diags_names=diag1',
            f'diag1.intervals={cfg["diag_int"]}',
            'diag1.diag_type=Full',
            'diag1.format=gpucompress',
            f'gpucompress.weights_path="{WEIGHTS}"',
            'gpucompress.algorithm=auto',
            'gpucompress.policy=ratio',
            f'gpucompress.error_bound={error_bound}',
            f'gpucompress.chunk_bytes={chunk_bytes}',
        ]
        warpx_cmd = ' '.join(warpx_parts)

        warpx_env = dict(self.mod_env)
        warpx_env['WARPX_DUMP_FIELDS'] = '1'
        warpx_env['WARPX_DUMP_DIR'] = raw_dir
        phase1 = Exec(
            f'env LD_LIBRARY_PATH={LD_LIBRARY_PATH} {warpx_cmd}',
            MpiExecInfo(
            nprocs=1,
            ppn=1,
            hostfile=self.hostfile,
            port=self.ssh_port,
            container=self._container_engine,
            container_image=self.deploy_image_name(),
            shared_dir=self.shared_dir,
            private_dir=self.private_dir,
            env=warpx_env,
            gpu=cfg.get('use_gpu', True),
            pipe_stdout=warpx_log,
            pipe_stderr=warpx_log,
        ))
        phase1.run()
        # Phase 1 (WarpX) warns and continues on non-zero; the diag* check
        # below is the real gate.
        p1_rc = max(phase1.exit_code.values()) if getattr(phase1, 'exit_code', None) else 0
        if p1_rc != 0:
            print(f"[gpucompress_warpx_delta] WARNING: WarpX exited {p1_rc} — see {warpx_log}")

        # ── Host-side: gather diag* dirs, derive dims, flatten symlinks ──
        diag_dirs = sorted(glob.glob(f'{raw_dir}/diag*'))
        if not diag_dirs:
            raise RuntimeError(
                f"No diag* directories in {raw_dir} — check {warpx_log}"
            )

        first_f32 = None
        for d in diag_dirs:
            matches = sorted(glob.glob(f'{d}/*.f32'))
            if matches:
                first_f32 = matches[0]
                break
        if not first_f32:
            raise RuntimeError(f"No .f32 files found under {raw_dir}")

        n_floats = os.path.getsize(first_f32) // 4
        # WarpX dumps each FAB component as a 1-D blob — use flat (1,N) dims.
        dims = f'1,{n_floats}'

        for d in diag_dirs:
            ts_name = os.path.basename(d)
            for f in sorted(glob.glob(f'{d}/*.f32')):
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
            f'--name warpx_{ncell_compact}',
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
        phase2 = Exec(
            f'env LD_LIBRARY_PATH={LD_LIBRARY_PATH} {bench_cmd}',
            MpiExecInfo(
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
        ))
        phase2.run()
        p2_rc = max(phase2.exit_code.values()) if getattr(phase2, 'exit_code', None) else 0
        if p2_rc != 0:
            raise RuntimeError(
                f"generic_benchmark failed (exit {p2_rc}) — see {bench_log}"
            )

    def stop(self):
        pass

    def clean(self):
        results_dir = self.config.get('results_dir')
        if results_dir and os.path.isdir(results_dir):
            Rm(results_dir).run()
        # Also sweep the auto-generated default-path variants created by start()
        Rm(f'/tmp/gpucompress_warpx_{self.pkg_id}_*').run()

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
        ncell = str(cfg['ncell']).strip()
        parts = ncell.split()
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(
                f"ncell must be three space-separated integers "
                f"(e.g. '32 32 256'), got {cfg['ncell']!r}"
            )
