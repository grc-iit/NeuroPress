"""
gpucompress_vpic_delta — VPIC Harris-sheet magnetic-reconnection benchmark
with GPUCompress integration, single-node Delta (Apptainer / Path B).

Requires ``gpucompress_base`` to precede this package in the pipeline YAML
so that HDF5, nvcomp and the GPUCompress library/weights/headers are
present in the shared build image.

Unlike the two-phase Nyx workload, VPIC is **single-phase**: the VPIC
benchmark-deck binary (``vpic_benchmark_deck.Linux``) runs every
compression phase (no-comp, fixed algorithms, nn*) inline inside one
simulation, driven entirely by environment variables.  The reference shell
script at ``bench_tests/vpic.sh`` is the canonical implementation;
this ``start()`` reproduces it 1:1.
"""
import os

from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, MpiExecInfo
from jarvis_cd.shell.process import Mkdir, Rm


# Cost-model weights per policy (same table as Nyx).
POLICY_WEIGHTS = {
    'ratio':    (0.0, 0.0, 1.0),
    'speed':    (1.0, 1.0, 0.0),
    'balanced': (1.0, 1.0, 1.0),
}

# Fixed + adaptive compression phases supported by the VPIC benchmark deck.
# Mirrors ALL_PHASES in bench_tests/vpic.sh.
ALL_PHASES = [
    'no-comp',
    'lz4', 'snappy', 'deflate', 'gdeflate', 'zstd',
    'ans', 'cascaded', 'bitcomp',
    'nn', 'nn-rl', 'nn-rl+exp50',
]
VALID_PHASES = set(ALL_PHASES) - {'no-comp'}  # 'no-comp' implied by hdf5_mode=default

# Absolute paths inside the built image. Must match build.sh / Dockerfile.deploy.
VPIC_BIN = '/opt/GPUCompress/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux'
WEIGHTS  = '/opt/GPUCompress/neural_net/weights/model.nnwt'

# Runtime LD_LIBRARY_PATH — Jarvis's auto-generated %environment points at
# /opt/<pkg>/install/lib (nonexistent in our SIF), so we prefix every Exec
# command with `env LD_LIBRARY_PATH=…` to override at exec time. Mirrors
# gpucompress_nyx_delta/pkg.py.
LD_LIBRARY_PATH = (
    '/.singularity.d/libs'            # host libcuda.so.1 bound via --nv
    ':/usr/local/cuda/lib64'          # CUDA runtime libs
    ':/opt/hdf5-install/lib'          # HDF5 2.0.0
    ':/opt/nvcomp/lib'                # nvcomp
    ':/opt/GPUCompress/build'         # libgpucompress + VOL/Filter .so
)


class GpucompressVpicDelta(Application):

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            # ── HDF5 mode / algorithm / policy ────────────────────────────
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

            # ── I/O volume parameters ──────────────────────────────────────
            {'name': 'nx',
             'msg': 'Grid cells per dim (VPIC_NX); data ≈ (NX+2)^3 × 64 B',
             'type': int, 'default': 64},
            {'name': 'timesteps',
             'msg': 'Number of benchmark write cycles (VPIC_TIMESTEPS)',
             'type': int, 'default': 3},
            {'name': 'sim_interval',
             'msg': 'Physics steps between writes (VPIC_SIM_INTERVAL)',
             'type': int, 'default': 190},
            {'name': 'chunk_mb',
             'msg': 'HDF5 chunk size in MB (VPIC_CHUNK_MB)',
             'type': int, 'default': 4},
            {'name': 'verify',
             'msg': 'Bitwise readback verify (VPIC_VERIFY): 1 = on, 0 = off',
             'type': int, 'default': 1},

            # ── Physics / data-variety parameters ─────────────────────────
            {'name': 'mi_me',
             'msg': 'Ion/electron mass ratio (VPIC_MI_ME, 1–400)',
             'type': int, 'default': 25},
            {'name': 'wpe_wce',
             'msg': 'Plasma/cyclotron frequency ratio (VPIC_WPE_WCE, 0.5–5)',
             'type': float, 'default': 3.0},
            {'name': 'ti_te',
             'msg': 'Ion/electron temperature ratio (VPIC_TI_TE, 0.1–10)',
             'type': float, 'default': 1.0},
            {'name': 'perturbation',
             'msg': 'Tearing-mode seed fraction of B0 (VPIC_PERTURBATION, 0–0.5)',
             'type': float, 'default': 0.1},
            {'name': 'guide_field',
             'msg': 'Out-of-plane guide field (VPIC_GUIDE_FIELD, 0–0.5)',
             'type': float, 'default': 0.0},
            {'name': 'nppc',
             'msg': 'Particles per cell (VPIC_NPPC)',
             'type': int, 'default': 2},
            {'name': 'warmup',
             'msg': 'Physics warmup steps before first snapshot (VPIC_WARMUP_STEPS)',
             'type': int, 'default': 100},

            # ── NN online-learning knobs (VPIC deck reads these directly) ─
            {'name': 'sgd_lr',
             'msg': 'SGD learning rate (VPIC_LR)',
             'type': float, 'default': 0.2},
            {'name': 'sgd_mape',
             'msg': 'MAPE threshold for SGD firing (VPIC_MAPE_THRESHOLD)',
             'type': float, 'default': 0.10},
            {'name': 'explore_k',
             'msg': 'Top-K exploration alternatives (VPIC_EXPLORE_K)',
             'type': int, 'default': 4},
            {'name': 'explore_thresh',
             'msg': 'Exploration error threshold (VPIC_EXPLORE_THRESH)',
             'type': float, 'default': 0.20},

            # ── MPI sizing ────────────────────────────────────────────────
            # VPIC requires VPIC_NX to be divisible by nprocs; default to 1
            # for the smoke-test path (NX=64 always works).
            {'name': 'nprocs',
             'msg': 'MPI processes (VPIC_NX must be divisible by this)',
             'type': int, 'default': 1},
            {'name': 'ppn',
             'msg': 'Processes per node',
             'type': int, 'default': 1},

            # ── Container build options ───────────────────────────────────
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

            # ── Runtime output path ───────────────────────────────────────
            # /tmp is bind-mounted into the apptainer instance by default,
            # so host and container see the same files there.
            {'name': 'results_dir',
             'msg': 'Output root (empty = /tmp/gpucompress_vpic_<pkg_id>_…)',
             'type': str, 'default': ''},

            # ── VOL-level contract (docs/reproducability.md §VOL Configuration) ──
            {'name': 'vol_mode',
             'msg': "GPUCOMPRESS_VOL_MODE: 'release' (default, NN + online SGD), "
                    "'bypass' (GPU->CPU passthrough, baseline I/O), "
                    "'trace' (per-chunk profiling of 32 configs, ~32x slower; "
                    "output feeds analysis/plot_trace.py and plot_e2e.py)",
             'type': str, 'default': 'release'},
            {'name': 'timing_csv_name',
             'msg': 'GPUCOMPRESS_TIMING_OUTPUT filename under results_dir '
                    '(e2e_ms + vol_ms CSV from gpucompress_vol_atexit)',
             'type': str, 'default': 'gpucompress_io_timing.csv'},
            {'name': 'results_dir_policy_suffix',
             'msg': 'Append "_<policy>" to results_dir at runtime so flipping '
                    "the policy knob between runs writes to distinct dirs "
                    "(used by figure_8 pipelines)",
             'type': bool, 'default': False},
            {'name': 'trace_csv_name',
             'msg': 'GPUCOMPRESS_TRACE_OUTPUT filename under results_dir '
                    '(only written when vol_mode=trace)',
             'type': str, 'default': 'gpucompress_trace.csv'},
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
        # NOTE: unused on the apptainer-native path (pipeline.py:1587 skips
        # deploy-phase Dockerfiles for apptainer builds). Kept correct for
        # docker/podman users.
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
            f"/tmp/gpucompress_vpic_{self.pkg_id}"
            f"_NX{cfg['nx']}_ts{cfg['timesteps']}_{cfg['hdf5_mode']}"
            f"{eb_tag}{verify_tag}"
        )
        # Expand $HOME / ~ so YAMLs can use portable paths.
        results_dir = os.path.expandvars(os.path.expanduser(results_dir))
        if cfg.get('results_dir_policy_suffix', False):
            results_dir = f"{results_dir}_{cfg['policy']}"
        Mkdir(results_dir).run()

        # Derive included / excluded phase list exactly as bench_tests/vpic.sh
        # does: in 'default' HDF5 mode run only no-comp; in 'vol' mode run the
        # single requested phase; exclude all others.
        include_phases = (
            'no-comp' if cfg['hdf5_mode'] == 'default' else cfg['phase']
        )
        exclude_phases = ','.join(
            p for p in ALL_PHASES
            if p not in set(include_phases.split(','))
        )

        vpic_log = f'{results_dir}/vpic_bench.log'

        # Build the full env var set that the VPIC benchmark deck reads via
        # getenv() (see benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx).
        vpic_env = dict(self.mod_env)
        vpic_env.update({
            # ── core GPUCompress ─────────────────────────────────────────
            'GPUCOMPRESS_DETAILED_TIMING': '1',
            'GPUCOMPRESS_WEIGHTS':         WEIGHTS,
            'HDF5_PLUGIN_PATH':            '/opt/GPUCompress/build',
            # ── I/O volume ───────────────────────────────────────────────
            'VPIC_NX':           str(cfg['nx']),
            'VPIC_NPPC':         str(cfg['nppc']),
            'VPIC_TIMESTEPS':    str(cfg['timesteps']),
            'VPIC_SIM_INTERVAL': str(cfg['sim_interval']),
            'VPIC_WARMUP_STEPS': str(cfg['warmup']),
            'VPIC_CHUNK_MB':     str(cfg['chunk_mb']),
            'VPIC_VERIFY':       str(verify),
            # ── Physics ──────────────────────────────────────────────────
            'VPIC_MI_ME':        str(cfg['mi_me']),
            'VPIC_WPE_WCE':      str(cfg['wpe_wce']),
            'VPIC_TI_TE':        str(cfg['ti_te']),
            'VPIC_PERTURBATION': str(cfg['perturbation']),
            'VPIC_GUIDE_FIELD':  str(cfg['guide_field']),
            # ── Compression ──────────────────────────────────────────────
            'VPIC_ERROR_BOUND':  str(error_bound),
            'VPIC_EXCLUDE':      exclude_phases,
            'VPIC_RESULTS_DIR':  results_dir,
            # VPIC_POLICIES drives policy selection in the deck; the deck
            # derives its own W0/W1/W2 from the policy string, so we do NOT
            # pass VPIC_W0/W1/W2 — matches bench_tests/vpic.sh exactly.
            'VPIC_POLICIES':     cfg['policy'],
            # ── NN online-learning knobs ─────────────────────────────────
            'VPIC_LR':              str(cfg['sgd_lr']),
            'VPIC_MAPE_THRESHOLD':  str(cfg['sgd_mape']),
            'VPIC_EXPLORE_K':       str(cfg['explore_k']),
            'VPIC_EXPLORE_THRESH':  str(cfg['explore_thresh']),
            # ── VOL-level contract ───────────────────────────────────────
            'GPUCOMPRESS_VOL_MODE':      cfg.get('vol_mode', 'release'),
            'GPUCOMPRESS_TIMING_OUTPUT': (
                f'{results_dir}/'
                f'{cfg.get("timing_csv_name", "gpucompress_io_timing.csv")}'
            ),
        })
        if cfg.get('vol_mode') == 'trace':
            vpic_env['GPUCOMPRESS_TRACE_OUTPUT'] = (
                f'{results_dir}/'
                f'{cfg.get("trace_csv_name", "gpucompress_trace.csv")}'
            )

        cmd = f'env LD_LIBRARY_PATH={LD_LIBRARY_PATH} {VPIC_BIN}'
        run = Exec(cmd, MpiExecInfo(
            nprocs=int(cfg['nprocs']),
            ppn=int(cfg['ppn']),
            hostfile=self.hostfile,
            port=self.ssh_port,
            container=self._container_engine,
            container_image=self.deploy_image_name(),
            shared_dir=self.shared_dir,
            private_dir=self.private_dir,
            env=vpic_env,
            gpu=cfg.get('use_gpu', True),
            pipe_stdout=vpic_log,
            pipe_stderr=vpic_log,
        ))
        run.run()

        # Accept non-zero exit if the log contains "normal exit" (VPIC convention).
        rc = max(run.exit_code.values()) if getattr(run, 'exit_code', None) else 0
        if rc != 0 and not self._log_has_normal_exit(vpic_log):
            raise RuntimeError(
                f"VPIC benchmark failed (exit {rc}) — see {vpic_log}"
            )

    def stop(self):
        pass

    def clean(self):
        results_dir = self.config.get('results_dir')
        if results_dir and os.path.isdir(results_dir):
            Rm(results_dir).run()
        # Sweep auto-generated default-path variants from start()
        Rm(f'/tmp/gpucompress_vpic_{self.pkg_id}_*').run()

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
        nprocs = int(cfg['nprocs'])
        nx = int(cfg['nx'])
        if nprocs > 0 and nx % nprocs != 0:
            raise ValueError(
                f"VPIC_NX ({nx}) must be divisible by nprocs ({nprocs})"
            )

    def _log_has_normal_exit(self, log_path):
        try:
            with open(log_path, 'r', errors='replace') as f:
                return 'normal exit' in f.read()
        except OSError:
            return False
