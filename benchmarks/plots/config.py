"""
Shared configuration for benchmark plotting scripts.

Directory layout:
  benchmarks/results/
    per_dataset/{dataset}/{policy}/     <- per-dataset figures
    cross_dataset/                      <- cross-dataset comparison figures
    paper_figures/                      <- selected figures for the paper

Environment variable overrides:
  GS_DIR, VPIC_DIR, SDR_DIR   - direct paths to result directories
  GS_L, GS_TS, GS_CHUNK       - Gray-Scott eval parameters
  VPIC_NX, VPIC_TS, VPIC_CHUNK - VPIC eval parameters
  SDR_CHUNK                    - SDRBench chunk size
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

# ── Eval result directories ──

def _resolve(env_override, default_path):
    override = os.environ.get(env_override)
    if override:
        return os.path.join(ROOT, override) if not os.path.isabs(override) else override
    return os.path.join(ROOT, default_path)

GS_L     = os.environ.get("GS_L", "400")
GS_CHUNK = os.environ.get("GS_CHUNK", "4")
GS_TS    = os.environ.get("GS_TS", "100")
GS_EVAL  = _resolve("GS_DIR", f"benchmarks/grayscott/results/eval_L{GS_L}_chunk{GS_CHUNK}mb_ts{GS_TS}")

VPIC_NX    = os.environ.get("VPIC_NX", "156")
VPIC_CHUNK = os.environ.get("VPIC_CHUNK", "4")
VPIC_TS    = os.environ.get("VPIC_TS", "100")
VPIC_EVAL  = _resolve("VPIC_DIR", f"benchmarks/vpic-kokkos/results/eval_NX{VPIC_NX}_chunk{VPIC_CHUNK}mb_ts{VPIC_TS}")

SDR_CHUNK = os.environ.get("SDR_CHUNK", "4")
SDR_EVAL  = _resolve("SDR_DIR", f"benchmarks/sdrbench/results/eval_chunk{SDR_CHUNK}mb")

AI_MODEL   = os.environ.get("AI_MODEL", "vit_b")
AI_DATASET = os.environ.get("AI_DATASET", "cifar10")
AI_CHUNK   = os.environ.get("AI_CHUNK", "4")
AI_EVAL    = _resolve("AI_DIR", f"benchmarks/ai_training/results/eval_{AI_MODEL}_{AI_DATASET}_chunk{AI_CHUNK}mb")

# Data lives in data/ai_training/, not data/sdrbench/

# ── Cost model policies ──

BALANCED   = "balanced_w1-1-1"
RATIO_ONLY = "ratio_only_w0-0-1"
SPEED_ONLY = "speed_only_w1-1-0"
ALL_POLICIES = [BALANCED, RATIO_ONLY, SPEED_ONLY]

# ── Output directories ──

RESULTS_DIR    = os.path.join(ROOT, "benchmarks/results")
PER_DATASET    = os.path.join(RESULTS_DIR, "per_dataset")
CROSS_DATASET  = os.path.join(RESULTS_DIR, "cross_dataset")
PAPER_FIGS     = os.path.join(RESULTS_DIR, "paper_figures")

# ── Dataset registry ──

DATASETS = {
    "gray_scott": {
        "eval_dir": GS_EVAL,
        "agg_csv": "benchmark_grayscott_vol.csv",
        "chunks_csv": "benchmark_grayscott_vol_chunks.csv",
        "timesteps_csv": "benchmark_grayscott_timesteps.csv",
        "timestep_chunks_csv": "benchmark_grayscott_timestep_chunks.csv",
    },
    "vpic": {
        "eval_dir": VPIC_EVAL,
        "agg_csv": "benchmark_vpic_deck.csv",
        "chunks_csv": "benchmark_vpic_deck_chunks.csv",
        "timesteps_csv": "benchmark_vpic_deck_timesteps.csv",
        "timestep_chunks_csv": "benchmark_vpic_deck_timestep_chunks.csv",
    },
}

# AI Training checkpoint datasets
AI_DATASETS = {
    "vit_b_cifar10": {"model": "vit_b_16", "params": "86M"},
    "gpt2_wikitext2": {"model": "gpt2", "params": "124M"},
}

# SDRBench datasets are auto-discovered from the eval directory
SDR_DATASETS = {
    "hurricane_isabel": {"dims": "100x500x500", "ext": ".bin.f32"},
    "nyx":              {"dims": "512x512x512", "ext": ".f32"},
    "cesm_atm":         {"dims": "1800x3600",   "ext": ".dat"},
}

# ── Helpers ──

def get_data_dir(dataset, policy=BALANCED):
    """Return the result directory for a dataset+policy."""
    if dataset in DATASETS:
        base = DATASETS[dataset]["eval_dir"]
    elif dataset in AI_DATASETS:
        base = AI_EVAL
    else:
        base = SDR_EVAL
    sub = os.path.join(base, policy)
    return sub if os.path.isdir(sub) else base

def get_csv(dataset, csv_type, policy=BALANCED):
    """Return path to a specific CSV for a dataset+policy."""
    d = get_data_dir(dataset, policy)
    if dataset in DATASETS:
        name = DATASETS[dataset].get(csv_type, "")
    else:
        name = f"benchmark_{dataset}_{csv_type.replace('_csv','')}.csv" if csv_type != "agg_csv" else f"benchmark_{dataset}.csv"
    return os.path.join(d, name)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def find_datasets_with_data(policy=BALANCED):
    """Return list of (dataset_name, data_dir) for datasets that have results."""
    found = []
    for ds in DATASETS:
        d = get_data_dir(ds, policy)
        agg = os.path.join(d, DATASETS[ds]["agg_csv"])
        if os.path.exists(agg):
            found.append((ds, d))
    # SDRBench
    import glob
    sdr = get_data_dir("hurricane_isabel", policy)
    if os.path.isdir(sdr):
        for f in sorted(glob.glob(os.path.join(sdr, "benchmark_*.csv"))):
            bn = os.path.basename(f)
            if "_chunks" not in bn and "_timesteps" not in bn:
                name = bn.replace("benchmark_", "").replace(".csv", "")
                found.append((name, sdr))
    return found
