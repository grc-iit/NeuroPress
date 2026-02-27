"""
Visualize HDF5 lossless benchmark: compression ratio, throughput, and NN adaptation.
Usage: python3 tests/benchmark_hdf5_viz.py [path/to/results.csv] [path/to/chunks.csv]
"""
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths relative to the script's own directory (works regardless of cwd)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV   = os.path.join(SCRIPT_DIR, 'benchmark_hdf5_results', 'benchmark_hdf5_results.csv')
DEFAULT_CHUNK = os.path.join(SCRIPT_DIR, 'benchmark_hdf5_results', 'benchmark_hdf5_chunks.csv')

CSV_PATH   = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
CHUNK_CSV  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CHUNK
OUT_DIR    = os.path.join(SCRIPT_DIR, 'benchmark_hdf5_results')
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV not found: {CSV_PATH}")
    print(f"  Run the benchmark first: ./build/benchmark_hdf5 <weights.nnwt>")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

# Load per-chunk data for adaptation plots (optional)
chunk_df = None
if os.path.exists(CHUNK_CSV):
    chunk_df = pd.read_csv(CHUNK_CSV)
    print(f"Loaded {len(chunk_df)} chunk rows from {CHUNK_CSV}")
else:
    print(f"No chunk CSV at {CHUNK_CSV} — skipping adaptation plots")

# Auto-detect patterns from the CSV data
PAT2LBL = {
    'constant': 'Constant', 'smooth_sine': 'Smooth Sine', 'ramp': 'Ramp',
    'gaussian': 'Gaussian', 'sparse': 'Sparse', 'step': 'Step',
    'hf_sine_noise': 'HF Sine+Noise', 'exp_decay': 'Exp Decay',
    'sawtooth': 'Sawtooth', 'mixed_freq': 'Mixed Freq',
    'lognormal': 'Log-Normal', 'impulse_train': 'Impulse Train',
    'mixed': 'Mixed (12 patterns)', 'uniform_ramp': 'Uniform Ramp',
    'contiguous': 'Contiguous (12 blocks)', 'cycling': 'Cycling (12 patterns)',
    'sine': 'Sine', 'random': 'Random',
}

PATTERNS = list(df['pattern'].unique())
LABELS   = [PAT2LBL.get(p, p.replace('_', ' ').title()) for p in PATTERNS]
print(f"Patterns in data: {PATTERNS}")

ALGOS    = ['lz4', 'snappy', 'deflate', 'gdeflate', 'zstd', 'ans', 'cascaded', 'bitcomp']
ALGO_LBL = ['LZ4', 'Snappy', 'Deflate', 'Gdeflate', 'Zstd', 'ANS', 'Cascaded', 'Bitcomp']

plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# ── Subsets ──

none_df   = df[df['mode'] == 'none']
static_df = df[df['mode'] == 'static']
nn_df     = df[df['mode'] == 'nn']

# ── Per-pattern aggregation ──

best_static, best_cfg = [], []
nn_ratio, nn_lbl = [], []
nn_orig_lbl, nn_explored, nn_sgd_fired = [], [], []

for pat in PATTERNS:
    s = static_df[static_df['pattern'] == pat]
    best_static.append(s['compression_ratio'].max() if len(s) else 1.0)
    if len(s):
        row = s.loc[s['compression_ratio'].idxmax()]
        best_cfg.append(row['algorithm'] + ('+shuf' if row.get('shuffle', 0) > 0 else ''))
    else:
        best_cfg.append('')

    n = nn_df[nn_df['pattern'] == pat]
    if len(n):
        r = n.iloc[0]
        nn_ratio.append(r['compression_ratio'])
        nn_lbl.append(r['algorithm'])
        nn_explored.append(int(r.get('explored', 0)))
        nn_sgd_fired.append(int(r.get('sgd_fired', 0)))
        orig = int(r.get('nn_original_action', -1))
        if orig >= 0:
            ai, sh = orig % 8, (orig // 16) % 2
            nn_orig_lbl.append(ALGOS[ai] + ('+shuf' if sh else ''))
        else:
            nn_orig_lbl.append('')
    else:
        nn_ratio.append(1.0); nn_lbl.append('')
        nn_orig_lbl.append(''); nn_explored.append(0); nn_sgd_fired.append(0)

x = np.arange(len(PATTERNS))

# ═══════════════════════════════════════════════════════════════
# Fig 1 — Compression Ratio: Best Static vs NN
# ═══════════════════════════════════════════════════════════════
w = 0.3
fig, ax = plt.subplots(figsize=(16, 7))
b1 = ax.bar(x - w/2, best_static, w, label='Best Static', color='#4A90D9')
b2 = ax.bar(x + w/2, nn_ratio,    w, label='NN Auto',     color='#5CB85C')
ax.set_yscale('log')
ax.set_ylabel('Compression Ratio (log)')
ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=30, ha='right')
ax.set_title('Lossless Compression Ratio: Best Static vs NN')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
ax.axhline(y=1, color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_ylim(bottom=0.8, top=ax.get_ylim()[1] * 5)

for bars, vals, c in [(b1, best_static, '#2C5F9E'), (b2, nn_ratio, '#2D7A2D')]:
    for bar, v in zip(bars, vals):
        if v > 1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                    f'{v:.0f}x' if v >= 10 else f'{v:.2f}x',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    rotation=90, color=c)

plt.tight_layout()
path = f'{OUT_DIR}/benchmark_ratio_comparison.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [1] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 2 — NN Analysis Table
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(20, 8))
ax.axis('off')

cols = ['Pattern', 'Best Static', 'Config',
        'NN Initial', 'NN Final', 'Ratio', 'NN/Best',
        'Explored', 'SGD']
rows = []
for i in range(len(PATTERNS)):
    pct = nn_ratio[i] / best_static[i] * 100 if best_static[i] > 0 else 0
    changed = nn_explored[i] and nn_orig_lbl[i] != nn_lbl[i]
    rows.append([
        LABELS[i],
        f'{best_static[i]:.2f}x', best_cfg[i],
        nn_orig_lbl[i], nn_lbl[i] if changed else '-',
        f'{nn_ratio[i]:.2f}x', f'{pct:.0f}%',
        'Yes' if nn_explored[i] else 'No',
        'Yes' if nn_sgd_fired[i] else 'No',
    ])

avg_pct = sum(nn_ratio[i] / best_static[i] * 100
              for i in range(len(PATTERNS)) if best_static[i] > 0) / len(PATTERNS)
n_expl = sum(nn_explored)
n_sgd  = sum(nn_sgd_fired)
n_chg  = sum(1 for i in range(len(PATTERNS))
             if nn_explored[i] and nn_orig_lbl[i] != nn_lbl[i])
rows.append(['AVERAGE', '', '', '', '',
             '', f'{avg_pct:.0f}%',
             f'{n_expl}/{len(PATTERNS)}', f'{n_sgd}/{len(PATTERNS)}'])

tbl = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.6)

for i, row in enumerate(rows):
    pct = float(row[6].replace('%', ''))
    tbl[i + 1, 6].set_facecolor('#d4edda' if pct >= 95 else '#fff3cd' if pct >= 80 else '#f8d7da')
    if i < len(PATTERNS):
        tbl[i + 1, 7].set_facecolor('#fff3cd' if row[7] == 'Yes' else '#d4edda')
        tbl[i + 1, 8].set_facecolor('#f8d7da' if row[8] == 'Yes' else '#d4edda')
        # Highlight NN Final when changed
        if row[4] != '-':
            tbl[i + 1, 4].set_facecolor('#fff3cd')

for j in range(len(cols)):
    tbl[len(rows), j].set_text_props(fontweight='bold')
    tbl[len(rows), j].set_facecolor('#e9ecef')
    tbl[0, j].set_facecolor('#343a40')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

ax.set_title('NN Prediction vs Best Static (NN Final shown only when exploration changed the choice)',
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
path = f'{OUT_DIR}/benchmark_nn_analysis.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [2] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 3 — Throughput: No Compression vs Best Static vs NN (Write & Read MB/s)
# ═══════════════════════════════════════════════════════════════
none_write, none_read = [], []
best_write, best_read = [], []
nn_write, nn_read = [], []

for pat in PATTERNS:
    nz = none_df[none_df['pattern'] == pat]
    if len(nz):
        r0 = nz.iloc[0]
        none_write.append(r0['write_mbps'])
        none_read.append(r0['read_mbps'])
    else:
        none_write.append(0); none_read.append(0)

    s = static_df[static_df['pattern'] == pat]
    if len(s):
        row = s.loc[s['compression_ratio'].idxmax()]
        best_write.append(row['write_mbps'])
        best_read.append(row['read_mbps'])
    else:
        best_write.append(0); best_read.append(0)

    n = nn_df[nn_df['pattern'] == pat]
    if len(n):
        r = n.iloc[0]
        nn_write.append(r['write_mbps'])
        nn_read.append(r['read_mbps'])
    else:
        nn_write.append(0); nn_read.append(0)

w = 0.25
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# Write throughput
b0 = ax1.bar(x - w, none_write, w, label='No Compression', color='#999999')
b1 = ax1.bar(x,     best_write, w, label='Best Static',    color='#4A90D9')
b2 = ax1.bar(x + w, nn_write,   w, label='NN Auto',        color='#5CB85C')
ax1.set_ylabel('Write Throughput (MB/s)')
ax1.set_xticks(x); ax1.set_xticklabels(LABELS, rotation=30, ha='right')
ax1.set_title('Write Throughput (compress + HDF5 write)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)
for bars in [b0, b1, b2]:
    for bar in bars:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=7)

# Read throughput
b3 = ax2.bar(x - w, none_read, w, label='No Compression', color='#999999')
b4 = ax2.bar(x,     best_read, w, label='Best Static',    color='#4A90D9')
b5 = ax2.bar(x + w, nn_read,   w, label='NN Auto',        color='#5CB85C')
ax2.set_ylabel('Read Throughput (MB/s)')
ax2.set_xticks(x); ax2.set_xticklabels(LABELS, rotation=30, ha='right')
ax2.set_title('Read Throughput (HDF5 read + decompress)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
for bars in [b3, b4, b5]:
    for bar in bars:
        if bar.get_height() > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=7)

plt.suptitle('Throughput: No Compression vs Best Static vs NN Auto', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
path = f'{OUT_DIR}/benchmark_throughput.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [3] {path}")

fig_num = 3

# ═══════════════════════════════════════════════════════════════
# Fig 4+ — NN Adaptation: Exploration Per Chunk + Windowed Rate
# ═══════════════════════════════════════════════════════════════

ADAPT_WINDOW = 16  # rolling / window size

def rolling_avg(arr, w):
    """Causal rolling average (expanding at the start)."""
    cum = np.cumsum(arr)
    cum = np.insert(cum, 0, 0)
    out = np.zeros(len(arr))
    for i in range(len(arr)):
        lo = max(0, i - w + 1)
        out[i] = (cum[i + 1] - cum[lo]) / (i - lo + 1)
    return out

if chunk_df is not None and len(chunk_df) > 0:
    chunk_patterns = chunk_df['pattern'].unique()
    has_chunk_pattern = 'chunk_pattern' in chunk_df.columns

    # Color palette for chunk patterns
    CPAT_COLORS = {
        'constant': '#e6194b', 'smooth_sine': '#3cb44b', 'ramp': '#4363d8',
        'gaussian': '#f58231', 'sparse': '#911eb4', 'step': '#42d4f4',
        'hf_sine_noise': '#f032e6', 'exp_decay': '#bfef45', 'sawtooth': '#fabed4',
        'mixed_freq': '#469990', 'lognormal': '#dcbeff', 'impulse_train': '#9A6324',
    }

    # Detect chunk size from aggregate CSV
    g_chunk_bytes_mb = 4  # default
    if 'original_bytes' in df.columns:
        none_rows = df[df['mode'] == 'none']
        if len(none_rows) > 0:
            total_bytes = none_rows.iloc[0]['original_bytes']
            n_ch = len(chunk_df)
            if n_ch > 0:
                g_chunk_bytes_mb = max(1, int(total_bytes / n_ch / (1024 * 1024)))

    for pat in chunk_patterns:
        pdf = chunk_df[chunk_df['pattern'] == pat].copy().reset_index(drop=True)
        n_chunks = len(pdf)
        if n_chunks == 0:
            continue

        chunks   = pdf['chunk_id'].values
        explored = pdf['explored'].values.astype(float)
        ratio    = pdf['ratio'].values

        expl_rolling  = rolling_avg(explored, ADAPT_WINDOW)
        ratio_rolling = rolling_avg(ratio, ADAPT_WINDOW)

        # Non-overlapping windows
        n_windows = n_chunks // ADAPT_WINDOW
        win_expl_pct = np.array([
            explored[i*ADAPT_WINDOW:(i+1)*ADAPT_WINDOW].mean() * 100
            for i in range(n_windows)])

        # ── Compute per-chunk APE against best ratio per chunk_pattern ──
        # Target = best (max) ratio achieved for each chunk_pattern across
        # the full run.  APE = |ratio - target| / target * 100.
        # For uniform mode (no chunk_pattern), use global best ratio.
        if has_chunk_pattern:
            cpats = pdf['chunk_pattern'].values
            best_per_cpat = {}
            for cp in set(cpats):
                best_per_cpat[cp] = ratio[cpats == cp].max()
            target = np.array([best_per_cpat[cp] for cp in cpats])
        else:
            target = np.full(n_chunks, ratio.max())

        ape = np.where(target > 0, np.abs(ratio - target) / target * 100.0, 0.0)
        mape_rolling = rolling_avg(ape, ADAPT_WINDOW)

        # Non-overlapping window MAPE
        win_mape = np.array([
            ape[i*ADAPT_WINDOW:(i+1)*ADAPT_WINDOW].mean()
            for i in range(n_windows)])

        # ── Detect pattern boundaries (contiguous blocks) ──
        pat_boundaries = []  # list of (chunk_idx, pattern_name)
        if has_chunk_pattern:
            prev_cp = None
            for i, cp in enumerate(cpats):
                if cp != prev_cp:
                    pat_boundaries.append((i, cp))
                    prev_cp = cp

        def draw_pattern_lines(ax, boundaries, ymin=None, ymax=None):
            """Draw vertical dashed lines at pattern transitions."""
            for idx, (ci, cp) in enumerate(boundaries):
                if ci == 0:
                    continue
                ax.axvline(x=ci, color='#444444', ls=':', lw=0.9, alpha=0.5,
                           zorder=4)

        fig_num += 1
        total_expl = int(explored.sum())
        has_exploration = total_expl > 0
        has_strip = has_chunk_pattern and len(set(cpats)) > 1

        # ── Build GridSpec layout ──
        from matplotlib.gridspec import GridSpec

        panel_specs = []  # (name, height_ratio)
        if has_exploration:
            panel_specs.append(('expl_top', 10))
        if has_strip:
            panel_specs.append(('strip', 1))
        panel_specs.append(('mape', 12))
        panel_specs.append(('ratio', 12))
        if has_exploration:
            panel_specs.append(('expl_bot', 8))

        names   = [s[0] for s in panel_specs]
        ratios  = [s[1] for s in panel_specs]
        n_panels = len(ratios)

        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(n_panels, 1, figure=fig, height_ratios=ratios,
                      hspace=0.08, top=0.93, bottom=0.05, left=0.07,
                      right=0.93)
        panel_map = {}
        for i, name in enumerate(names):
            panel_map[name] = fig.add_subplot(gs[i])

        fig.suptitle(
            f'NN Adaptation  —  {PAT2LBL.get(pat, pat)}  '
            f'({n_chunks:,} chunks x {g_chunk_bytes_mb} MB '
            f'= {n_chunks * g_chunk_bytes_mb:,} MB)',
            fontsize=17, fontweight='bold')

        # Shared x-axis setup helper
        def style_xaxis(ax, show_label=True):
            ax.set_xlim(0, n_chunks)
            ax.tick_params(axis='both', labelsize=11)
            if show_label:
                ax.set_xlabel('Chunk', fontsize=12, fontweight='bold')
            else:
                ax.set_xlabel('')
                ax.tick_params(axis='x', labelbottom=False)

        # ── Exploration triggered per chunk ──
        if has_exploration:
            ax_e = panel_map['expl_top']
            expl_idx   = np.where(explored > 0)[0]
            noexpl_idx = np.where(explored == 0)[0]
            ax_e.bar(noexpl_idx, np.ones(len(noexpl_idx)), width=1.0,
                     color='#d4edda', alpha=0.5, label='No exploration')
            ax_e.bar(expl_idx, np.ones(len(expl_idx)), width=1.0,
                     color='#f5c6cb', alpha=0.8, label='Exploration triggered')
            ax_e.plot(chunks, expl_rolling * 100, color='#c0392b', lw=2.5,
                      label=f'Rolling avg ({ADAPT_WINDOW}-chunk)')
            ax_e.set_ylabel('Exploration (%)', fontsize=12, fontweight='bold')
            ax_e.set_title('Exploration Triggered Per Chunk',
                           fontsize=14, fontweight='bold', pad=8)
            ax_e.set_ylim(-5, 110)
            ax_e.legend(fontsize=10, loc='upper right', framealpha=0.9)
            ax_e.grid(axis='y', alpha=0.25)
            ax_e.text(0.01, 0.88,
                      f'{total_expl}/{n_chunks} explored '
                      f'({100*total_expl/n_chunks:.0f}%)',
                      transform=ax_e.transAxes, fontsize=11, fontweight='bold',
                      bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc',
                                boxstyle='round,pad=0.3'))
            draw_pattern_lines(ax_e, pat_boundaries, ymax=110)
            style_xaxis(ax_e, show_label=False)

        # ── Pattern strip ──
        if has_strip:
            ax_s = panel_map['strip']
            strip_colors = np.array([
                matplotlib.colors.to_rgba(CPAT_COLORS.get(cp, '#888'))
                for cp in cpats
            ]).reshape(1, n_chunks, 4)
            ax_s.imshow(strip_colors, aspect='auto',
                        extent=[0, n_chunks, 0, 1], interpolation='nearest')
            ax_s.set_xlim(0, n_chunks)
            ax_s.set_yticks([])
            ax_s.tick_params(axis='x', labelbottom=False)
            ax_s.set_ylabel('Pattern', fontsize=9, fontweight='bold',
                            rotation=0, labelpad=40, va='center')

        # ── MAPE panel ──
        ax_m = panel_map['mape']
        if has_chunk_pattern:
            for cp in sorted(set(cpats)):
                mask = cpats == cp
                c = CPAT_COLORS.get(cp, '#e67e22')
                ax_m.scatter(chunks[mask], ape[mask], s=6, alpha=0.30,
                             color=c, zorder=2, edgecolors='none')
        else:
            ax_m.scatter(chunks, ape, s=6, alpha=0.25, color='#e67e22',
                         zorder=2, edgecolors='none')

        ax_m.plot(chunks, mape_rolling, color='#c0392b', lw=2.5, zorder=3,
                  label=f'Rolling MAPE ({ADAPT_WINDOW}-chunk)')

        q1 = min(n_chunks // 4, n_chunks)
        q4_start = max(n_chunks - n_chunks // 4, 0)
        mape_first_q = ape[:q1].mean() if q1 > 0 else 0
        mape_last_q  = ape[q4_start:].mean() if q4_start < n_chunks else 0
        ax_m.text(0.01, 0.90,
                  f'First 25%: {mape_first_q:.1f}%   '
                  f'Last 25%: {mape_last_q:.1f}%   '
                  f'Overall: {ape.mean():.1f}%',
                  transform=ax_m.transAxes, fontsize=11, fontweight='bold',
                  bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ccc',
                            boxstyle='round,pad=0.3'))

        if n_windows > 0:
            win_x = np.array([(i + 0.5) * ADAPT_WINDOW for i in range(n_windows)])
            ax_m2 = ax_m.twinx()
            ax_m2.bar(win_x, win_mape, width=ADAPT_WINDOW * 0.9,
                      alpha=0.10, color='#8e44ad', zorder=1)
            ax_m2.set_ylabel('Window MAPE (%)', fontsize=10,
                             color='#8e44ad')
            ax_m2.tick_params(axis='y', labelcolor='#8e44ad', labelsize=10)
            ymax = max(ape.max() * 1.1, 10)
            ax_m.set_ylim(0, ymax)
            ax_m2.set_ylim(0, ymax)

        ax_m.set_ylabel('APE (%)', fontsize=12, fontweight='bold')
        ax_m.set_title('MAPE: Ratio Error vs Best-Per-Pattern  (lower = better)',
                       fontsize=14, fontweight='bold', pad=8)
        ax_m.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax_m.grid(axis='y', alpha=0.25)
        draw_pattern_lines(ax_m, pat_boundaries, ymax=ax_m.get_ylim()[1])
        style_xaxis(ax_m, show_label=False)

        # ── Compression ratio convergence ──
        ax_r = panel_map['ratio']
        if has_chunk_pattern:
            unique_cpats = sorted(set(cpats))
            for cp in unique_cpats:
                mask = cpats == cp
                c = CPAT_COLORS.get(cp, '#2980b9')
                ax_r.scatter(chunks[mask], ratio[mask], s=8, alpha=0.40,
                             color=c, zorder=2, edgecolors='none')
            if 1 < len(unique_cpats) <= 12:
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=CPAT_COLORS.get(cp, '#2980b9'),
                           markersize=6, label=PAT2LBL.get(cp, cp))
                           for cp in unique_cpats]
                leg = ax_r.legend(handles=handles, fontsize=8, loc='upper right',
                                  ncol=3, title='Chunk pattern',
                                  title_fontsize=9, framealpha=0.95,
                                  borderpad=0.4, handletextpad=0.3,
                                  columnspacing=0.8)
                leg.get_frame().set_linewidth(0.5)
        else:
            ax_r.scatter(chunks, ratio, s=8, alpha=0.40, color='#2980b9',
                         zorder=2, edgecolors='none')

        ax_r.plot(chunks, ratio_rolling, color='#e74c3c', lw=2.5, zorder=3,
                  label=f'Rolling avg ({ADAPT_WINDOW}-chunk)')
        if n_chunks > ADAPT_WINDOW:
            final_avg = ratio[-ADAPT_WINDOW:].mean()
            ax_r.axhline(y=final_avg, color='#27ae60', ls='--', lw=2.0,
                         alpha=0.7, label=f'Final avg: {final_avg:.1f}x')
        ax_r.set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
        ax_r.set_title('Compression Ratio Convergence',
                       fontsize=14, fontweight='bold', pad=8)
        ax_r.grid(alpha=0.25)
        draw_pattern_lines(ax_r, pat_boundaries, ymax=ax_r.get_ylim()[1])
        is_last = not has_exploration
        style_xaxis(ax_r, show_label=is_last)

        # ── Exploration rate per window ──
        if has_exploration:
            ax_b = panel_map['expl_bot']
            win_labels = [f'{i*ADAPT_WINDOW+1}-{(i+1)*ADAPT_WINDOW}'
                          for i in range(n_windows)]
            colors_bar = ['#e74c3c' if p > 50 else '#f39c12' if p > 0
                          else '#27ae60' for p in win_expl_pct]
            ax_b.bar(range(n_windows), win_expl_pct, color=colors_bar,
                     edgecolor='white', lw=0.5)
            ax_b.set_xticks(range(n_windows))
            step = max(1, n_windows // 20)
            ax_b.set_xticklabels(
                [win_labels[i] if i % step == 0 else ''
                 for i in range(n_windows)],
                rotation=45, ha='right', fontsize=9)
            ax_b.set_xlabel('Chunk Window', fontsize=12, fontweight='bold')
            ax_b.set_ylabel('Exploration (%)', fontsize=12, fontweight='bold')
            ax_b.set_title(
                f'Exploration Rate per {ADAPT_WINDOW}-Chunk Window',
                fontsize=14, fontweight='bold', pad=8)
            ax_b.set_ylim(0, 105)
            ax_b.grid(axis='y', alpha=0.25)
            ax_b.tick_params(axis='y', labelsize=11)

            zero_start = None
            for i in range(n_windows):
                if win_expl_pct[i] == 0 and zero_start is None:
                    zero_start = i
                elif win_expl_pct[i] > 0:
                    zero_start = None
            if zero_start is not None and zero_start < n_windows - 1:
                ax_b.axvspan(zero_start - 0.5, n_windows - 0.5, alpha=0.08,
                             color='green', zorder=0)
                ax_b.text((zero_start + n_windows) / 2, 50, 'Model adapted',
                          ha='center', va='center', fontsize=14,
                          color='#27ae60', fontweight='bold', alpha=0.6)

        path = f'{OUT_DIR}/adaptation_{pat}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  [{fig_num}/{fig_num}] {path}")

    # Adaptation summary
    print("\n=== Adaptation Summary ===")
    for pat in chunk_patterns:
        pdf = chunk_df[chunk_df['pattern'] == pat]
        n = len(pdf)
        n_expl = int(pdf['explored'].sum())
        expl_chunks = pdf[pdf['explored'] > 0]['chunk_id']
        last_expl = int(expl_chunks.max()) if len(expl_chunks) > 0 else -1
        print(f"  {pat}: {n_expl}/{n} explored ({100*n_expl/n:.0f}%), "
              f"last exploration at chunk {last_expl}")

    if has_chunk_pattern and len(chunk_df['chunk_pattern'].unique()) > 1:
        print("\n=== Per-Pattern Compression Summary ===")
        for cp in sorted(chunk_df['chunk_pattern'].unique()):
            sub = chunk_df[chunk_df['chunk_pattern'] == cp]
            med = sub['ratio'].median()
            mean = sub['ratio'].mean()
            n_e = int(sub['explored'].sum())
            print(f"  {PAT2LBL.get(cp, cp):20s}: median {med:7.2f}x, mean {mean:7.2f}x, "
                  f"explored {n_e}/{len(sub)}")

print(f"\nDone — {fig_num} figures saved to {OUT_DIR}/")
