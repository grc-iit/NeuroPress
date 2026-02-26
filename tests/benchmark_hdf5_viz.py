"""
Visualize HDF5 lossless benchmark: compression ratio and throughput analysis.
Usage: python3 tests/benchmark_hdf5_viz.py [path/to/results.csv]
"""
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else 'tests/benchmark_hdf5_results/benchmark_hdf5_results.csv'
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_hdf5_results')
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")

PATTERNS = ['ramp', 'sparse']
LABELS   = ['Ramp', 'Sparse']
ALGOS    = ['lz4', 'snappy', 'deflate', 'gdeflate', 'zstd', 'ans', 'cascaded', 'bitcomp']
ALGO_LBL = ['LZ4', 'Snappy', 'Deflate', 'Gdeflate', 'Zstd', 'ANS', 'Cascaded', 'Bitcomp']

plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.facecolor'] = 'white'

# ── Subsets ──

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
print(f"  [1/6] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 2 — NN / Best Static (%)
# ═══════════════════════════════════════════════════════════════
eff = [n / b * 100 if b > 0 else 0 for n, b in zip(nn_ratio, best_static)]

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#5CB85C' if e >= 95 else '#F0AD4E' if e >= 50 else '#D9534F' for e in eff]
bars = ax.bar(x, eff, 0.5, color=colors)
ax.axhline(y=100, color='red', ls='--', lw=1.5, label='100% = Best Static')
ax.set_ylabel('NN / Best Static (%)'); ax.set_xticks(x)
ax.set_xticklabels(LABELS, rotation=30, ha='right')
ax.set_title('NN Efficiency: Ratio as % of Best Static')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(eff) * 1.15 if max(eff) > 0 else 110)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.0f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
avg = sum(eff) / len(eff)
ax.axhline(y=avg, color='blue', ls=':', lw=1, alpha=0.6)
ax.text(len(x) - 0.5, avg + 2, f'avg {avg:.0f}%', color='blue', fontsize=9)

plt.tight_layout()
path = f'{OUT_DIR}/benchmark_nn_efficiency.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [2/6] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 3 — Heatmap: No-Shuffle vs Shuffle
# ═══════════════════════════════════════════════════════════════
hm_ns = np.ones((len(PATTERNS), len(ALGOS)))
hm_s4 = np.ones((len(PATTERNS), len(ALGOS)))

for i, pat in enumerate(PATTERNS):
    for j, algo in enumerate(ALGOS):
        for hm, sv in [(hm_ns, 0), (hm_s4, 4)]:
            sub = static_df[(static_df['pattern'] == pat) &
                            (static_df['algorithm'] == algo) &
                            (static_df['shuffle'] == sv)]
            if len(sub):
                hm[i, j] = sub['compression_ratio'].max()

vmax = max(np.log10(np.clip(hm_ns, 1, None)).max(),
           np.log10(np.clip(hm_s4, 1, None)).max())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
for ax, hm, t in [(a1, hm_ns, 'No Shuffle'), (a2, hm_s4, 'Shuffle (4-byte)')]:
    im = ax.imshow(np.log10(np.clip(hm, 1, None)), cmap='YlOrRd',
                   aspect='auto', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(ALGOS))); ax.set_xticklabels(ALGO_LBL, rotation=45, ha='right')
    ax.set_yticks(range(len(PATTERNS))); ax.set_yticklabels(LABELS)
    ax.set_title(f'Lossless Ratio — {t}', fontsize=11, fontweight='bold')
    for i in range(len(PATTERNS)):
        for j in range(len(ALGOS)):
            v = hm[i, j]
            txt = f'{v:.1f}x' if v < 100 else f'{v:.0f}x'
            c = 'white' if np.log10(max(v, 1)) > vmax * 0.55 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7, color=c)
cbar = fig.colorbar(im, ax=[a1, a2], shrink=0.8, pad=0.02)
cbar.set_label('log10(Compression Ratio)')
plt.suptitle('Lossless Ratio by Algorithm x Pattern', fontsize=13, fontweight='bold', y=0.99)
fig.subplots_adjust(top=0.9, right=0.88, wspace=0.15)
path = f'{OUT_DIR}/benchmark_algo_heatmap.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [3/6] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 4 — Per-Pattern: Algos vs NN (subplots)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (pat, lbl) in enumerate(zip(PATTERNS, LABELS)):
    ax = axes.flat[idx]
    sub = static_df[static_df['pattern'] == pat]
    ns, s4 = [], []
    for algo in ALGOS:
        for lst, sv in [(ns, 0), (s4, 4)]:
            s = sub[(sub['algorithm'] == algo) & (sub['shuffle'] == sv)]
            lst.append(s['compression_ratio'].max() if len(s) else 1.0)

    nn_v = nn_ratio[idx]
    n, w = len(ALGOS), 0.3
    ax.bar(np.arange(n) - w/2, ns, w, label='No Shuffle', color='#4A90D9', alpha=0.85)
    ax.bar(np.arange(n) + w/2, s4, w, label='Shuffle 4B', color='#F0AD4E', alpha=0.85)
    ax.bar([n], [nn_v], w * 2, label='NN', color='#5CB85C')
    ax.text(n, nn_v * 1.05, f'{nn_v:.0f}x' if nn_v >= 10 else f'{nn_v:.2f}x',
            ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(ALGO_LBL + ['NN'], rotation=45, ha='right', fontsize=6)
    ax.set_title(lbl, fontsize=10, fontweight='bold')
    ax.set_ylabel('Ratio', fontsize=8)
    all_v = ns + s4 + [nn_v]
    if max(all_v) / max(min(v for v in all_v if v > 0), 0.1) > 20:
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=6, loc='upper left')

plt.suptitle('Per-Pattern Lossless Ratio: All Algorithms vs NN', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
path = f'{OUT_DIR}/benchmark_per_algo_vs_nn.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [4/6] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 5 — NN Analysis Table
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(20, 5))
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
print(f"  [5/6] {path}")

# ═══════════════════════════════════════════════════════════════
# Fig 6 — Throughput: Best Static vs NN (Write & Read MB/s)
# ═══════════════════════════════════════════════════════════════
best_write, best_read = [], []
nn_write, nn_read = [], []

for pat in PATTERNS:
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

w = 0.2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

# Write throughput
b1 = ax1.bar(x - w/2, best_write, w, label='Best Static', color='#4A90D9')
b2 = ax1.bar(x + w/2, nn_write,   w, label='NN Auto',     color='#5CB85C')
ax1.set_ylabel('Write Throughput (MB/s)')
ax1.set_xticks(x); ax1.set_xticklabels(LABELS, rotation=30, ha='right')
ax1.set_title('Write Throughput (compress + HDF5 write)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)
for bars in [b1, b2]:
    for bar in bars:
        if bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=7)

# Read throughput
b3 = ax2.bar(x - w/2, best_read, w, label='Best Static', color='#4A90D9')
b4 = ax2.bar(x + w/2, nn_read,   w, label='NN Auto',     color='#5CB85C')
ax2.set_ylabel('Read Throughput (MB/s)')
ax2.set_xticks(x); ax2.set_xticklabels(LABELS, rotation=30, ha='right')
ax2.set_title('Read Throughput (HDF5 read + decompress)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
for bars in [b3, b4]:
    for bar in bars:
        if bar.get_height() > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=7)

plt.suptitle('Throughput: Best Static Config vs NN Auto', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
path = f'{OUT_DIR}/benchmark_throughput.png'
plt.savefig(path, bbox_inches='tight'); plt.close()
print(f"  [6/6] {path}")

print(f"\nDone — 6 figures saved to {OUT_DIR}/")
