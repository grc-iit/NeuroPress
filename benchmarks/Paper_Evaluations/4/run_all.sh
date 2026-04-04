#!/bin/bash
# Run all Section 4 paper evaluations sequentially
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "Section 4: Full Paper Evaluation Pipeline"
echo "============================================================"

# 4.2.1 Exploration Threshold Sweep (~2.5 hours)
echo ""
echo ">>> 4.2.1 Exploration Threshold Sweep"
bash "$SCRIPT_DIR/4.2.1_eval_exploration_threshold.sh"
python3 "$SCRIPT_DIR/4.2.1_plot_threshold_sweep.py"

# 4.2.1 RL Adaptiveness (~15 min)
echo ""
echo ">>> 4.2.1 RL Adaptiveness on Unseen Workloads"
bash "$SCRIPT_DIR/4.2.1_eval_rl_adaptiveness.sh"
python3 "$SCRIPT_DIR/4.2.1_plot_rl_adaptiveness.py"

echo ""
echo "============================================================"
echo "All Section 4 evaluations complete."
echo "Results: $SCRIPT_DIR/results/"
echo "============================================================"
