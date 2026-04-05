#!/bin/bash
# Run all Section 4 paper evaluations sequentially
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SGD_LR=${SGD_LR:-0.2}
POLICY=${POLICY:-balanced}
ERROR_BOUND=${ERROR_BOUND:-0.01}
export SGD_LR POLICY ERROR_BOUND

echo "============================================================"
echo "Section 4: Full Paper Evaluation Pipeline"
echo "  Policy: $POLICY  LR: $SGD_LR  Error bound: $ERROR_BOUND"
echo "============================================================"

# 4.2.1 Exploration Threshold Sweep (~2.5 hours)
echo ""
echo ">>> 4.2.1 Exploration Threshold Sweep"
# Error bound tag for directory names
if [ "$ERROR_BOUND" = "0" ] || [ "$ERROR_BOUND" = "0.0" ]; then
    EB_TAG="lossless"
else
    EB_TAG="eb${ERROR_BOUND}"
fi

bash "$SCRIPT_DIR/4.2.1_eval_exploration_threshold.sh"
python3 "$SCRIPT_DIR/4.2.1_plot_threshold_sweep.py" "$SCRIPT_DIR/results/threshold_sweep_${POLICY}_${EB_TAG}_lr${SGD_LR}"

# 4.2.1 RL Adaptiveness (~15 min)
echo ""
echo ">>> 4.2.1 RL Adaptiveness on Unseen Workloads"
bash "$SCRIPT_DIR/4.2.1_eval_rl_adaptiveness.sh"
python3 "$SCRIPT_DIR/4.2.1_plot_rl_adaptiveness.py" "$SCRIPT_DIR/results/rl_adaptiveness_${POLICY}_${EB_TAG}_lr${SGD_LR}"

# 4.2.1 VPIC Threshold Sweep (~12 min)
echo ""
echo ">>> 4.2.1 VPIC Threshold Sweep"
VPIC_ERROR_BOUND=$ERROR_BOUND bash "$SCRIPT_DIR/4.2.1_eval_vpic_threshold_sweep.sh"
python3 "$SCRIPT_DIR/4.2.1_plot_threshold_sweep.py" "$SCRIPT_DIR/results/vpic_threshold_sweep_${POLICY}_${EB_TAG}_lr${SGD_LR}"

# 4.2.1 VPIC RL Adaptiveness (~5 min)
echo ""
echo ">>> 4.2.1 VPIC RL Adaptiveness"
VPIC_ERROR_BOUND=$ERROR_BOUND bash "$SCRIPT_DIR/4.2.1_eval_vpic_adaptiveness.sh"
python3 "$SCRIPT_DIR/4.2.1_plot_rl_adaptiveness.py" "$SCRIPT_DIR/results/vpic_adaptiveness_${POLICY}_${EB_TAG}_lr${SGD_LR}"

# 4.2.1 AI Workload Adaptiveness (~30 min) — run separately:
#   bash benchmarks/Paper_Evaluations/4/4.2.1_eval_ai_workloads.sh

echo ""
echo "============================================================"
echo "All Section 4 evaluations complete."
echo "  Policy: $POLICY  LR: $SGD_LR"
echo "Results: $SCRIPT_DIR/results/"
echo "============================================================"
