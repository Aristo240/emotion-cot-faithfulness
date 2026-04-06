#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh <phase> <model> [extra args...]
#
# Examples:
#   ./run.sh pilot   llama-70b
#   ./run.sh phase1  llama-70b
#   ./run.sh phase2  llama-70b --quick
#   ./run.sh phase2  llama-70b
#   ./run.sh phase3  llama-70b
#   ./run.sh phase3  llama-70b --skip-judge
#   ./run.sh phase4  llama-70b
#   ./run.sh phase4  llama-70b --control random

PHASE="${1:?Usage: ./run.sh <pilot|phase1|phase2|phase3|phase4> <model> [extra args...]}"
MODEL="${2:?Usage: ./run.sh <phase> <model> [extra args...]}"
shift 2
EXTRA_ARGS="$*"

# ── Resolve project root (where this script lives) ─────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ── HuggingFace auth (gated models like Llama) ─────────────
# Token loaded from .env (not committed to git)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# ── GPU config ──────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ── Map phase name to script ───────────────────────────────
case "$PHASE" in
    pilot)  SCRIPT="scripts/00_pilot.py" ;;
    phase1) SCRIPT="scripts/01_run_phase1.py" ;;
    phase2) SCRIPT="scripts/02_run_phase2.py" ;;
    phase3) SCRIPT="scripts/03_run_phase3.py" ;;
    phase4) SCRIPT="scripts/04_run_controls.py" ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: pilot, phase1, phase2, phase3, phase4"
        exit 1
        ;;
esac

# ── Log directory ───────────────────────────────────────────
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOG_DIR/${PHASE}_${MODEL}_${TIMESTAMP}.log"

# ── Launch with nohup ───────────────────────────────────────
echo "Launching: $PHASE  model=$MODEL  extra=$EXTRA_ARGS"
echo "GPUs:      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Log:       $LOGFILE"
echo ""

PYTHON="/specific/scratches/scratch/naamarozen/conda_envs/emotion-cot/bin/python"

nohup "$PYTHON" "$SCRIPT" --model "$MODEL" $EXTRA_ARGS \
    > "$LOGFILE" 2>&1 &

PID=$!
echo "Started as PID $PID"
echo "Monitor with:  tail -f $LOGFILE"
