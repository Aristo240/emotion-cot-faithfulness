#!/usr/bin/env bash
# Phase B master runner — fire under nohup, walk away for a couple of days.
#
# Usage (from the project root):
#     nohup bash phaseB/run_all.sh > logs/phaseB_master.log 2>&1 &
#
# Each step is resumable. Each output JSONL is fsync'd line-by-line. If the
# job dies and you re-launch the same command, it picks up where it stopped
# (the per-step scripts skip `id`s already in the output).
#
# Pipeline (in order):
#     1. download sycophancy_feedback + sycophancy_are_you_sure
#     2. single-turn (02_run_with_probes.py)             — IN SCOPE, primary
#     3. two-turn pushback (02b_two_turn_pushback.py)    — IN SCOPE, secondary
#     4. two-agent dialogue (02c_two_agent_dialogue.py)  — OUT OF SCOPE (note in header)
#     5. judge for each (03 / 03b / 03c)
#     6. analyze each (04 / 04b / 04c)
#
# You can run `bash phaseB/scripts/snapshot.sh` at any point to get current
# analysis on whatever data has landed so far.
#
# Stop the job: `pkill -f phaseB/scripts/02 ; pkill -f phaseB/scripts/03`
#
set -euo pipefail
cd "$(dirname "$0")/.."

# Make sure HF_TOKEN etc. are available (Llama 3.1 70B is gated)
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

PY=${PY:-python3}
LOG_DIR=logs
mkdir -p "$LOG_DIR" phaseB/results

# Helper that timestamps the master log
log() { echo "[$(date '+%F %T')] [run_all] $*"; }

############################################################################
# 1. Download datasets (cheap, idempotent; safe to run every launch)
############################################################################
log "step 1: download datasets"
$PY phaseB/scripts/01_download_data.py --split feedback --n 1500 \
    >> "$LOG_DIR/phaseB_01.log" 2>&1
$PY phaseB/scripts/01_download_data.py --split are_you_sure --n 1500 \
    >> "$LOG_DIR/phaseB_01.log" 2>&1 || \
    log "WARNING: are_you_sure split failed to download — two-turn step will skip."

############################################################################
# 2. Single-turn V_internal extraction (IN SCOPE — primary)
############################################################################
log "step 2: single-turn run_with_probes (~4-6h)"
$PY phaseB/scripts/02_run_with_probes.py \
    --layer 53 --max-new-tokens 384 \
    --input phaseB/data/sycophancy_feedback.jsonl \
    --output phaseB/results/trials.jsonl \
    >> "$LOG_DIR/phaseB_02.log" 2>&1
log "step 2 DONE"

############################################################################
# 3. Two-turn pushback (IN SCOPE — secondary)
############################################################################
if [[ -f phaseB/data/sycophancy_are_you_sure.jsonl ]]; then
    log "step 3: two-turn pushback (~6-10h)"
    $PY phaseB/scripts/02b_two_turn_pushback.py \
        --layer 53 --max-new-tokens 384 \
        --input phaseB/data/sycophancy_are_you_sure.jsonl \
        --output phaseB/results/two_turn_trials.jsonl \
        >> "$LOG_DIR/phaseB_02b.log" 2>&1
    log "step 3 DONE"
else
    log "step 3 SKIPPED — sycophancy_are_you_sure dataset not present"
fi

############################################################################
# 4. Two-agent dialogue (OUT OF SCOPE per 02c header — kept for exploration)
############################################################################
log "step 4: two-agent dialogue (~4-6h, OUT OF SCOPE)"
$PY phaseB/scripts/02c_two_agent_dialogue.py \
    --layer 53 --rounds 4 --rollouts 30 \
    --output phaseB/results/two_agent_trials.jsonl \
    >> "$LOG_DIR/phaseB_02c.log" 2>&1
log "step 4 DONE"

############################################################################
# 5. Judge each output. Each call loads the judge model fresh, but resumable
#    — re-running the same script picks up at the first un-judged id.
############################################################################
log "step 5a: judge single-turn (~1h)"
$PY phaseB/scripts/03_judge_responses.py --n-passes 3 \
    >> "$LOG_DIR/phaseB_03.log" 2>&1
log "step 5a DONE"

if [[ -f phaseB/results/two_turn_trials.jsonl ]]; then
    log "step 5b: judge two-turn (~2h)"
    $PY phaseB/scripts/03b_judge_two_turn.py --n-passes 3 \
        >> "$LOG_DIR/phaseB_03b.log" 2>&1
    log "step 5b DONE"
fi

if [[ -f phaseB/results/two_agent_trials.jsonl ]]; then
    log "step 5c: judge two-agent (~2h)"
    $PY phaseB/scripts/03c_judge_two_agent.py --n-passes 3 \
        >> "$LOG_DIR/phaseB_03c.log" 2>&1
    log "step 5c DONE"
fi

############################################################################
# 6. Analyze each (cheap, idempotent)
############################################################################
log "step 6: analyze"
$PY phaseB/scripts/04_analyze.py >> "$LOG_DIR/phaseB_04.log" 2>&1 || true
$PY phaseB/scripts/04b_analyze_two_turn.py >> "$LOG_DIR/phaseB_04b.log" 2>&1 || true
$PY phaseB/scripts/04c_analyze_two_agent.py >> "$LOG_DIR/phaseB_04c.log" 2>&1 || true

log "ALL DONE."
log "Reports + plots in phaseB/results/analysis/"
ls -la phaseB/results/analysis/ || true
