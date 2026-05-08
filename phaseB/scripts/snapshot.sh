#!/usr/bin/env bash
# Run all three Phase B analyzers on whatever data has landed so far.
# Safe to run any time during a long nohup'd run_all.sh — it does not touch
# the generation or judge processes.
#
# Usage:
#     bash phaseB/scripts/snapshot.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-python3}
echo "[snapshot $(date '+%F %T')] reading current results/..."
echo "  trials.jsonl:               $(wc -l < phaseB/results/trials.jsonl 2>/dev/null || echo 0) lines"
echo "  trials_judged.jsonl:        $(wc -l < phaseB/results/trials_judged.jsonl 2>/dev/null || echo 0) lines"
echo "  two_turn_trials.jsonl:      $(wc -l < phaseB/results/two_turn_trials.jsonl 2>/dev/null || echo 0) lines"
echo "  two_turn_trials_judged:     $(wc -l < phaseB/results/two_turn_trials_judged.jsonl 2>/dev/null || echo 0) lines"
echo "  two_agent_trials.jsonl:     $(wc -l < phaseB/results/two_agent_trials.jsonl 2>/dev/null || echo 0) lines"
echo "  two_agent_trials_judged:    $(wc -l < phaseB/results/two_agent_trials_judged.jsonl 2>/dev/null || echo 0) lines"
echo

run_if_judged() {
    local file=$1; local script=$2; local label=$3
    if [[ -s "$file" ]]; then
        echo "=== $label ==="
        $PY "$script" 2>&1 | tail -20
        echo
    else
        echo "=== $label  SKIPPED (no judged data yet) ==="
        echo
    fi
}

run_if_judged phaseB/results/trials_judged.jsonl \
              phaseB/scripts/04_analyze.py "single-turn"
run_if_judged phaseB/results/two_turn_trials_judged.jsonl \
              phaseB/scripts/04b_analyze_two_turn.py "two-turn pushback"
run_if_judged phaseB/results/two_agent_trials_judged.jsonl \
              phaseB/scripts/04c_analyze_two_agent.py "two-agent (out of scope)"

echo "[snapshot done] plots updated in phaseB/results/analysis/"
ls -la phaseB/results/analysis/ 2>/dev/null || true
