#!/bin/bash
# Watch for Phase 2 completion, then automatically run judge reclassification.
#
# Usage:
#   nohup bash scripts/watch_and_judge.sh > logs/watch_and_judge.log 2>&1 &
#
# This script:
#   1. Polls every 60s to check if phase2_steering.py (PID 1133479) is still running
#   2. Once it finishes, runs the Qwen judge reclassification on all Phase 2 data
#   3. Logs everything to logs/watch_and_judge.log

PHASE2_PID=1133479
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher started. Monitoring PID $PHASE2_PID"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Project dir: $PROJECT_DIR"

# Phase 1: Wait for Phase 2 to finish
while kill -0 $PHASE2_PID 2>/dev/null; do
    TASK_A=$(wc -l < results/phase2/task_a.jsonl 2>/dev/null || echo 0)
    TASK_B=$(wc -l < results/phase2/task_b.jsonl 2>/dev/null || echo 0)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 still running (PID $PHASE2_PID). Task A: $TASK_A, Task B: $TASK_B"
    sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2 FINISHED. Final counts:"
TASK_A=$(wc -l < results/phase2/task_a.jsonl 2>/dev/null || echo 0)
TASK_B=$(wc -l < results/phase2/task_b.jsonl 2>/dev/null || echo 0)
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Task A: $TASK_A entries"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Task B: $TASK_B entries"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================"

# Phase 2: Run judge reclassification
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting judge reclassification (Qwen 2.5 72B)..."
python scripts/run_judge_reclassification.py --n-passes 3 --batch-size 16
JUDGE_EXIT=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================================"
if [ $JUDGE_EXIT -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Judge reclassification COMPLETED SUCCESSFULLY"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Judge reclassification FAILED (exit code $JUDGE_EXIT)"
fi

# Summary
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output files:"
for f in results/phase2/task_a_judged.jsonl results/phase2/task_b_judged.jsonl results/phase2/judge_reliability.json; do
    if [ -f "$f" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')]   $f ($(wc -l < "$f" 2>/dev/null || echo '?') lines)"
    fi
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher done."
