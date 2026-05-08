#!/usr/bin/env bash
# Watch the Phase 4 blockers job (run_blockers.py); once it finishes,
# merge new Task A rows into results/phase2/task_a.jsonl, run the Qwen
# judge (resumes — only new rows are judged), then rebuild Phase 3 and
# the preregistered H5 verdict.
#
# Usage:
#   nohup bash scripts/watch_blockers_then_judge.sh > logs/watch_blockers_<ts>.log 2>&1 &
#
# Safe to leave running overnight: every step is resumable; if any step
# fails, downstream steps skip and the script exits non-zero.

set -u  # but NOT -e: we want to keep going even if a single step fails

BLOCKERS_PID=${BLOCKERS_PID:-2202333}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON=/specific/scratches/scratch/naamarozen/conda_envs/emotion-cot/bin/python
TS() { date '+%Y-%m-%d %H:%M:%S'; }

PHASE4_DIR="results/phase4/llama70b"
PHASE2_TASK_A="results/phase2/task_a.jsonl"

echo "[$(TS)] Watcher starting. Monitoring blockers PID=$BLOCKERS_PID"
echo "[$(TS)] Project dir: $PROJECT_DIR"

# ---- 1. Wait for blockers PID to exit ----
while kill -0 "$BLOCKERS_PID" 2>/dev/null; do
    FG=$(wc -l < "$PHASE4_DIR/finegrained.jsonl"        2>/dev/null || echo 0)
    EX=$(wc -l < "$PHASE4_DIR/extended_unsteered.jsonl" 2>/dev/null || echo 0)
    RD=$(wc -l < "$PHASE4_DIR/random_directions.jsonl"  2>/dev/null || echo 0)
    TI=$(wc -l < "$PHASE4_DIR/text_injection.jsonl"     2>/dev/null || echo 0)
    echo "[$(TS)] blockers alive — fg=$FG ext=$EX rand=$RD inj=$TI"
    sleep 300
done
echo "[$(TS)] ============================================"
echo "[$(TS)] Blockers PID $BLOCKERS_PID has exited."
echo "[$(TS)] ============================================"

# ---- 2. Wait briefly for any final flush, then snapshot counts ----
sleep 10
for f in finegrained extended_unsteered random_directions text_injection; do
    n=$(wc -l < "$PHASE4_DIR/$f.jsonl" 2>/dev/null || echo 0)
    echo "[$(TS)] $f.jsonl: $n rows"
done

# ---- 3. Merge new Task-A-shaped rows into results/phase2/task_a.jsonl ----
# Judge resumes by key; existing keys are skipped, so we only need to
# append NEW rows. We dedupe by key in case of double-running.
echo "[$(TS)] Merging Phase 4 rows into $PHASE2_TASK_A ..."
$PYTHON - <<'PYEOF'
import json, os
from pathlib import Path

phase4 = Path("results/phase4/llama70b")
target = Path("results/phase2/task_a.jsonl")

# Load existing keys
existing = set()
if target.exists():
    with open(target) as f:
        for line in f:
            try:
                existing.add(json.loads(line).get("key"))
            except Exception:
                pass

added = 0
src_files = [
    "finegrained.jsonl",
    "extended_unsteered.jsonl",
    # Skip random_directions and text_injection: their probe semantics
    # are different (random direction or text-injected), and they go
    # through a separate Phase-4 analysis pipeline.
]
with open(target, "a") as out:
    for fname in src_files:
        p = phase4 / fname
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                k = row.get("key")
                if not k or k in existing:
                    continue
                out.write(json.dumps(row) + "\n")
                existing.add(k)
                added += 1
print(f"merged {added} new rows into {target}")
PYEOF

# ---- 4. Run the Qwen judge (resumes from existing task_a_judged.jsonl) ----
echo "[$(TS)] Starting Qwen judge reclassification..."
JUDGE_LOG="logs/judge_postblockers_$(date +%Y%m%d_%H%M%S).log"
$PYTHON scripts/run_judge_reclassification.py --n-passes 3 --batch-size 16 --skip-task-b \
    > "$JUDGE_LOG" 2>&1
JUDGE_EXIT=$?
echo "[$(TS)] Judge exit=$JUDGE_EXIT (log: $JUDGE_LOG)"
if [ $JUDGE_EXIT -ne 0 ]; then
    echo "[$(TS)] JUDGE FAILED — skipping Phase 3 + H5 steps. Inspect $JUDGE_LOG"
    exit $JUDGE_EXIT
fi

# ---- 5. Rebuild Phase 3 measurements from new judged jsonl ----
echo "[$(TS)] Rebuilding Phase 3 measurements..."
$PYTHON scripts/03_run_phase3.py --model llama-70b 2>&1 | tail -40
P3_EXIT=${PIPESTATUS[0]}
echo "[$(TS)] Phase 3 exit=$P3_EXIT"
if [ $P3_EXIT -ne 0 ]; then
    echo "[$(TS)] Phase 3 FAILED — skipping H5"
    exit $P3_EXIT
fi

# ---- 6. Run preregistered H5 confirmatory verdict ----
echo "[$(TS)] Running preregistered H5 holdout evaluation..."
$PYTHON scripts/h5_holdout.py --model llama-70b 2>&1 | tail -25
H5_EXIT=${PIPESTATUS[0]}
echo "[$(TS)] H5 exit=$H5_EXIT"

# ---- 7. Symmetry check on refreshed measurements ----
echo "[$(TS)] Re-running judge symmetry check..."
$PYTHON scripts/judge_symmetry_check.py --model llama-70b 2>&1 | tail -20

echo "[$(TS)] ============================================"
echo "[$(TS)] WATCHER DONE."
echo "[$(TS)] Outputs:"
echo "[$(TS)]   results/phase2/task_a_judged.jsonl"
echo "[$(TS)]   results/phase3/llama70b/faithfulness_measurements.json"
echo "[$(TS)]   results/phase3/llama70b/h5_holdout_report.json   <-- preregistered verdict"
echo "[$(TS)]   results/phase3/llama70b/judge_symmetry.json"
echo "[$(TS)] ============================================"
