# Phase B — three experiments, one nohup

Three experiments, sharing the same Llama 3.1 70B + Qwen 2.5 72B + 50 emotion vectors stack.

| # | Experiment | Scope for the workshop | Outputs |
|---|---|---|---|
| **B1** | Single-turn V_internal as a sycophancy monitor on Anthropic `sycophancy_feedback`. | **In scope (primary)** — adds external validity for H5. | `results/trials.jsonl` → `trials_judged.jsonl` → `analysis/phaseB_*.png` |
| **B2** | Two-turn pushback on Anthropic `sycophancy_are_you_sure`: does V_internal *drift* between turns predict capitulation? | **In scope (secondary)** — one extra section + figure. | `results/two_turn_trials.jsonl` → `two_turn_trials_judged.jsonl` → `analysis/two_turn_*.png` |
| **B3** | Two-agent dialogue: a stressed "manager" pressures an "engineer" Llama; track engineer probe drift. | **⚠️ OUT OF SCOPE** — kept here for exploration / Phase 6. The script header explains why. | `results/two_agent_trials.jsonl` → `two_agent_trials_judged.jsonl` → `analysis/two_agent_*.png` |

## One-line launch (Sunday morning)

```bash
cd /specific/scratches/scratch/naamarozen/emotion-cot-faithfulness
nohup bash phaseB/run_all.sh > logs/phaseB_master.log 2>&1 &
echo $! > logs/phaseB_master.pid
disown                # so it survives ssh logout
tail -f logs/phaseB_master.log     # detach with Ctrl-C; the job keeps running
```

That single command runs the full pipeline (download → B1 → B2 → B3 → judge → analyze). The whole thing is 24-36 hours unattended on 8x V100. No babysitting needed.

### Resume after a crash

Re-running the same nohup command picks up at the first un-processed `id` in every step. Each output JSONL is fsync'd line-by-line, so partial output is always usable — never delete partial files manually.

### Get progress without disturbing the run

```bash
bash phaseB/scripts/snapshot.sh
```

Runs all three analyzers on whatever data has landed so far and updates the plots in `phaseB/results/analysis/`. Doesn't touch the running generation or judge processes. Safe to run any time, as often as you like.

## Pipeline (what `run_all.sh` actually does)

```
                                         single-turn (~5h)        two-turn (~10h)        two-agent (~5h)
 ┌─────────────┐                        ┌─────────────────┐      ┌────────────────┐    ┌────────────────┐
 │ 01 download │ ──► sycophancy_*.jsonl │ 02 (B1)         │  ──► │ 02b (B2)       │ ─► │ 02c (B3)       │
 │   data      │                        │ Llama 70B fwd + │      │ Llama 70B 2-turn│   │ manager↔engineer│
 └─────────────┘                        │ generate        │      │                 │   │                │
                                        └────────┬────────┘      └────────┬────────┘    └────────┬───────┘
                                                 ▼                        ▼                     ▼
                                          trials.jsonl       two_turn_trials.jsonl   two_agent_trials.jsonl
                                                 │                        │                     │
                                                 ▼                        ▼                     ▼
                                          ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
                                          │ 03 Qwen 72B      │ │ 03b judge t1+t2  │ │ 03c judge each   │
                                          │ judge sycophancy │ │ + capitulation   │ │ engineer round   │
                                          └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
                                                   ▼                    ▼                   ▼
                                          trials_judged.jsonl   two_turn_judged.jsonl   two_agent_judged.jsonl
                                                   │                    │                   │
                                                   ▼                    ▼                   ▼
                                          ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
                                          │ 04 ROC, AUC,     │ │ 04b drift→cap.   │ │ 04c trajectory   │
                                          │ DeLong test      │ │ ROC, LR test     │ │ per round        │
                                          └──────────────────┘ └──────────────────┘ └──────────────────┘
                                                   │                    │                   │
                                                   ▼                    ▼                   ▼
                                            analysis/phaseB_*.png  analysis/two_turn_*.png  analysis/two_agent_*.png
```

## Hypotheses

**B1 (in scope).** V_internal[desperate] / V_internal[-calm] read at the user-token position predicts whether the model will be sycophantic in its first response, at AUC > 0.65 with bootstrap CI not crossing 0.5. V_text composite AUC is significantly lower (DeLong p < 0.05). Replicates the Phase 4 H5 finding on a public, externally-labelled benchmark.

**B2 (in scope).** Define `Δ = V_internal_t2[desperate] − V_internal_t1[desperate]`. Capitulation under pushback is predicted by Δ at AUC > 0.6, *and* the likelihood-ratio test shows Δ adds value beyond V_int_t2 alone (p < 0.05). New section: *"Probe drift across pressure predicts capitulation."*

**B3 (out of scope).** Engineer V_internal[desperate] increases monotonically across rounds in dialogues that end in shortcut, but stays flat in dialogues that hold firm. Pre-shortcut round AUC > 0.65. Mental note: do NOT use this in the workshop paper without random-context-length and turn-count controls — see header of `02c_two_agent_dialogue.py`.

## What "success" looks like for the workshop

Add **two** new sections to the existing paper draft:

> **§5 External validation: probes generalise to a public sycophancy benchmark.**  
> *(uses B1 outputs)*  
> 1500 prompts from Anthropic's `sycophancy_feedback`. V_internal AUC = X.XX [boot 95% CI], V_text AUC = Y.YY, DeLong p = Z. Within-category AUC stable across splits. Threshold-based monitor at TPR=0.95 yields FPR=W%.

> **§6 Probe drift predicts capitulation under pushback.**  
> *(uses B2 outputs)*  
> 1500 are-you-sure dialogues. Δ V_internal[desperate] (turn 2 − turn 1) predicts capitulation at AUC = X.XX. LR test confirms drift adds explanatory power beyond turn-2 probe alone (chi² = ..., p = ...). Per-category breakdown stable.

Together these convert the paper from "interpretability story on synthetic prompts" → "interpretability + an externally-validated runtime monitor whose drift signal works under multi-turn pressure."

## File layout

```
phaseB/
├── README.md                              this file
├── run_all.sh                             nohup-friendly master orchestrator
├── data/                                  populated by 01
│   ├── sycophancy_feedback.jsonl          (B1 input)
│   └── sycophancy_are_you_sure.jsonl      (B2 input)
├── results/                               created at runtime
│   ├── trials.jsonl                       (B1 raw)
│   ├── trials_judged.jsonl                (B1 + judge)
│   ├── two_turn_trials.jsonl              (B2 raw)
│   ├── two_turn_trials_judged.jsonl       (B2 + judge)
│   ├── two_agent_trials.jsonl             (B3 raw, OUT-OF-SCOPE)
│   ├── two_agent_trials_judged.jsonl      (B3 + judge, OUT-OF-SCOPE)
│   └── analysis/
│       ├── phaseB_report.json             B1 numbers
│       ├── phaseB_roc.png
│       ├── phaseB_v_internal_boxplots.png
│       ├── phaseB_within_category.png
│       ├── two_turn_report.json           B2 numbers
│       ├── two_turn_roc.png
│       ├── two_turn_drift_boxplot.png
│       ├── two_agent_report.json          B3 numbers (out-of-scope)
│       └── two_agent_trajectory.png
└── scripts/
    ├── 01_download_data.py                HF → JSONL
    ├── 02_run_with_probes.py              B1 generation
    ├── 02b_two_turn_pushback.py           B2 generation
    ├── 02c_two_agent_dialogue.py          B3 generation (out-of-scope header)
    ├── 03_judge_responses.py              B1 judge
    ├── 03b_judge_two_turn.py              B2 judge (per-turn + capitulation)
    ├── 03c_judge_two_agent.py             B3 judge (per-engineer-round)
    ├── 04_analyze.py                      B1 analysis
    ├── 04b_analyze_two_turn.py            B2 analysis
    ├── 04c_analyze_two_agent.py           B3 analysis
    └── snapshot.sh                        run all 3 analyzers on partial data
```

## Hardware / time budget

- 8x V100 32GB, fp16, pipeline-parallel via `device_map="auto"`.
- B1: ~1500 items × ~10s = ~4-6h
- B2: ~1500 items × ~22s (2 forward+gen each) = ~9-12h
- B3: ~250 dialogues × ~80s = ~5-7h (4 rounds × ~2 model calls per round)
- Judge B1: ~1h. Judge B2: ~2h. Judge B3: ~2h.
- Total: ~24-36h unattended.

## Caveats up front

1. **Phase 4's random-direction null didn't show emotion-specific specificity** (Fisher p=0.83). When B1/B2 work, "V_internal[desperate]" should be reported as a residual-stream regularity that *correlates with* sycophancy, not as a literal "emotion vector causally driving capitulation." The mental note for B3 expands on this — don't oversell drift as an emotion phenomenon.
2. **Single-pass judge.** Three judge passes give us inter-pass agreement, but a second-judge replication (e.g. Claude as confirmatory judge) would be a future extension.
3. **B2 pushback template is fixed.** `--pushback-mode rotate` is available if you want to robustness-check across phrasings.
4. **B3 has confound risk.** Long contexts shift V_internal even without emotion change. The 02c header lists the controls a future B3 paper would need.

## What to do if it breaks at 3am

Each step's log is in `logs/phaseB_*.log`. The job runs in a single bash process so nohup keeps everything alive. If the process is killed:

```bash
# diagnose
tail -200 logs/phaseB_master.log
tail -200 logs/phaseB_02.log    # or phaseB_02b/02c/03/03b/03c

# fix the underlying issue, then re-launch — it resumes from the same place
nohup bash phaseB/run_all.sh > logs/phaseB_master.log 2>&1 &
```

If a single step is wedged (e.g. judge OOM), kill that one step, drop `--max-new-tokens` or `--batch-size` in `run_all.sh`, then relaunch. Never rerun a generation step with `rm -f` — the resumability checkpoint is the existing JSONL.
