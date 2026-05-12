# Rigor Analyses (GPU-free)
Input: faithfulness_measurements.json, n=1170 measurements

## (F) V_text judge dimensions — variance screen
- **Kept** (8): v_text_judge_valence, v_text_judge_arousal, v_text_judge_dominance, v_text_judge_urgency, v_text_judge_composure, v_text_judge_frustration, v_text_judge_hedging, v_text_judge_self_interruption
- **Dropped** (0): (none)

## Task A: n=520 (steered=480, unsteered=40), events=44

## (K) DeLong paired AUC — V_internal_desperate vs V_text_composite
- V_internal_desperate AUC = 0.799
- V_text_composite AUC    = 0.593
- DeLong z=3.624, p=0.0002898

## (L) Leave-one-task-out CV on Task A
- V_internal LOGO AUC = 0.394
- V_text     LOGO AUC = 0.336

### Within-task 5-fold CV (contrast with LOGO above)
- fast_sum_v1: V_int=0.776, V_text=0.317 (n=130, events=14)
- fast_sum_v3: V_int=0.632, V_text=0.632 (n=130, events=30)

**Generalization gap:** V_internal within-task mean = 0.704 vs LOGO = 0.394. Large drop confirms RT7 probe–task confound: the probe predicts shortcut *within* a task but the learned direction does not transfer.

## (I) Decorrelated V_internal (residualize vs task_id)
- AUC(V_int_desp residualized vs task_id) = 0.674
  (raw V_int_desp AUC on same subset = 0.799)

## H5 — univariate Task A unsteered, bootstrap 95% CI
- Univariate V_int_desp AUC = 0.901 [0.785, 0.983] (n=40, events=7)

## (M) Steering effect sizes (Task A, desperate)
- desperate @ -0.50: shortcut rate = 0.075 (n=40)
- desperate @ -0.30: shortcut rate = 0.100 (n=40)
- desperate @ -0.20: shortcut rate = 0.100 (n=40)
- desperate @ +0.20: shortcut rate = 0.050 (n=40)
- desperate @ +0.30: shortcut rate = 0.025 (n=40)
- desperate @ +0.50: shortcut rate = 0.025 (n=40)
- unsteered baseline: 0.175 (n=40)
  - Cohen's h (-0.50 vs unsteered) = -0.308
  - Cohen's h (-0.30 vs unsteered) = -0.220
  - Cohen's h (-0.20 vs unsteered) = -0.220
  - Cohen's h (+0.20 vs unsteered) = -0.412
  - Cohen's h (+0.30 vs unsteered) = -0.546
  - Cohen's h (+0.50 vs unsteered) = -0.546

## (J) Multiple-comparisons correction (BH + Bonferroni)

| Test | raw p | Bonferroni | BH-FDR | significant (BH 0.05) |
|---|---|---|---|---|
| delong_internal_vs_text_taskA | 0.0002898 | 0.002318 | 0.001159 | ✓ |
| fisher_desp-0.50_vs_unsteered | 0.3109 | 1 | 0.4146 |  |
| fisher_desp-0.30_vs_unsteered | 0.5179 | 1 | 0.5179 |  |
| fisher_desp-0.20_vs_unsteered | 0.5179 | 1 | 0.5179 |  |
| fisher_desp+0.20_vs_unsteered | 0.1543 | 1 | 0.2469 |  |
| fisher_desp+0.30_vs_unsteered | 0.05676 | 0.4541 | 0.1135 |  |
| fisher_desp+0.50_vs_unsteered | 0.05676 | 0.4541 | 0.1135 |  |
| mannwhitney_vintdesp_unsteered | 0.0001833 | 0.001466 | 0.001159 | ✓ |
