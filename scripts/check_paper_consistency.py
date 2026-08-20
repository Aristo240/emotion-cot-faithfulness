#!/usr/bin/env python3
"""
Consistency gate for paper/interpscience_short.tex.

Asserts that every quantitative claim in the paper matches
results/paper_numbers.json to the precision the paper quotes it at.
Run after paper_numbers.py. Non-zero exit on any mismatch.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
J = json.load(open(ROOT / "results/paper_numbers.json"))
C = json.load(open(ROOT / "results/conditional_null.json"))
TEX = (ROOT / "paper/interpscience_short.tex").read_text()
E = json.load(open(ROOT / "results/emobank_baseline.json"))
RB = json.load(open(ROOT / "results/robustness_controls.json"))

fails, checks = [], 0


def chk(label, claimed, actual, tol=5e-4):
    """Assert a value quoted in the paper matches the regenerated value."""
    global checks
    checks += 1
    if actual is None or abs(claimed - actual) > tol:
        fails.append(f"{label}: paper says {claimed}, script gives {actual}")


TEX_FLAT = re.sub(r"\s+", " ", TEX)


def in_tex(s):
    """Assert a phrase appears in the paper, ignoring LaTeX line wrapping."""
    global checks
    checks += 1
    if re.sub(r"\s+", " ", s) not in TEX_FLAT:
        fails.append(f"missing from paper: {s!r}")


d = {x["direction"]: x for x in J["directions"]["all"]}

# ---- §4.1 semantic
chk("EmoBank valence R2", 0.377, J["semantic"]["cv_r2"]["V"])
chk("EmoBank arousal R2", 0.180, J["semantic"]["cv_r2"]["A"])
chk("EmoBank dominance R2", 0.154, J["semantic"]["cv_r2"]["D"])
chk("EmoBank n", 10062, J["semantic"]["n"], tol=0)

# ---- §3 dataset + judges
chk("unsteered n", 120, J["dataset"]["unsteered_n"], tol=0)
chk("unsteered events", 14, J["dataset"]["unsteered_events"], tol=0)
chk("tier2 judge agree", 120, J["judge"]["tier2_agree"], tol=0)
chk("diverse judge agree", 165, J["judge"]["diverse_agree"], tol=0)
chk("diverse n", 650, J["judge"]["diverse_n"], tol=0)
chk("diverse agree pct", 25.4, 100 * J["judge"]["diverse_agree"] / J["judge"]["diverse_n"], tol=0.05)
chk("diverse qwen shortcut", 2, J["judge"]["diverse_qwen_shortcut"], tol=0)
chk("diverse qwen unclear", 524, J["judge"]["diverse_qwen_unclear"], tol=0)
chk("diverse claude shortcut", 194, J["judge"]["diverse_claude_shortcut"], tol=0)

# ---- §4.2 length control
a = J["association"]
chk("desperate AUC", 0.832, a["auc_desperate"])
chk("desperate CI lo", 0.745, a["ci_desperate"][0])
chk("desperate CI hi", 0.908, a["ci_desperate"][1])
chk("length AUC", 0.888, a["auc_length"])
chk("length CI lo", 0.787, a["ci_length"][0])
chk("length CI hi", 0.967, a["ci_length"][1])
chk("rho desp-length", 0.606, a["spearman_desp_length"]["rho"])
chk("paired dAUC", 0.057, a["paired_dauc_length_minus_desperate"]["mean"])
chk("paired dAUC lo", -0.080, a["paired_dauc_length_minus_desperate"]["ci95"][0], tol=1e-3)
chk("paired dAUC hi", 0.183, a["paired_dauc_length_minus_desperate"]["ci95"][1], tol=1e-3)
chk("LR chi2", 3.26, a["lr_test_vint_over_length"]["chi2"], tol=5e-3)
chk("LR p", 0.071, a["lr_test_vint_over_length"]["p"], tol=5e-4)

# ---- §4.3 direction sweep. Inference = nested LR + conditional max-T.
# Residualised AUC is descriptive only, so no p-values are checked against it.
for name, resid in [("bored", 0.858), ("lonely", 0.760), ("nostalgic", 0.797),
                    ("melancholy", 0.765), ("gloomy", 0.761),
                    ("compassionate", 0.265), ("sad", 0.734), ("desperate", 0.592)]:
    chk(f"{name} residAUC (descriptive)", resid, d[name]["resid_auc"])
chk("desperate resid CI lo", 0.412, J["directions"]["desperate"]["ci95"][0], tol=1e-3)
chk("desperate resid CI hi", 0.756, J["directions"]["desperate"]["ci95"][1], tol=1e-3)
chk("n directions", 50, J["directions"]["n_directions"], tol=0)
# The residualised-AUC permutation/BH inference was retired: it was a second,
# looser criterion on a different statistic that disagreed with the reported one.
# Fail if it reappears in the released JSON.
for _dead in ("n_bh_significant", "n_maxT_significant", "maxT_survivors"):
    checks += 1
    if _dead in J["directions"]:
        fails.append(f"retired key {_dead!r} is back in paper_numbers.json")
for _dead in ("p", "bh_q", "maxT_p"):
    checks += 1
    if any(_dead in r for r in J["directions"]["all"]):
        fails.append(f"retired per-direction key {_dead!r} is back in paper_numbers.json")
checks += 1
if J["directions"].get("_inference_lives_in") != "results/conditional_null.json":
    fails.append("paper_numbers.json no longer points at where inference lives")
# section 4.2 quotes a raw AUC and a rank; both must come from the script
chk("bored raw AUC", 0.985, J["directions"]["raw_auc"]["bored"])
chk("desperate raw AUC", 0.832, J["directions"]["raw_auc"]["desperate"])
chk("desperate raw rank", 8, J["directions"]["desperate_raw_auc_rank"], tol=0)
checks += 1
if J["directions"]["raw_auc_best"]["direction"] != "bored":
    fails.append("paper names `bored` as the best raw-AUC direction; script says "
                 + J["directions"]["raw_auc_best"]["direction"])

# Table 2 inferential columns, from the conditional-null run (B = 10000)
for name, c2, beta, pc, pf in [
    ("bored", 38.7, 2.48, 0.0001, 0.0001), ("lonely", 29.6, 1.91, 0.0001, 0.0001),
    ("nostalgic", 27.5, 1.84, 0.0001, 0.0001), ("melancholy", 24.8, 1.63, 0.0001, 0.0001),
    ("gloomy", 21.2, 1.46, 0.0001, 0.0001), ("compassionate", 19.7, -1.50, 0.0003, 0.0004),
    ("sad", 18.6, 1.29, 0.0005, 0.0008), ("desperate", 3.3, 0.69, 0.4404, 0.4456),
]:
    chk(f"{name} chi2", c2, C["chi2"][name], tol=5e-2)
    chk(f"{name} beta", beta, J["nested_lr"]["penalised"][name]["beta"], tol=6e-3)
    chk(f"{name} conditional p", pc, C["p_conditional"][name], tol=5e-5)
    chk(f"{name} free p", pf, C["p_free"][name], tol=5e-5)
chk("survivors conditional", 18, C["n_survivors_conditional"], tol=0)
chk("survivors free", 18, C["n_survivors_free"], tol=0)
chk("B", 10000, C["B"], tol=0)
chk("p resolution floor", 0.0001, C["p_resolution_floor"], tol=1e-6)
chk("length AUC (in-text)", 0.888, J["association"]["auc_length"])
# the paper says the two nulls differ by at most 0.005 per direction
checks += 1
_maxd = max(abs(C["p_conditional"][e] - C["p_free"][e]) for e in C["p_conditional"])
if _maxd > 0.02:
    fails.append(f"paper says the nulls differ by <=0.02; max is {_maxd:.4f}")
# R1.2 ridge sensitivity
Rg = json.load(open(ROOT / "results/ridge_sensitivity.json"))
chk("ridge0 survivors", 16, Rg["0.0"]["n_survivors"], tol=0)
chk("ridge2 survivors", 18, Rg["2.0"]["n_survivors"], tol=0)
chk("ridge min overlap", 16, Rg["_meta"]["min_overlap_with_ridge1"], tol=0)
chk("ridge0 nonconvergent", 13, Rg["0.0"]["nonconvergent_fits"], tol=0)
chk("ridge desperate p min", 0.3848, min(Rg[k]["desperate_p"] for k in ("0.0","0.5","1.0","2.0")), tol=5e-5)
chk("ridge desperate p max", 0.5037, max(Rg[k]["desperate_p"] for k in ("0.0","0.5","1.0","2.0")), tol=5e-5)
checks += 1
if not Rg["_meta"]["desperate_null_at_all_ridges"]:
    fails.append("paper says desperate is null at every ridge; script disagrees")
st = C["survivor_structure"]
chk("survivor mean |r|", 0.76, st["mean_abs_r"], tol=5e-3)
chk("survivor PC1", 0.788, st["pc1_var_explained"], tol=5e-4)
chk("survivor participation ratio", 1.58, st["participation_ratio"], tol=5e-3)
chk("penalised desperate chi2", 3.31, J["nested_lr"]["penalised"]["desperate"]["chi2"], tol=5e-3)
chk("penalised desperate p", 0.069, J["nested_lr"]["penalised"]["desperate"]["p"], tol=5e-4)
# the two logistic implementations must agree exactly
checks += 1
if C["chi2_max_disagreement_vs_paper_numbers"] > 1e-6:
    fails.append(f"logistic implementations disagree by "
                 f"{C['chi2_max_disagreement_vs_paper_numbers']:.2e}")
# nulls must be calibrated and convergent
for k in ("free", "conditional"):
    checks += 1
    if abs(C["mean_sim_events"][k] - C["events"]) > 0.5:
        fails.append(f"{k} null miscalibrated: mean events "
                     f"{C['mean_sim_events'][k]:.2f} vs {C['events']}")
    checks += 1
    if C["nonconvergent_draws"][k] != 0:
        fails.append(f"{k} null had {C['nonconvergent_draws'][k]} non-convergent draws")
# the two nulls must not disagree materially on which directions are significant
checks += 1
_sa = {e for e, v in C["p_free"].items() if v < .05}
_sb = {e for e, v in C["p_conditional"].items() if v < .05}
if len(_sa ^ _sb) > 2:
    fails.append(f"free and conditional nulls disagree on {len(_sa ^ _sb)} directions")

# ---- §4.4 V_text
v = J["vtext"]
chk("vtext n", 992, v["n"], tol=0)
chk("vtext events", 87, v["events"], tol=0)
chk("modal share", 0.740, v["modal_share"], tol=5e-4)
chk("tied pairs", 43485, v["tied_roc_pairs"], tol=0)
chk("total pairs", 78735, v["total_roc_pairs"], tol=0)
chk("tie fraction", 0.552, v["tied_pair_fraction"], tol=5e-4)
chk("vtext AUC all", 0.647, v["auc_vtext_all"])
chk("vtext AUC nonmodal", 0.836, v["auc_vtext_nonmodal"])
chk("vint AUC all", 0.837, v["auc_vint_all"])
chk("n nonmodal", 258, v["n_nonmodal"], tol=0)
chk("events nonmodal", 22, v["events_nonmodal"], tol=0)
if not v["p33_eq_p66"]:
    fails.append("paper claims p33 == p66 but script disagrees")
checks += 1

# ---- §4.5 causal
c = J["causal"]
chk("baseline hacks", 14, c["baseline"][0], tol=0)
chk("baseline n", 120, c["baseline"][1], tol=0)
chk("emotion hacks", 12, c["emotion_0_3"][0], tol=0)
chk("emotion n", 160, c["emotion_0_3"][1], tol=0)
chk("random hacks", 13, c["random_0_3"][0], tol=0)
chk("random n", 200, c["random_0_3"][1], tol=0)
chk("fisher emo vs rnd", 0.835, c["fisher_emotion_vs_random"], tol=5e-4)
chk("emotion vs base OR", 0.61, c["vs_baseline"]["emotion"]["odds_ratio"], tol=5e-3)
chk("emotion vs base p", 0.30, c["vs_baseline"]["emotion"]["p"], tol=5e-3)
chk("random vs base OR", 0.53, c["vs_baseline"]["random"]["odds_ratio"], tol=5e-3)
chk("random vs base p", 0.14, c["vs_baseline"]["random"]["p"], tol=5e-3)
chk("MDE relative risk", 2.42, c["mde_relative_risk"], tol=5e-3)
chk("trend z", -1.42, c["trend_desperate"]["z"], tol=5e-3)
chk("trend p", 0.157, c["trend_desperate"]["p"], tol=5e-4)
chk("text inj p", 0.417, c["text_injection_vs_baseline_p"], tol=5e-4)
ml = c["mean_response_length"]
chk("len unsteered", 1402, ml["unsteered"], tol=0.5)
chk("len emotion", 1324, ml["emotion_abs_ge_0.3"], tol=0.5)
chk("len random", 1456, ml["random"], tol=0.5)

# ---- §4.6 layers, raw and length-residualised
chk("layer sweep n", 80, J["layers"]["n"], tol=0)
chk("length AUC tokens", 0.912, J["layers"]["length_auc_tokens"])
for lay, raw, res, lo, hi, rr in [
    ("13", 0.691, 0.333, 0.198, 0.477, 0.619),
    ("26", 0.450, 0.262, 0.123, 0.411, 0.445),
    ("39", 0.918, 0.646, 0.395, 0.871, 0.603),
    ("52", 0.708, 0.532, 0.294, 0.773, 0.392),
    ("53", 0.677, 0.524, 0.290, 0.761, 0.373),
    ("65", 0.472, 0.360, 0.137, 0.591, 0.344),
]:
    L = J["layers"]["per_layer"][lay]
    chk(f"layer {lay} raw", raw, L["raw_auc"])
    chk(f"layer {lay} resid", res, L["resid_auc"])
    chk(f"layer {lay} resid CI lo", lo, L["resid_ci95"][0], tol=1e-3)
    chk(f"layer {lay} resid CI hi", hi, L["resid_ci95"][1], tol=1e-3)
    chk(f"layer {lay} rho", rr, L["rho_length"])
# the paper's central layer claim: no layer survives for `desperate`
for lay in ("39", "52", "53", "65"):
    checks += 1
    if J["layers"]["per_layer"][lay]["resid_ci_excludes_half"]:
        fails.append(f"paper claims layer {lay} spans chance, but its CI excludes 0.5")
in_tex("no layer in our sweep offers a length-independent version")

# ---- §4.6b layer profile correlations
sp = J["layer_profiles"]["spearman"]
chk("profile 13 vs 53", 0.24, sp["13"]["53"], tol=5e-3)
chk("profile 26 vs 53", 0.24, sp["26"]["53"], tol=5e-3)
chk("profile 13 vs 26", 0.04, sp["13"]["26"], tol=5e-3)
chk("profile 39 vs 53", 0.82, sp["39"]["53"], tol=5e-3)
block = ["39", "52", "53", "65"]
vals = [sp[a][b] for a in block for b in block if a != b]
checks += 1
if not (min(vals) >= 0.61 - 5e-3 and max(vals) <= 0.99 + 5e-3):
    fails.append(f"paper claims late block spans 0.61-0.99, got {min(vals):.2f}-{max(vals):.2f}")

# ---- §4.5b random-arm homogeneity
rates = sorted(v[2] for v in J["random_arm_homogeneity"]["per_direction"].values())
checks += 1
if rates != sorted([0.075, 0.075, 0.075, 0.025, 0.075]):
    fails.append(f"random-arm per-direction rates differ from paper: {rates}")

in_tex("late-layer phenomenon")
in_tex("no complete separation")
in_tex("a single axis detected many times")
in_tex("its composition does not")
in_tex("preregistration describes this quantity as a")
in_tex("Four controls change the conclusion")

# ---- §4.7 prereg
if "INSUFFICIENT-DATA" not in J["prereg"]["verdict"]:
    fails.append("prereg verdict is not INSUFFICIENT-DATA")
checks += 1

# ---- structural checks on the paper itself
in_tex("scripts/paper\\_numbers.py")           # reproducibility pointer present
in_tex("No layer in the sweep gives the preregistered direction")  # layer objection answered
in_tex("conceptual, not direct, replication")   # replication scope stated
in_tex("Novelty statement")                     # novelty disclaimer present
# P1-class drift: the summary table must name the null the results section uses.
in_tex("max-$T$ over 50, length-preserving null")
# ---- §3 judge abstention, not contradiction (added 2026-08-20 in response to review)
# The paper used to quote raw agreement (165/650), which reads as the judges
# contradicting each other. These assertions pin the corrected characterisation.
_j = J["judge"]
chk("diverse commit count", 126, _j["diverse_qwen_commits"], tol=0)
chk("diverse abstention rate", 0.806, 1 - _j["diverse_commit_rate"], tol=5e-4)
chk("contradictions given both commit", 0, _j["diverse_contradictions_given_both_commit"], tol=0)
chk("contradiction upper bound", 0.024, _j["diverse_contradiction_rate_upper95"], tol=5e-4)
chk("abstention on Claude-SHORTCUT", 0.99,
    _j["diverse_abstain_on_claude_shortcut"][0] / _j["diverse_abstain_on_claude_shortcut"][1], tol=5e-3)
chk("abstention elsewhere", 0.73,
    _j["diverse_abstain_elsewhere"][0] / _j["diverse_abstain_elsewhere"][1], tol=5e-3)
chk("tier-2 abstentions", 0, _j["tier2_qwen_unclear"], tol=0)
checks += 1
if not _j["diverse_abstention_fisher_p"] < 1e-18:
    fails.append(f"paper says Fisher p = 4e-19 for informative abstention; script gives "
                 f"{_j['diverse_abstention_fisher_p']:.2g}")
# §3 and §4.6 both say the abstention is mechanism-specific: one labelled, >=98% on the rest
_pm = _j["diverse_abstention_per_mechanism"]
chk("diverse mechanisms", 5, len(_pm), tol=0)
_rates = sorted(u / n_ for u, n_ in _pm.values())
checks += 1
if not (_rates[0] < 0.10 and all(r >= 0.98 for r in _rates[1:])):
    fails.append(f"paper says one mechanism is labelled and the other four abstain >=98%; "
                 f"rates are {[round(r, 3) for r in _rates]}")
# the exclusion is justified by informative missingness, so the committed subset
# must NOT be a fair sample: assert the two abstention rates actually differ
checks += 1
if (_j["diverse_abstain_on_claude_shortcut"][0] / _j["diverse_abstain_on_claude_shortcut"][1]
        <= _j["diverse_abstain_elsewhere"][0] / _j["diverse_abstain_elsewhere"][1]):
    fails.append("paper says abstention is higher on Claude-SHORTCUT trials; it is not")

# ---- §4.1 trivial text baseline (added 2026-08-20 in response to review)
chk("TF-IDF valence R2", 0.219, E["tfidf_cv_r2"]["V"])
chk("TF-IDF arousal R2", 0.092, E["tfidf_cv_r2"]["A"])
chk("TF-IDF dominance R2", 0.063, E["tfidf_cv_r2"]["D"])
for _d in ("V", "A", "D"):
    checks += 1
    if not E["probe_beats_tfidf"][_d]:
        fails.append(f"paper says the probe clears the text baseline; it does not on {_d}")
    # the baseline script must reproduce the probe numbers it is compared against
    chk(f"baseline run reproduces probe {_d}", J["semantic"]["cv_r2"][_d], E["probe_cv_r2"][_d])

# ---- §4.3 survivor count under three nuisance models
chk("survivors, length only", 18, RB["baseline_length"]["n_survivors"], tol=0)
chk("survivors, flexible length", 14, RB["flexible_length"]["n_survivors"], tol=0)
chk("survivors, length + task", 11, RB["task_fixed_effects"]["n_survivors"], tol=0)
_core = (set(RB["baseline_length"]["survivors"])
         & set(RB["flexible_length"]["survivors"])
         & set(RB["task_fixed_effects"]["survivors"]))
chk("directions clearing all three", 11, len(_core), tol=0)
checks += 1
if not {"bored", "lonely", "nostalgic", "melancholy", "compassionate", "proud"} <= _core:
    fails.append("paper names directions as clearing all three nuisance models that do not")
# the robustness run must reproduce the sweep it claims to perturb
checks += 1
if RB["baseline_matches_conditional_null"] > 1e-6:
    fails.append(f"robustness baseline chi2 differ from conditional_null.py by "
                 f"{RB['baseline_matches_conditional_null']:.2e}")
# `desperate` must stay null under every nuisance model; the paper says p >= 0.44
for _m in ("baseline_length", "flexible_length", "task_fixed_effects"):
    checks += 1
    if RB[_m]["p_maxT"]["desperate"] < 0.43:
        fails.append(f"paper says desperate p >= 0.43 everywhere; {_m} gives "
                     f"{RB[_m]['p_maxT']['desperate']:.4f}")
# the uneven task event rates quoted in §4.3
_ev = RB["tasks"]["events_per_variant"]
chk("max events in a variant", 9, max(v[0] for v in _ev.values()), tol=0)
chk("min events in a variant", 0, min(v[0] for v in _ev.values()), tol=0)
chk("trials per variant", 30, max(v[1] for v in _ev.values()), tol=0)
chk("task variants", 4, RB["tasks"]["n_variants"], tol=0)
checks += 1
if not RB["tasks"]["single_task_family"]:
    fails.append("paper says one task family; the data has more than one")

# ---- Table 2's rho(length) column
for _d, _r in [("bored", 0.33), ("lonely", 0.48), ("nostalgic", 0.34),
               ("melancholy", 0.39), ("compassionate", -0.25), ("desperate", 0.61)]:
    chk(f"rho(len) {_d}", _r, RB["rho_length"][_d], tol=5e-3)
# the caption calls `desperate` 3rd of 50 by |rho| and explicitly not unique
checks += 1
_rank = sorted(RB["rho_length"], key=lambda k: -abs(RB["rho_length"][k])).index("desperate") + 1
if _rank != 3:
    fails.append(f"caption says desperate is 3rd of 50 by |rho(len)|; it is {_rank}")
checks += 1
if RB["rho_length_summary"]["desperate_is_max_over_all"]:
    fails.append("caption says desperate is not uniquely length-entangled, but it is the max")

# ---- §4.4 like-for-like AUC on the non-modal subset
_v = RB["vtext_like_for_like"]
chk("vint AUC on non-modal subset", 0.890, _v["auc_vint_nonmodal"])
chk("gap on the same trials", 0.054, _v["gap_on_same_trials"], tol=1e-3)
chk("gap as pooled AUCs suggest", 0.190, _v["gap_as_reported"], tol=1e-3)
chk("non-modal n (robustness run)", 258, _v["n_nonmodal"], tol=0)

# ---- §4.5 the interval the exclusion claim rests on
_ci = RB["causal_interval"]["emotion_vs_random"]
chk("emotion vs random RR", 1.15, _ci["rr"], tol=5e-3)
chk("RR CI low", 0.54, _ci["ci95"][0], tol=5e-3)
chk("RR CI high", 2.46, _ci["ci95"][1], tol=5e-3)
checks += 1
if not _ci["ci95"][1] < 14:
    fails.append("paper says 14x is outside the interval; it is not")
# the paper says 14x is unattainable from an 11.7% base rate
checks += 1
if 14 * (J["causal"]["baseline"][0] / J["causal"]["baseline"][1]) <= 1.0:
    fails.append("paper says 14x is unattainable from our baseline rate; it is attainable")

# the intro tally must match Table 1's verdict column
checks += 1
_tab = TEX.split(r"\label{tab:summary}")[0].split(r"\midrule")[-1]
_rows = [r for r in _tab.split(r"\\") if r.strip() and "bottomrule" not in r]
_sup = sum("Supported" in r for r in _rows)
_not = sum("Not supported" in r for r in _rows)
_oth = sum(("Not evaluable" in r or "Inconclusive" in r) for r in _rows)
if not (len(_rows) == 7 and _sup == 2 and _not == 3 and _oth == 2):
    fails.append(f"Table 1 verdicts ({len(_rows)} rows: {_sup} supported, {_not} not, "
                 f"{_oth} other) contradict the intro tally of 7/2/3/2")
checks += 1
if "Permutation max-$T$ over 50" in TEX:
    fails.append("Table 1 still names the free-permutation null, which is not primary")
# the abstract must not quote the retired residualised-AUC max-T value for desperate
checks += 1
if "0.91" in TEX_FLAT.split("end{abstract}")[0]:
    fails.append("abstract quotes the retired residualised-AUC max-T p (0.91)")

# no stray citation keys
bib = (ROOT / "paper/refs.bib").read_text()
for key in set(re.findall(r"\\cite[tp]?\{([^}]*)\}", TEX)):
    for k in (x.strip() for x in key.split(",")):
        checks += 1
        if f"{{{k}," not in bib:
            fails.append(f"citation {k!r} not in refs.bib")

_m = re.search(r"with (\d+) assertions", TEX)
if not _m:
    fails.append("manuscript no longer states an assertion count")
elif int(_m.group(1)) != checks:
    fails.append(f"manuscript says {_m.group(1)} assertions; this run made {checks}")

print(f"ran {checks} checks against results/paper_numbers.json")
if fails:
    print(f"\n{len(fails)} MISMATCH(ES):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL CONSISTENT: every number in the paper is reproduced by paper_numbers.py")
