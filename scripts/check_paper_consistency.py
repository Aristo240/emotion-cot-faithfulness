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
# section 4.2 quotes a raw AUC and a rank; both must come from the script
chk("bored raw AUC", 0.985, J["directions"]["raw_auc"]["bored"])
chk("desperate raw AUC", 0.832, J["directions"]["raw_auc"]["desperate"])
chk("desperate raw rank", 8, J["directions"]["desperate_raw_auc_rank"], tol=0)
checks += 1
if J["directions"]["raw_auc_best"]["direction"] != "bored":
    fails.append("paper names `bored` as the best raw-AUC direction; script says "
                 + J["directions"]["raw_auc_best"]["direction"])

# Table 2 inferential columns, from the conditional-null run
for name, c2, beta, pc, pf in [
    ("bored", 38.7, 2.48, 0.0005, 0.0005), ("lonely", 29.6, 1.91, 0.0005, 0.0005),
    ("nostalgic", 27.5, 1.84, 0.0005, 0.0005), ("melancholy", 24.8, 1.63, 0.0005, 0.0005),
    ("gloomy", 21.2, 1.46, 0.0005, 0.0005), ("compassionate", 19.7, -1.50, 0.0010, 0.0005),
    ("sad", 18.6, 1.29, 0.0010, 0.0010), ("desperate", 3.3, 0.69, 0.4693, 0.4273),
]:
    chk(f"{name} chi2", c2, C["chi2"][name], tol=5e-2)
    chk(f"{name} beta", beta, J["nested_lr"]["penalised"][name]["beta"], tol=6e-3)
    chk(f"{name} conditional p", pc, C["p_conditional"][name], tol=5e-5)
    chk(f"{name} free p", pf, C["p_free"][name], tol=5e-5)
chk("survivors conditional", 17, C["n_survivors_conditional"], tol=0)
chk("B", 2000, C["B"], tol=0)
chk("p resolution floor", 0.0005, C["p_resolution_floor"], tol=1e-6)
chk("conditional stricter count", 42, C["n_conditional_ge_free"], tol=0)
# R1.1: the generating model must be the unpenalised MLE, not the shrunk fit
gm = C["generating_model"]
chk("generating slope (MLE)", 1.559, gm["slope_mle"], tol=5e-3)
chk("ridge slope (not used)", 1.388, gm["slope_ridge"], tol=5e-3)
chk("shrinkage avoided", 0.11, gm["shrinkage"], tol=5e-3)
# R2.1: direction-only (magnitude-removed) readout
do = C["direction_only"]
chk("direction-only survivors", 15, do["n_survivors"], tol=0)
chk("survivors kept", 8, do["n_original_survivors_kept"], tol=0)
chk("overlap size", 8, len(do["overlap_with_scalar"]), tol=0)
chk("scalar-only size", 9, len(do["scalar_only"]), tol=0)
chk("direction-only-new size", 7, len(do["direction_only_new"]), tol=0)
chk("desperate chi2 direction-only", 0.25, do["chi2"]["desperate"], tol=5e-3)
chk("desperate p direction-only", 1.0000, do["p_conditional"]["desperate"], tol=5e-4)
chk("rho desperate-length before", 0.606, do["desperate_rho_length_before"])
chk("rho desperate-length after", 0.657, do["desperate_rho_length_after"])
chk("max chi2 direction-only", 28.02, max(do["chi2"].values()), tol=5e-2)
# the robust core named in the paper must actually be the overlap set
_core = {"bored", "brooding", "gloomy", "grateful", "lonely", "melancholy", "nostalgic", "sad"}
checks += 1
if set(do["overlap_with_scalar"]) != _core:
    fails.append(f"paper names core {sorted(_core)}; script overlap is {do['overlap_with_scalar']}")
# the paper names these as added by each readout
checks += 1
for e in ("proud", "hopeful", "jubilant"):
    if e not in do["scalar_only"]:
        fails.append(f"paper says {e} is scalar-readout-only; script disagrees")
checks += 1
for e in ("hostile", "contemptuous", "enraged", "defiant"):
    if e not in do["direction_only_new"]:
        fails.append(f"paper says {e} is direction-only-new; script disagrees")
chk("frozen fraction", 0.60, C["frozen_fraction"], tol=5e-3)
chk("length AUC (in-text)", 0.888, J["association"]["auc_length"])
st = C["survivor_structure"]
chk("survivor mean |r|", 0.75, st["mean_abs_r"], tol=5e-3)
chk("survivor PC1", 0.786, st["pc1_var_explained"], tol=5e-4)
chk("survivor participation ratio", 1.59, st["participation_ratio"], tol=5e-3)
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
# conditional null must not be looser than the free null overall
checks += 1
if C["n_conditional_ge_free"] <= C["n_directions"] // 2:
    fails.append("conditional null is looser than the free null -- design is wrong")

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
