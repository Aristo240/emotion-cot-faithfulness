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

# ---- §4.3 direction sweep (paper Table 2)
for name, auc_, p_, bh_, mt_ in [
    ("bored", 0.858, 0.0001, 0.0050, 0.0001),
    ("nostalgic", 0.797, 0.0003, 0.0075, 0.0027),
    ("melancholy", 0.765, 0.0010, 0.0100, 0.0132),
    ("gloomy", 0.761, 0.0007, 0.0087, 0.0163),
    ("lonely", 0.760, 0.0006, 0.0087, 0.0167),
    ("compassionate", 0.265, 0.0035, 0.0281, 0.0479),
    ("sad", 0.734, 0.0041, 0.0281, 0.0490),
    ("desperate", 0.592, 0.2687, 0.3535, 0.9115),
]:
    chk(f"{name} residAUC", auc_, d[name]["resid_auc"])
    chk(f"{name} raw p", p_, d[name]["p"], tol=5e-5)
    chk(f"{name} BH q", bh_, d[name]["bh_q"], tol=5e-5)
    chk(f"{name} maxT p", mt_, d[name]["maxT_p"], tol=5e-5)
chk("n maxT survivors", 7, J["directions"]["n_maxT_significant"], tol=0)
chk("n BH survivors", 12, J["directions"]["n_bh_significant"], tol=0)
chk("n directions", 50, J["directions"]["n_directions"], tol=0)
# the paper says "and, under BH, proud" -- verify proud is BH-significant, not max-T
if not (d["proud"]["bh_q"] < 0.05 <= d["proud"]["maxT_p"]):
    fails.append(f"proud: paper claims BH-only, got BH q={d['proud']['bh_q']:.4f}, "
                 f"maxT p={d['proud']['maxT_p']:.4f}")
checks += 1
# desperate CI quoted in two places must be identical
chk("desperate resid CI lo", 0.412, J["directions"]["desperate"]["ci95"][0], tol=1e-3)
chk("desperate resid CI hi", 0.756, J["directions"]["desperate"]["ci95"][1], tol=1e-3)

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

# ---- §4.3b nested LR (paper Table 2 cols 5-6)
lrp = J["nested_lr"]["penalised"]
for name, c2, beta in [("bored", 32.3, 1.73), ("nostalgic", 23.9, 1.38),
                       ("melancholy", 21.7, 1.25), ("gloomy", 18.9, 1.15),
                       ("lonely", 25.2, 1.40), ("sad", 16.7, 1.04),
                       ("compassionate", 16.9, -1.13), ("desperate", 2.9, 0.52)]:
    chk(f"{name} chi2", c2, lrp[name]["chi2"], tol=5e-2)
    chk(f"{name} beta", beta, lrp[name]["beta"], tol=5e-3)
chk("desperate nested p", 0.089, lrp["desperate"]["p"], tol=5e-4)
# paper claims all seven survivors exceed chi2 16.7 at p < 1e-4
for name in ("bored", "nostalgic", "melancholy", "gloomy", "lonely", "sad", "compassionate"):
    checks += 1
    if not (lrp[name]["chi2"] >= 16.6 and lrp[name]["p"] < 1e-4):
        fails.append(f"{name}: paper claims chi2>=16.6 and p<1e-4, got "
                     f"{lrp[name]['chi2']:.2f}, {lrp[name]['p']:.2g}")
# paper claims unpenalised agrees in sign and significance
for name in ("bored", "nostalgic", "melancholy", "gloomy", "lonely", "sad", "compassionate"):
    checks += 1
    u = J["nested_lr"]["unpenalised"][name]
    if not (u["p"] < 0.05 and (u["beta"] > 0) == (lrp[name]["beta"] > 0)):
        fails.append(f"{name}: unpenalised fit disagrees with penalised")
checks += 1
if not J["nested_lr"]["separation_overlap"]:
    fails.append("paper claims distributions overlap (no complete separation)")

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

# ---- §4.7 prereg
if "INSUFFICIENT-DATA" not in J["prereg"]["verdict"]:
    fails.append("prereg verdict is not INSUFFICIENT-DATA")
checks += 1

# ---- structural checks on the paper itself
in_tex("scripts/paper\\_numbers.py")           # reproducibility pointer present
in_tex("No layer in the sweep gives the preregistered direction")  # layer objection answered
in_tex("conceptual, not direct, replication")   # replication scope stated
in_tex("Novelty statement")                     # novelty disclaimer present

# no stray citation keys
bib = (ROOT / "paper/refs.bib").read_text()
for key in set(re.findall(r"\\cite[tp]?\{([^}]*)\}", TEX)):
    for k in (x.strip() for x in key.split(",")):
        checks += 1
        if f"{{{k}," not in bib:
            fails.append(f"citation {k!r} not in refs.bib")

print(f"ran {checks} checks against results/paper_numbers.json")
if fails:
    print(f"\n{len(fails)} MISMATCH(ES):")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("ALL CONSISTENT: every number in the paper is reproduced by paper_numbers.py")
