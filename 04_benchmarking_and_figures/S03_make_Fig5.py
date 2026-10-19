#!/usr/bin/env python3
"""
S03_make_Fig5.py  ->  03_results/R05_Figure5_PRS_benchmark.{png,pdf,tif}
Figure 5. Out-of-sample polygenic-score benchmark of three variant panels.
  Panels: GWAS-significant (training GWAS, clumped), ML-EAT FDR <= 5 % (pruned), ML-EAT prioritization set
          (top-K by training-partition cross-validated dMSE, K = Kneedle size, clumped).
  A/B  ROC curves for discriminating tall individuals (height >= sex-specific 80th percentile)
       in the held-out 20% test set, female (A) and male (B); AUROC with DeLong 95% CI in legend.
  C/D  Mean measured height (95% CI) by PRS decile, female (C) and male (D).
Reads 01_inputs (profiles + phenotypes) and 03_results/R01, R03 (metrics + deciles from S02).
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "..", "01_inputs"); RES = os.path.join(HERE, "..", "03_results")
PANELS = ["GWAS_sig", "MLEAT_FDR", "MLEAT_trainCV"]          # main-figure panels (all selected without the test partition*)
# *MLEAT_FDR was defined on the held-out ranking; all its SNPs are GWAS-Bonferroni-significant, and it ranks last, so any
#  optimism can only work against the conclusion.  MLEAT_Knee (in-sample) is shown in Figure S3 only.
LABEL = {"GWAS_sig": "GWAS-significant", "MLEAT_FDR": "ML-EAT FDR ≤ 5%", "MLEAT_trainCV": "ML-EAT training-partition set",
         "MLEAT_Knee": "ML-EAT Kneedle set (in-sample)"}
COLOR = {"GWAS_sig": "#2a78d6", "MLEAT_FDR": "#eb6834", "MLEAT_trainCV": "#1baf7a", "MLEAT_Knee": "#898781"}   # slots 1-3 + muted
LS    = {"GWAS_sig": "-", "MLEAT_FDR": (0, (4, 2)), "MLEAT_trainCV": (0, (1.2, 1.2)), "MLEAT_Knee": (0, (4, 1.5, 1, 1.5))}
MK    = {"GWAS_sig": "o", "MLEAT_FDR": "s", "MLEAT_trainCV": "^", "MLEAT_Knee": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.2,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

M = pd.read_csv(os.path.join(RES, "R01_prs_metrics_long.csv")).set_index(["sex", "panel"])
D = pd.read_csv(os.path.join(RES, "R03_prs_decile_means.csv"))
IND = pd.read_csv(os.path.join(RES, "R04_prs_individual_scores.tsv.gz"), sep="\t")

fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))
for col, sex in enumerate(["female", "male"]):
    sub = IND[IND["sex"] == sex]; y = sub["label_top20"].to_numpy()
    # ---------- ROC ----------
    ax = axes[0, col]
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=0.8, ls=(0, (2, 2)), zorder=1)
    for p in PANELS:
        fpr, tpr, _ = roc_curve(y, sub[f"PRSz_{p}"].to_numpy())
        m = M.loc[(sex, p)]
        ax.plot(fpr, tpr, color=COLOR[p], lw=1.6, ls=LS[p], zorder=3,
                label=f"{LABEL[p]} ({int(m['n_variants'])} SNPs)\nAUROC {m['auroc_top20']:.3f} "
                      f"[{m['auroc_top20_ci_lo']:.3f}–{m['auroc_top20_ci_hi']:.3f}]")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([0, .2, .4, .6, .8, 1]); ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_xlabel("False-positive rate (1 − specificity)")
    ax.set_ylabel("True-positive rate (sensitivity)")
    n_case = int(m["n_cases_top20"]); N = int(m["N_test"])
    ax.set_title(f"{sex.capitalize()} test set (N = {N:,}; tall = top 20%, n = {n_case:,})", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, handlelength=2.4, labelspacing=0.6, borderaxespad=0.2, fontsize=6.6)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    # ---------- deciles ----------
    ax = axes[1, col]
    dsub = D[D["sex"] == sex]
    mean_h = M.loc[(sex, "GWAS_sig"), "height_mean"]
    ax.axhline(mean_h, color="#c3c2b7", lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(10.5, mean_h, f"test-set mean {mean_h:.1f} cm", color=MUTED, fontsize=6.5, va="top", ha="right")
    for k, p in enumerate(PANELS):
        dd = dsub[dsub["panel"] == p].sort_values("decile")
        x = dd["decile"].to_numpy() + (k - 1) * 0.16
        ax.errorbar(x, dd["mean_height"], yerr=[dd["mean_height"] - dd["ci_lo"], dd["ci_hi"] - dd["mean_height"]],
                    color=COLOR[p], fmt=MK[p], ms=4, mfc=COLOR[p], mec="white", mew=0.6, lw=1.3, ls=LS[p],
                    capsize=1.5, elinewidth=0.8, zorder=3,
                    label=f"{LABEL[p]}: D10 − D1 = {M.loc[(sex, p), 'decile10_minus_decile1_cm']:.1f} cm")
    ax.set_xticks(range(1, 11)); ax.set_xlim(0.4, 10.6)
    ax.set_xlabel("PGS decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Mean measured height, cm (95% CI)")
    # y-range: pad around observed decile means
    lo, hi = dsub["ci_lo"].min(), dsub["ci_hi"].max(); pad = (hi - lo) * 0.12
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{sex.capitalize()}: measured height by PGS decile", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False, handlelength=2.6, ncol=1)
    ax.grid(color=GRID, lw=0.5, axis="y"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

for ax, letter in zip(axes.flat, "ABCD"):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
fig.tight_layout(h_pad=2.2, w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(RES, f"R05_Figure5_PRS_benchmark.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
try:
    fig.savefig(os.path.join(RES, "R05_Figure5_PRS_benchmark.tif"), dpi=300, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
except Exception as e:
    print("TIFF export skipped:", e)
print("saved R05_Figure5_PRS_benchmark.*")
