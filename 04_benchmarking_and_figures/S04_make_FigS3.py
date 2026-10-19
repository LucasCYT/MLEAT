#!/usr/bin/env python3
"""
S04_make_FigS3.py -> 03_results/R06_FigureS3_PRS_metrics_summary.{png,pdf}
Supplementary figure: sensitivity of the PRS benchmark to the binary height definition and
the continuous metrics behind Figure 5 (three out-of-sample panels; the in-sample Kneedle reference was dropped on 2026-09-10).
  A/B  AUROC (DeLong 95% CI) for top 10%, top 20% and bottom 20% height, female (A) / male (B)
  C    Pearson r (Fisher-z 95% CI) between PRS and measured height, by sex and panel
  D    Incremental R2 (%) of each PRS over a covariate-only model (AGE + PC1-PC20)
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "03_results")
PANELS = ["GWAS_sig", "MLEAT_FDR", "MLEAT_trainCV"]          # out-of-sample panels only (Kneedle in-sample reference dropped 2026-09-10)
LABEL = {"GWAS_sig": "GWAS-significant", "MLEAT_FDR": "ML-EAT FDR ≤ 5%", "MLEAT_trainCV": "ML-EAT training-partition set",
         "MLEAT_Knee": "ML-EAT Kneedle set (in-sample)"}
COLOR = {"GWAS_sig": "#2a78d6", "MLEAT_FDR": "#eb6834", "MLEAT_trainCV": "#1baf7a", "MLEAT_Knee": "#898781"}
MK = {"GWAS_sig": "o", "MLEAT_FDR": "s", "MLEAT_trainCV": "^", "MLEAT_Knee": "D"}
MFC = {p: c for p, c in COLOR.items()}; MFC["MLEAT_Knee"] = "white"      # hollow marker = in-sample reference
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "legend.fontsize": 7.2,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
M = pd.read_csv(os.path.join(RES, "R01_prs_metrics_long.csv"))
DEFS = [("top10", "Top 10%\n(tall)"), ("top20", "Top 20%\n(tall)"), ("bottom20", "Bottom 20%\n(short)")]

fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
def style(ax):
    ax.grid(color=GRID, lw=0.5, axis="y"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

for col, sex in enumerate(["female", "male"]):
    ax = axes[0, col]; sub = M[M.sex == sex].set_index("panel")
    ax.axhline(0.5, color="#c3c2b7", lw=0.8, ls=(0, (2, 2)))
    for k, p in enumerate(PANELS):
        x = np.arange(len(DEFS)) + (k - 1) * 0.22
        a = [sub.loc[p, f"auroc_{d}"] for d, _ in DEFS]
        lo = [sub.loc[p, f"auroc_{d}_ci_lo"] for d, _ in DEFS]; hi = [sub.loc[p, f"auroc_{d}_ci_hi"] for d, _ in DEFS]
        ax.errorbar(x, a, yerr=[np.subtract(a, lo), np.subtract(hi, a)], fmt=MK[p], color=COLOR[p], ms=5, mfc=MFC[p], mec=COLOR[p] if p == "MLEAT_Knee" else "white", mew=0.8,
                    capsize=2, elinewidth=1, lw=0, label=f"{LABEL[p]} ({int(sub.loc[p, 'n_variants'])} SNPs)")
        for xi, ai in zip(x, a):
            ax.text(xi, ai + 0.012, f"{ai:.2f}", ha="center", va="bottom", fontsize=6.2, color=INK)
    ax.set_xticks(range(len(DEFS))); ax.set_xticklabels([t for _, t in DEFS])
    ax.set_ylim(0.48, 0.72); ax.set_yticks([0.5,0.55,0.6,0.65,0.7]); ax.set_ylabel("AUROC (DeLong 95% CI)")
    ax.set_title(f"{sex.capitalize()} test set (N = {int(sub.loc['GWAS_sig','N_test']):,})", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False); style(ax)

# C: Pearson r
ax = axes[1, 0]
for k, p in enumerate(PANELS):
    x = np.arange(2) + (k - 1) * 0.22
    r = [M[(M.sex == s) & (M.panel == p)]["pearson_r"].item() for s in ("female", "male")]
    lo = [M[(M.sex == s) & (M.panel == p)]["r_ci_lo"].item() for s in ("female", "male")]
    hi = [M[(M.sex == s) & (M.panel == p)]["r_ci_hi"].item() for s in ("female", "male")]
    ax.errorbar(x, r, yerr=[np.subtract(r, lo), np.subtract(hi, r)], fmt=MK[p], color=COLOR[p], ms=5, mfc=MFC[p], mec=COLOR[p] if p == "MLEAT_Knee" else "white", mew=0.8,
                capsize=2, elinewidth=1, lw=0, label=LABEL[p])
    for xi, ri in zip(x, r): ax.text(xi, ri + 0.012, f"{ri:.2f}", ha="center", va="bottom", fontsize=6.2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Female", "Male"]); ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 0.34); ax.set_ylabel("Pearson r, PGS vs height (95% CI)")
ax.set_title("Correlation with measured height", loc="left", fontweight="bold"); ax.legend(loc="upper left", frameon=False, fontsize=6.8); style(ax)

# D: incremental R2
ax = axes[1, 1]; w = 0.22
for k, p in enumerate(PANELS):
    x = np.arange(2) + (k - 1.5) * w
    v = [100 * M[(M.sex == s) & (M.panel == p)]["r2_incremental"].item() for s in ("female", "male")]
    ax.bar(x, v, width=w - 0.02, color=COLOR[p] if p != "MLEAT_Knee" else "white", edgecolor=COLOR[p], hatch="///" if p == "MLEAT_Knee" else None, lw=0.8, label=LABEL[p], zorder=3)
    for xi, vi in zip(x, v): ax.text(xi, vi + 0.3, f"{vi:.1f}", ha="center", va="bottom", fontsize=6.2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Female", "Male"]); ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("Incremental R² over age + PC1–PC20 (%)"); ax.set_ylim(0, 8)
ax.set_title("Variance explained beyond covariates", loc="left", fontweight="bold"); ax.legend(loc="upper right", frameon=False, fontsize=6.8); style(ax)

for ax, letter in zip(axes.flat, "ABCD"):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")
fig.tight_layout(h_pad=2.2, w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(RES, f"R06_FigureS3_PRS_metrics_summary.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
print("saved R06_FigureS3_PRS_metrics_summary.*")
