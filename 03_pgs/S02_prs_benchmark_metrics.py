#!/usr/bin/env python3
"""
S02_prs_benchmark_metrics.py
Out-of-sample benchmark of three sex-specific PRS panels on the held-out 20% test set.

Panels (weights = BETA from the sex-specific 80%-training-set GWAS, PLINK --score):
  GWAS_sig   : training-GWAS genome-wide-significant variants (P < 5e-8), LD-clumped r2<0.1 / 250 kb
  MLEAT_FDR  : ML-EAT empirical-FDR <= 5% high-confidence set, LD-pruned (--indep-pairwise 250kb 1 0.1)
  MLEAT_Knee : ML-EAT Kneedle prioritization set, LD-pruned (same parameters)

Metrics per sex x panel
  Pearson r (Fisher-z 95% CI, two-sided p), Spearman rho
  R2 (simple) and incremental R2 over a covariate-only model (AGE + PC1-PC20)
  beta per 1 SD of PRS (cm) with SE, from height ~ PRS_z + AGE + PC1-20
  AUROC (DeLong 95% CI) for three binary height definitions:
      top20  : height >= sex-specific 80th percentile of the test set (primary, "tall")
      top10  : height >= 90th percentile
      bottom20 : height <= 20th percentile ("short"; PRS sign flipped so AUROC > 0.5 means discrimination)
  Decile analysis: mean height (95% CI) by PRS decile; top-vs-bottom decile difference
Pairwise comparisons between panels (same individuals):
  AUROC: DeLong paired test; Pearson r and AUROC: paired bootstrap (B = 2000, seed 20260908)

Outputs (03_results/):
  R01_prs_metrics_long.csv        one row per sex x panel
  R02_prs_pairwise_comparisons.csv  one row per sex x definition x panel pair
  R03_prs_decile_means.csv        one row per sex x panel x decile
  R04_prs_individual_scores.tsv.gz   merged per-individual table (IID, height, PRS z-scores)
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
INP = os.path.join(HERE, "..", "01_inputs")
RES = os.path.join(HERE, "..", "03_results")
os.makedirs(RES, exist_ok=True)
SEED, B = 20260908, 2000
PC = [f"PCA{i}" for i in range(1, 21)]
PANELS = ["GWAS_sig", "MLEAT_FDR", "MLEAT_trainCV", "MLEAT_Knee"]   # MLEAT_Knee = selected on the test partition (in-sample)
FILES = {
    "female": {"pheno": "I01_female_test20_pheno.tsv",
               "GWAS_sig": "I03_female_PRS_GWASsig_clumped.profile",
               "MLEAT_FDR": "I04_female_PRS_MLEAT_FDR_pruned.profile",
               "MLEAT_Knee": "I05_female_PRS_MLEAT_Kneedle_pruned.profile",
               "MLEAT_trainCV": "I18_female_PRS_MLEAT_trainCV_clumped.profile"},
    "male":   {"pheno": "I02_male_test20_pheno.tsv",
               "GWAS_sig": "I06_male_PRS_GWASsig_clumped.profile",
               "MLEAT_FDR": "I07_male_PRS_MLEAT_FDR_pruned.profile",
               "MLEAT_Knee": "I08_male_PRS_MLEAT_Kneedle_pruned.profile",
               "MLEAT_trainCV": "I19_male_PRS_MLEAT_trainCV_clumped.profile"},
}
DEFS = {"top20": (80, +1), "top10": (90, +1), "bottom20": (20, -1)}

# ---------------------------------------------------------------- DeLong ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T; return out

def delong(preds, y):
    """preds: (k, n) scores; y: (n,) binary. Returns aucs (k,), covariance S (k,k). Sun & Xu 2014."""
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    m, n = len(pos), len(neg); k = preds.shape[0]
    P = np.hstack([preds[:, pos], preds[:, neg]])
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(P[r, :m]); ty[r] = _midrank(P[r, m:]); tz[r] = _midrank(P[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    S = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, S

def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    r2 = 1 - resid.var() / y.var()
    dof = len(y) - X1.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X1.T @ X1)
    return beta, np.sqrt(np.diag(cov)), r2

def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

# ------------------------------------------------------------- main loop ----
metrics, pairwise, deciles, indiv = [], [], [], []
rng = np.random.default_rng(SEED)
for sex in ["female", "male"]:
    ph = pd.read_csv(os.path.join(INP, FILES[sex]["pheno"]), sep="\t")
    df = ph[["IID", "AGE", "BODY_HEIGHT"] + PC].copy()
    nsnp = {}
    for p in PANELS:
        pr = pd.read_csv(os.path.join(INP, FILES[sex][p]), sep="\t")
        nsnp[p] = int(pr["CNT"].max() // 2)
        df = df.merge(pr[["IID", "SCORE"]].rename(columns={"SCORE": f"PRS_{p}"}), on="IID", how="inner")
    df = df.dropna(subset=["BODY_HEIGHT"] + [f"PRS_{p}" for p in PANELS]).reset_index(drop=True)
    N = len(df); y = df["BODY_HEIGHT"].to_numpy(float)
    for p in PANELS:
        s = df[f"PRS_{p}"].to_numpy(float); df[f"PRSz_{p}"] = (s - s.mean()) / s.std(ddof=1)
    cov = df[["AGE"] + PC].to_numpy(float)
    _, _, r2_base = ols(cov, y)
    labels = {}
    for d, (pct, sign) in DEFS.items():
        thr = np.percentile(y, pct)
        labels[d] = ((y >= thr) if sign > 0 else (y <= thr)).astype(int)
    # ---- per-panel metrics ----
    Zs = np.vstack([df[f"PRSz_{p}"].to_numpy(float) for p in PANELS])
    auc_res = {}
    for d, (pct, sign) in DEFS.items():
        aucs, S = delong(sign * Zs, labels[d]); auc_res[d] = (aucs, S)
    sex_metrics = {}
    for i, p in enumerate(PANELS):
        z = Zs[i]
        r, pval = stats.pearsonr(z, y); lo, hi = fisher_ci(r, N)
        rho, _ = stats.spearmanr(z, y)
        beta, se, r2_full = ols(np.column_stack([z, cov]), y)
        row = dict(sex=sex, panel=p, n_variants=nsnp[p], N_test=N,
                   pearson_r=r, r_ci_lo=lo, r_ci_hi=hi, pearson_p=pval, spearman_rho=rho,
                   r2_simple=r ** 2, r2_covariates_only=r2_base, r2_incremental=r2_full - r2_base,
                   beta_cm_per_SD=beta[1], beta_se=se[1], beta_p=2 * stats.norm.sf(abs(beta[1] / se[1])),
                   height_mean=y.mean(), height_sd=y.std(ddof=1))
        for d in DEFS:
            a, S = auc_res[d]; sd = np.sqrt(S[i, i])
            row[f"auroc_{d}"] = a[i]; row[f"auroc_{d}_ci_lo"] = a[i] - 1.96 * sd; row[f"auroc_{d}_ci_hi"] = a[i] + 1.96 * sd
            row[f"n_cases_{d}"] = int(labels[d].sum())
        # deciles
        dec = pd.qcut(z, 10, labels=False) + 1
        for k in range(1, 11):
            yy = y[dec == k]
            deciles.append(dict(sex=sex, panel=p, decile=k, n=len(yy), mean_height=yy.mean(), sd=yy.std(ddof=1),
                                ci_lo=yy.mean() - 1.96 * yy.std(ddof=1) / np.sqrt(len(yy)),
                                ci_hi=yy.mean() + 1.96 * yy.std(ddof=1) / np.sqrt(len(yy)),
                                pct_tall_top20=labels["top20"][dec == k].mean() * 100))
        row["decile10_minus_decile1_cm"] = y[dec == 10].mean() - y[dec == 1].mean()
        t, tp = stats.ttest_ind(y[dec == 10], y[dec == 1], equal_var=False); row["decile10_vs_1_p"] = tp
        metrics.append(row); sex_metrics[p] = row
    # ---- pairwise: DeLong + paired bootstrap ----
    nP = len(PANELS)
    boot_r = np.zeros((B, nP)); boot_auc = {d: np.zeros((B, nP)) for d in DEFS}
    for b in range(B):
        idx = rng.integers(0, N, N); yb = y[idx]
        for i in range(nP):
            zb = Zs[i][idx]
            boot_r[b, i] = np.corrcoef(zb, yb)[0, 1]
            for d, (pct, sign) in DEFS.items():
                boot_auc[d][b, i] = roc_auc_score(labels[d][idx], sign * zb)
    pairs = [(i, j) for i in range(nP) for j in range(nP) if i > j]   # every panel pair, A = later in PANELS
    for i, j in pairs:
        dr = boot_r[:, i] - boot_r[:, j]
        pr_boot = 2 * min((dr <= 0).mean(), (dr >= 0).mean())
        rA, rB = sex_metrics[PANELS[i]]["pearson_r"], sex_metrics[PANELS[j]]["pearson_r"]
        base = dict(sex=sex, panel_A=PANELS[i], panel_B=PANELS[j], r_A=rA, r_B=rB, r_diff=rA - rB,
                    r_diff_boot_ci_lo=np.percentile(dr, 2.5), r_diff_boot_ci_hi=np.percentile(dr, 97.5),
                    r_diff_boot_p=max(pr_boot, 1 / B))
        for d in DEFS:
            a, S = auc_res[d]
            zstat = (a[i] - a[j]) / np.sqrt(S[i, i] + S[j, j] - 2 * S[i, j])
            da = boot_auc[d][:, i] - boot_auc[d][:, j]
            pa_boot = 2 * min((da <= 0).mean(), (da >= 0).mean())
            pairwise.append(dict(base, definition=d, auroc_A=a[i], auroc_B=a[j], auroc_diff=a[i] - a[j],
                                 delong_z=zstat, delong_p=2 * stats.norm.sf(abs(zstat)),
                                 auroc_diff_boot_ci_lo=np.percentile(da, 2.5), auroc_diff_boot_ci_hi=np.percentile(da, 97.5),
                                 auroc_diff_boot_p=max(pa_boot, 1 / B)))
    keep = ["IID", "AGE", "BODY_HEIGHT"] + [f"PRS_{p}" for p in PANELS] + [f"PRSz_{p}" for p in PANELS]
    out = df[keep].copy(); out.insert(0, "sex", sex)
    for d in DEFS: out[f"label_{d}"] = labels[d]
    indiv.append(out)
    print(f"{sex}: N={N}, cases top20={labels['top20'].sum()}, r2_base(age+PC)={r2_base:.4f}")

M = pd.DataFrame(metrics); M.to_csv(os.path.join(RES, "R01_prs_metrics_long.csv"), index=False)
P = pd.DataFrame(pairwise); P.to_csv(os.path.join(RES, "R02_prs_pairwise_comparisons.csv"), index=False)
D = pd.DataFrame(deciles); D.to_csv(os.path.join(RES, "R03_prs_decile_means.csv"), index=False)
pd.concat(indiv).to_csv(os.path.join(RES, "R04_prs_individual_scores.tsv.gz"), sep="\t", index=False, compression="gzip")

pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: f"{v:.4g}")
print(M[["sex", "panel", "n_variants", "N_test", "pearson_r", "r_ci_lo", "r_ci_hi", "r2_incremental", "beta_cm_per_SD",
         "auroc_top20", "auroc_top20_ci_lo", "auroc_top20_ci_hi", "auroc_top10", "auroc_bottom20", "decile10_minus_decile1_cm"]].to_string(index=False))
print(P[["sex", "definition", "panel_A", "panel_B", "auroc_diff", "delong_p", "r_diff", "r_diff_boot_ci_lo", "r_diff_boot_ci_hi", "r_diff_boot_p"]].to_string(index=False))
