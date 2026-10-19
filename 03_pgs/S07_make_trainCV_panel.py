#!/usr/bin/env python3
"""
S07_make_trainCV_panel.py
Training-only ML-EAT prioritization panel for the PGS benchmark (no re-run of the add-one screen needed).

Uses the per-SNP cross-validated MSE (5-fold CV inside the 80 % training partition, column `lasso_mse_cv`
of NewData/{sex}_cv_5_lasso_aligned_merged_20260702.csv).  dMSE_CV = MSE_CV(baseline) - MSE_CV(baseline + SNP).
The panel is the top-K SNPs by dMSE_CV with K fixed at the published held-out Kneedle set size
(1,048 female / 378 male), so the 20 % test partition is never used for selection.

Outputs (03_results/):
  R10_{sex}_MLEAT_trainCV_ranking.tsv.gz      rsID, dMSE_CV, rank_CV, dMSE_heldout, rank_heldout, trainGWAS A1/BETA/P
  R11_{sex}_MLEAT_trainCV_top{K}_clumpinput.txt   SNP + pseudo-P (= rank_CV / M) for PLINK --clump (index SNP = best dMSE_CV per locus)
  R11_{sex}_MLEAT_trainCV_top{K}_snplist.txt      plain SNP list (for --extract)
  R12_trainCV_panel_summary.csv               composition of the panels vs the training GWAS
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.abspath(os.path.join(HERE, "..", ".."))                  # NewData/
RES  = os.path.join(HERE, "..", "03_results")
K = {"female": 1048, "male": 378}
ASSOC = {"female": "Figure5/Source/female_GWAS_BODY_HEIGHT_c33_80_ADD.assoc.linear",
         "male":   "Figure5/Source/male_GWAS_BODY_HEIGHT_c29_80_ADD.assoc.linear"}
rows = []
for sex in ["female", "male"]:
    df = pd.read_csv(os.path.join(RAW, f"{sex}_cv_5_lasso_aligned_merged_20260702.csv"),
                     usecols=["Feature", "Feature_aligned", "lasso_mse_test", "lasso_mse_cv"])
    base = df[df.Feature == "basic"].iloc[0]; s = df[df.Feature != "basic"].copy()
    s["dMSE_CV"] = base.lasso_mse_cv - s.lasso_mse_cv
    s["dMSE_heldout"] = base.lasso_mse_test - s.lasso_mse_test
    s = s.rename(columns={"Feature_aligned": "rsID"})[["rsID", "dMSE_CV", "dMSE_heldout"]]
    M = len(s)
    s["rank_CV"] = s.dMSE_CV.rank(ascending=False, method="first").astype(int)
    s["rank_heldout"] = s.dMSE_heldout.rank(ascending=False, method="first").astype(int)
    g = pd.read_csv(os.path.join(RAW, ASSOC[sex]), sep=r"\s+", usecols=["SNP", "A1", "BETA", "P"]).rename(columns={"SNP": "rsID", "A1": "train_A1", "BETA": "train_BETA", "P": "train_P"})
    s = s.merge(g, on="rsID", how="left").sort_values("rank_CV")
    s.to_csv(os.path.join(RES, f"R10_{sex}_MLEAT_trainCV_ranking.tsv.gz"), sep="\t", index=False, compression="gzip")
    top = s.head(K[sex]).copy()
    top["P_pseudo"] = top.rank_CV / M                       # monotone in dMSE_CV; only the ORDER matters for --clump
    top[["rsID", "P_pseudo"]].rename(columns={"rsID": "SNP", "P_pseudo": "P"}).to_csv(
        os.path.join(RES, f"R11_{sex}_MLEAT_trainCV_top{K[sex]}_clumpinput.txt"), sep="\t", index=False)
    top[["rsID"]].to_csv(os.path.join(RES, f"R11_{sex}_MLEAT_trainCV_top{K[sex]}_snplist.txt"), index=False, header=False)
    held = set(s[s.rank_heldout <= K[sex]].rsID)
    rows.append(dict(sex=sex, K=K[sex], n_variants_genomewide=M,
                     n_trainGWAS_P_lt_5e8=int((top.train_P < 5e-8).sum()), n_trainGWAS_P_lt_1e5=int((top.train_P < 1e-5).sum()),
                     n_trainGWAS_P_ge_1e3=int((top.train_P >= 1e-3).sum()), max_train_P=top.train_P.max(),
                     min_dMSE_CV=top.dMSE_CV.min(), overlap_with_heldout_Kneedle_set=len(set(top.rsID) & held),
                     n_missing_in_trainGWAS=int(top.train_P.isna().sum())))
    print(f"{sex}: top-{K[sex]} by dMSE_CV written; train-GWAS P range {top.train_P.min():.1e} – {top.train_P.max():.1e}")
S = pd.DataFrame(rows); S.to_csv(os.path.join(RES, "R12_trainCV_panel_summary.csv"), index=False)
print(S.to_string(index=False))
