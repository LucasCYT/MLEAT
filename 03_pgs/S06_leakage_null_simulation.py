#!/usr/bin/env python3
"""
S06_leakage_null_simulation.py -> 03_results/R09_leakage_null_simulation.csv
Diagnostic for README open question #1: how large can the PGS correlation be when variants are
SELECTED by test-set dMSE and the PGS is then EVALUATED on the same test set, even if no SNP has any
true effect?  Weights are training-set betas throughout (as in the real pipeline).

Idealised model (standardised units, independent SNPs, no LD, no true effects):
  beta_hat_j ~ N(0, 1/N_train)        training-GWAS effect estimate
  c_j        ~ N(0, 1/N_test)         test-set sample covariance of genotype j with residual height
  dMSE_j     = 2*beta_hat_j*c_j - beta_hat_j^2      change in test MSE when SNP j (with beta_hat_j) is added
  Kneedle-style panel = top-K by dMSE_j; "LD pruning" mimicked by keeping a random subset of size K_pruned
  PGS r on the same test set  = sum(beta*c) / sqrt(sum(beta^2))
  PGS r on an independent set = sum(beta*c_new) / sqrt(sum(beta^2)),  c_new fresh draws
Compare with a training-only selected panel (top-K by |beta_hat|), which is the GWAS-significant analogue.
"""
import os, numpy as np, pandas as pd
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03_results")
rng = np.random.default_rng(20260908)
M = 510_276
rows = []
for sex, N_train, N_test, K, K_pruned in [("female", 33_812, 8_454, 1048, 321), ("male", 26_345, 6_587, 378, 75)]:
    for rep in range(50):
        b = rng.normal(0, 1/np.sqrt(N_train), M); c = rng.normal(0, 1/np.sqrt(N_test), M)
        sel = np.argsort(-(2*b*c - b**2))[:K]
        keep = rng.choice(sel, K_pruned, replace=False)
        c_new = rng.normal(0, 1/np.sqrt(N_test), K_pruned)
        sel_g = rng.choice(np.argsort(-np.abs(b))[:K], K_pruned, replace=False)
        rows.append(dict(sex=sex, rep=rep, K=K, K_pruned=K_pruned,
                         r_testselected_same_testset=(b[keep]*c[keep]).sum()/np.sqrt((b[keep]**2).sum()),
                         r_testselected_independent_set=(b[keep]*c_new).sum()/np.sqrt((b[keep]**2).sum()),
                         r_trainselected_same_testset=(b[sel_g]*c[sel_g]).sum()/np.sqrt((b[sel_g]**2).sum()),
                         frac_sign_agree_selected=np.mean(np.sign(b[keep]) == np.sign(c[keep]))))
df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "R09_leakage_null_simulation.csv"), index=False)
s = df.groupby("sex")[["r_testselected_same_testset", "r_testselected_independent_set", "r_trainselected_same_testset"]].agg(["mean", "std"])
print("PURE NULL (no true SNP effects):"); print(s.round(3).to_string())
print("Observed Kneedle-panel PGS r in the real test set: female 0.42, male 0.27")
