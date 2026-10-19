#!/usr/bin/env python3
"""
S01_prepare_inputs.py
Collect every input the Figure 5 (PRS benchmark) pipeline needs into 01_inputs/,
so that S02-S05 can run without touching the raw Figure5 folder.

Run from the Figure5 folder:   python3 02_scripts/S01_prepare_inputs.py
Raw sources (../Source/):
  Merged_{Female,Male}_Data_withTWB2ID_*_test20.txt      -> I01/I02 (held-out 20% test-set phenotypes + covariates)
  {female,male}_prs_test20_result.profile                -> I03/I06 (PLINK --score, GWAS-significant clumped panel)
  --indep-pairwise (LD Pruning)結果/*FDR*_result.profile  -> I04/I07 (ML-EAT empirical-FDR <=5% panel, LD-pruned)
  --indep-pairwise (LD Pruning)結果/*NewKneePoint_LD*.profile -> I05/I08 (ML-EAT Kneedle panel, LD-pruned)
  weight files (SNP, A1, BETA from the 80% training GWAS)  -> I09-I14
  {female,male}_GWAS_BODY_HEIGHT_*_80_ADD.assoc.linear   -> I15/I16 (rows for panel SNPs only) + I17 (summary counts)
  PLINK .log files                                       -> 01_inputs/logs/
"""
import os, shutil, glob
import pandas as pd
import numpy as np

RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Source"))   # raw PLINK output (moved here 2026-09-09)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_inputs"))
LD  = os.path.join(RAW, "--indep-pairwise (LD Pruning)結果")
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)

PC = [f"PCA{i}" for i in range(1, 21)]

def safe_write(df, path, **kw):
    """Google Drive occasionally holds a lock on a synced file (OSError 35). The file content is
    deterministic, so a locked destination that already exists is left as it is with a warning."""
    try:
        df.to_csv(path, **kw); return True
    except OSError as e:
        if os.path.exists(path):
            print(f"WARNING: {os.path.basename(path)} is locked by the file system ({e.strerror}); keeping the existing copy")
            return False
        raise

# ---- phenotypes (test 20%) ----------------------------------------------------
pheno_src = {"female": "Merged_Female_Data_withTWB2ID_C37_test20.txt",
             "male":   "Merged_Male_Data_withTWB2ID_C33_test20.txt"}
pheno_out = {"female": "I01_female_test20_pheno.tsv", "male": "I02_male_test20_pheno.tsv"}
for sex, f in pheno_src.items():
    df = pd.read_csv(os.path.join(RAW, f), sep=r"\s+", engine="python")
    keep = ["IID", "AGE", "BODY_HEIGHT"] + PC
    df = df[keep].copy()
    df["BODY_HEIGHT"] = pd.to_numeric(df["BODY_HEIGHT"], errors="coerce")
    safe_write(df, os.path.join(OUT, pheno_out[sex]), sep="\t", index=False)
    print(f"{pheno_out[sex]}: {len(df)} rows, {df['BODY_HEIGHT'].isna().sum()} missing height")

# ---- PLINK --score profiles ----------------------------------------------------
profiles = {
    ("female", "GWAS_sig"):   (os.path.join(RAW, "female_prs_test20_result.profile"),                 "I03_female_PRS_GWASsig_clumped.profile"),
    ("female", "MLEAT_FDR"):  (os.path.join(LD,  "female_prs_test20_FDR_LD_14_result.profile"),        "I04_female_PRS_MLEAT_FDR_pruned.profile"),
    ("female", "MLEAT_Knee"): (os.path.join(LD,  "female_prs_test20_NewKneePoint_LD_result.profile"),  "I05_female_PRS_MLEAT_Kneedle_pruned.profile"),
    ("male",   "GWAS_sig"):   (os.path.join(RAW, "male_prs_test20_result.profile"),                   "I06_male_PRS_GWASsig_clumped.profile"),
    ("male",   "MLEAT_FDR"):  (os.path.join(LD,  "male_prs_test20_FDR_LD_result.profile"),             "I07_male_PRS_MLEAT_FDR_pruned.profile"),
    ("male",   "MLEAT_Knee"): (os.path.join(LD,  "male_prs_test20_NewKneePoint_LD_result.profile"),    "I08_male_PRS_MLEAT_Kneedle_pruned.profile"),
}
for (sex, panel), (src, dst) in profiles.items():
    df = pd.read_csv(src, sep=r"\s+", engine="python")
    safe_write(df, os.path.join(OUT, dst), sep="\t", index=False)
    print(f"{dst}: {len(df)} individuals, CNT(max)={df['CNT'].max()} -> {df['CNT'].max()//2} variants scored")

# ---- training-only ML-EAT panel (S07/S08, profiles returned from the cluster 2026-09-09) ----
TCV = os.path.join(OUT, "trainCV_panel")
for sex, dst in [("female", "I18_female_PRS_MLEAT_trainCV_clumped.profile"), ("male", "I19_male_PRS_MLEAT_trainCV_clumped.profile")]:
    src = os.path.join(TCV, f"{sex}_prs_test20_MLEAT_trainCV_clumped_result.profile")
    if os.path.exists(src):
        df = pd.read_csv(src, sep=r"\s+", engine="python")
        safe_write(df, os.path.join(OUT, dst), sep="\t", index=False)
        print(f"{dst}: {len(df)} individuals, CNT(max)={df['CNT'].max()} -> {df['CNT'].max()//2} variants scored")
    else:
        print(f"WARNING: {src} not found; training-CV panel skipped")

# ---- weight files ------------------------------------------------------------
weights = {
    ("female", "GWAS_sig"):   (os.path.join(RAW, "female_prs_weights.txt"),                         "I09_female_weights_GWASsig_clumped.txt"),
    ("female", "MLEAT_FDR"):  (os.path.join(LD,  "female_prs_weights_FDR_from84_LD.txt"),            "I10_female_weights_MLEAT_FDR_pruned.txt"),
    ("female", "MLEAT_Knee"): (os.path.join(LD,  "female_prs_weights_NewKneePoint_LD.txt"),          "I11_female_weights_MLEAT_Kneedle_pruned.txt"),
    ("male",   "GWAS_sig"):   (os.path.join(RAW, "male_prs_weights.txt"),                           "I12_male_weights_GWASsig_clumped.txt"),
    ("male",   "MLEAT_FDR"):  (os.path.join(LD,  "male_prs_weights_FDR_LD.txt"),                     "I13_male_weights_MLEAT_FDR_pruned.txt"),
    ("male",   "MLEAT_Knee"): (os.path.join(LD,  "male_prs_weights_NewKneePoint_LD.txt"),            "I14_male_weights_MLEAT_Kneedle_pruned.txt"),
    ("female", "MLEAT_trainCV"): (os.path.join(OUT, "trainCV_panel", "female_prs_weights_MLEAT_trainCV_clumped.txt"), "I20_female_weights_MLEAT_trainCV_clumped.txt"),
    ("male",   "MLEAT_trainCV"): (os.path.join(OUT, "trainCV_panel", "male_prs_weights_MLEAT_trainCV_clumped.txt"),   "I21_male_weights_MLEAT_trainCV_clumped.txt"),
}
panel_snps = {"female": {}, "male": {}}
for (sex, panel), (src, dst) in weights.items():
    if not os.path.exists(src):
        print(f"WARNING: {src} missing; panel {panel} skipped"); continue
    df = pd.read_csv(src, sep="\t")
    safe_write(df, os.path.join(OUT, dst), sep="\t", index=False)
    panel_snps[sex][panel] = set(df["SNP"])
    print(f"{dst}: {len(df)} SNPs")

# ---- training-set GWAS: subset rows + summary counts ------------------------
assoc = {"female": "female_GWAS_BODY_HEIGHT_c33_80_ADD.assoc.linear",
         "male":   "male_GWAS_BODY_HEIGHT_c29_80_ADD.assoc.linear"}
sub_out = {"female": "I15_female_trainGWAS_panelSNPs.tsv", "male": "I16_male_trainGWAS_panelSNPs.tsv"}
summary = []
for sex, f in assoc.items():
    g = pd.read_csv(os.path.join(RAW, f), sep=r"\s+", engine="c",
                    usecols=["CHR", "SNP", "BP", "A1", "NMISS", "BETA", "STAT", "P"])
    g["P"] = pd.to_numeric(g["P"], errors="coerce")
    n_all = len(g)
    summary.append({"sex": sex, "file": f, "n_variants_tested": n_all,
                    "n_P_lt_5e-8": int((g["P"] < 5e-8).sum()),
                    "n_P_lt_1e-5": int((g["P"] < 1e-5).sum()),
                    "n_P_lt_1e-3": int((g["P"] < 1e-3).sum()),
                    "N_training_median_NMISS": int(g["NMISS"].median())})
    allsnps = set().union(*panel_snps[sex].values())
    sub = g[g["SNP"].isin(allsnps)].copy()
    for panel, s in panel_snps[sex].items():
        sub[f"in_{panel}"] = sub["SNP"].isin(s).astype(int)
    safe_write(sub.sort_values("P"), os.path.join(OUT, sub_out[sex]), sep="\t", index=False)
    print(f"{sub_out[sex]}: {len(sub)} rows")
safe_write(pd.DataFrame(summary), os.path.join(OUT, "I17_trainGWAS_summary.csv"), index=False)
print(pd.DataFrame(summary).to_string(index=False))

# ---- PLINK logs --------------------------------------------------------------
for src in glob.glob(os.path.join(RAW, "*.log")) + glob.glob(os.path.join(LD, "*.log")) + \
           glob.glob(os.path.join(RAW, "*.nopred")) + glob.glob(os.path.join(LD, "*.nopred")):
    shutil.copy(src, os.path.join(OUT, "logs", os.path.basename(src)))
print("logs copied:", len(os.listdir(os.path.join(OUT, "logs"))))
