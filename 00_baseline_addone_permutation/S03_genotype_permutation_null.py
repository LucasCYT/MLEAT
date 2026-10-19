#!/usr/bin/env python3
"""
S03_genotype_permutation_null.py
Genotype-permutation null for the ML-EAT empirical FDR (paper Methods,
"Permutation-based empirical significance").

For each sampled SNP:
  * permutation_id 0  : the observed add-one fit (BM features + that SNP);
  * permutation_id k>0: the SNP's genotype vector is permuted across individuals
                        WITHIN THE 80 % TRAINING PARTITION ONLY
                        (numpy RandomState(k).permutation), while the phenotype,
                        all baseline features and the 20 % test partition are
                        left unchanged; the add-one model is refitted and its
                        held-out MSE recorded.
1,000 permutations per SNP.  SNPs are sampled from three strata of the
genome-wide add-one ranking (top / middle / bottom; 1,000 per stratum in
females, 300 in males), giving 3.0 x 10^6 (female) and 9.0 x 10^5 (male) null
values after pooling (S04 + 01_prioritization_and_fdr).

Input
  --data-dir   directory with per-sex analysis matrices (*.mtx or *.mtx.csv);
               each file holds BODY_HEIGHT, the baseline features and a block
               of SNP dosage columns (column names rsID_allele).  The sex is
               taken from the file name ('female' / 'male').
  --snp-list   CSV whose first column lists the stratum's SNPs (rsID_allele),
               e.g. New_Knee_Point_SNP_List_Female_Top1000.csv
Output (in --results-dir/permutation_{sex}/)
  {SNP}.csv    columns: permutation_id, mse, mse_cv, permuted_snp_train_value
               (mse = held-out test MSE; permutation_id 0 = observed)
  summary.csv  one row per processed SNP

No phenotype outlier filtering is applied (all participants retained).

Usage (one stratum, one sex; see slurm/ for the cluster job scripts):
  python S03_genotype_permutation_null.py \
      --data-dir  mergerdataFolder/female_S0_S51/ \
      --snp-list  New_Knee_Point_SNP_List_Female_Top1000.csv \
      --results-dir results_filter_v4_NoFilter_SnpTrainShuff/ \
      --sex female --permutations 1000

Provenance: cleaned from run_analysis_T3_filtered_v4_{,Middle_,Last_}NO_FILTER_
alpha001_SNPTrainShuffle.py (2026-07-30); the three stratum-specific copies,
which differed only in the hard-coded SNP-list file name, are replaced by the
--snp-list argument.  The fitting and permutation code is unchanged.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mleat_core import PHENOTYPE, bm_features, detect_sex, fit_and_score, split_indices


def find_matrix_files(data_dir):
    p = Path(data_dir)
    return sorted(list(p.glob('*.mtx')) + list(p.glob('*.mtx.csv')))


def load_snp_list(path):
    return pd.read_csv(path).iloc[:, 0].tolist()


def load_matrix(data_file, snp_list, features):
    """Load BODY_HEIGHT + available baseline features + available stratum SNPs."""
    header = pd.read_csv(data_file, nrows=0).columns.tolist()
    feats = [f for f in features if f in header]
    snps = [s for s in snp_list if s in header]
    df = pd.read_csv(data_file, usecols=[PHENOTYPE] + feats + snps)
    return df, snps, feats


def permutation_null_for_snp(df, snp_col, features, n_permutations):
    """Observed fit (permutation_id 0) + n_permutations train-only genotype permutations."""
    X = df[features + [snp_col]].values
    y = df[PHENOTYPE].values
    train_idx, test_idx = split_indices(len(y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rows = []
    # permutation_id 0: observed genotype
    try:
        mse, mse_cv = fit_and_score(X_train, X_test, y_train, y_test)
        rows.append({'permutation_id': 0, 'mse': mse, 'mse_cv': mse_cv,
                     'permuted_snp_train_value': X_train[0, -1]})
    except Exception as e:  # keep the run going; record NaN
        print(f"    permutation 0 failed: {e}")
        rows.append({'permutation_id': 0, 'mse': np.nan, 'mse_cv': np.nan,
                     'permuted_snp_train_value': np.nan})

    # permutation_id 1..N: permute the SNP column within the training partition only
    for perm_id in range(1, n_permutations + 1):
        X_train_perm = X_train.copy()
        X_train_perm[:, -1] = np.random.RandomState(perm_id).permutation(X_train_perm[:, -1])
        try:
            mse, mse_cv = fit_and_score(X_train_perm, X_test, y_train, y_test)
            rows.append({'permutation_id': perm_id, 'mse': mse, 'mse_cv': mse_cv,
                         'permuted_snp_train_value': X_train_perm[0, -1]})
        except Exception as e:
            print(f"    permutation {perm_id} failed: {e}")
            rows.append({'permutation_id': perm_id, 'mse': np.nan, 'mse_cv': np.nan,
                         'permuted_snp_train_value': np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="ML-EAT genotype-permutation null (train-only SNP shuffle)")
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--snp-list', required=True, help="CSV, first column = stratum SNPs (rsID_allele)")
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--sex', choices=['female', 'male'], required=True)
    ap.add_argument('--permutations', type=int, default=1000)
    a = ap.parse_args()

    files = [f for f in find_matrix_files(a.data_dir) if detect_sex(f.name) == a.sex]
    if not files:
        sys.exit(f"no {a.sex} *.mtx / *.mtx.csv files in {a.data_dir}")
    snp_list = load_snp_list(a.snp_list)
    features = bm_features(a.sex)
    out_dir = Path(a.results_dir) / f"permutation_{a.sex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{a.sex}: {len(files)} matrix file(s), {len(snp_list)} stratum SNPs, "
          f"{len(features)} BM features, {a.permutations} permutations")

    summary = []
    for data_file in files:
        df, snps, feats = load_matrix(data_file, snp_list, features)
        if len(feats) != len(features):
            print(f"  warning: {data_file.name} lacks BM features "
                  f"{sorted(set(features) - set(feats))}")
        print(f"  {data_file.name}: {len(df)} participants, {len(snps)} stratum SNPs")
        for i, snp in enumerate(snps, 1):
            res = permutation_null_for_snp(df, snp, feats, a.permutations)
            res.to_csv(out_dir / f"{snp}.csv", index=False)
            summary.append({'SNP': snp, 'permutations': len(res), 'source_file': data_file.name})
            print(f"    [{i}/{len(snps)}] {snp}: observed test MSE = {res.loc[0, 'mse']:.6f}")
    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print(f"done: {len(summary)} SNPs -> {out_dir}")


if __name__ == '__main__':
    main()
