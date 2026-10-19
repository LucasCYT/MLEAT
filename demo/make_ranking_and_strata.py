#!/usr/bin/env python3
"""
make_ranking_and_strata.py
(1) Concatenate the per-file add-one results of S02 into one ranking file in the
    format of {sex}_cv_5_lasso_aligned_merged_20260702.csv
    (Feature, columns, lasso_mse_test, lasso_mse_cv; baseline row Feature == 'basic').
(2) Draw the SNP strata for the genotype-permutation null (S03):
      Top    = the N highest-ranked SNPs (lowest held-out MSE)
      Middle = N SNPs drawn at random from the middle third of the ranking
      Last   = the N lowest-ranked SNPs
    written as strata/New_Knee_Point_SNP_List_{Sex}_{Stratum}{N}.csv (column SNP_List).

Usage:
  python make_ranking_and_strata.py --addone-dir lassoLoop_RES_female --sex female \
         --n 20 --ranking-out female_ranking.csv --strata-dir strata
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--addone-dir', required=True)
    ap.add_argument('--sex', choices=['female', 'male'], required=True)
    ap.add_argument('--n', type=int, required=True, help="SNPs per stratum")
    ap.add_argument('--ranking-out', required=True)
    ap.add_argument('--strata-dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.addone_dir, f'height_merged_{a.sex}_*_lasso_results.csv')))
    r = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    base = r[r['columns'] == 'basic_columns'].head(1)
    snps = r[r['columns'] != 'basic_columns'].drop_duplicates('columns')
    out = pd.concat([base, snps], ignore_index=True)
    out.insert(0, 'Feature', ['basic' if c == 'basic_columns' else c.split('_')[0]
                              for c in out['columns']])
    out = out[['Feature', 'columns', 'lasso_mse_test', 'lasso_mse_cv']]
    out.to_csv(a.ranking_out, index=False)

    ranked = out[out.Feature != 'basic'].sort_values('lasso_mse_test')['columns'].tolist()
    M = len(ranked)
    rng = np.random.default_rng(a.seed)
    mid = ranked[M // 3: 2 * M // 3]
    strata = {'Top': ranked[:a.n],
              'Middle': list(rng.choice(mid, a.n, replace=False)),
              'Last': ranked[-a.n:]}
    os.makedirs(a.strata_dir, exist_ok=True)
    S = a.sex.capitalize()
    for k, v in strata.items():
        pd.DataFrame({'SNP_List': v}).to_csv(
            os.path.join(a.strata_dir, f'New_Knee_Point_SNP_List_{S}_{k}{a.n}.csv'), index=False)
    print(f"{a.sex}: ranking of {M} SNPs -> {a.ranking_out}; strata of {a.n} SNPs -> {a.strata_dir}/")


if __name__ == '__main__':
    main()
