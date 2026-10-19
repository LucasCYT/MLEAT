#!/usr/bin/env python3
"""
S04_merge_permutation_null.py
Merge the per-SNP permutation files written by S03 into one wide table per
stratum and sex - the "genotype-permutation null files" read by
01_prioritization_and_fdr/mleat_downstream_analysis.py (--perm-null) and
novelty_screen_FDRset_20260731.py.

Input   --perm-dir   results_.../permutation_{sex}/  (one {SNP}.csv per SNP)
Output  --out        CSV with columns permutation_id, mse_{SNP}, mse_{SNP}, ...
                     (held-out test MSE; permutation_id 0 = observed, > 0 = null).
                     The pooled null used for the FDR excludes permutation_id 0.

Usage (female top stratum):
  python S04_merge_permutation_null.py \
      --perm-dir results_filter_v4_NoFilter_SnpTrainShuff/permutation_female \
      --out      results_filter_v4_NoFilter_SnpTrainShuff/Top1000_female_snps_merged.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Merge per-SNP permutation results")
    ap.add_argument('--perm-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    cols = []
    for f in sorted(Path(a.perm_dir).glob('*.csv')):
        if f.name.lower() == 'summary.csv':
            continue
        d = pd.read_csv(f, usecols=['permutation_id', 'mse'], index_col='permutation_id')
        cols.append(d.rename(columns={'mse': f"mse_{f.stem}"}))
    merged = pd.concat(cols, axis=1, join='outer').reset_index()
    merged.to_csv(a.out, index=False)
    n_null = int((merged['permutation_id'] > 0).sum()) * (merged.shape[1] - 1)
    print(f"wrote {a.out}: {merged.shape[1] - 1} SNPs x {len(merged)} rows "
          f"({n_null} null values, permutation_id > 0)")


if __name__ == '__main__':
    main()
