#!/usr/bin/env python3
"""
kneedle_prioritization.py
Define the ML-EAT prioritization set from the genome-wide add-one ranking.

Method (Kneedle maximum-distance criterion, Satopaa et al. 2011):
  1. Order the per-SNP held-out test MSE in ascending order (lowest MSE =
     largest add-one dMSE = strongest incremental predictive contribution).
  2. Represent the ordered values as a quantile-quantile plot against the
     standard normal distribution: x_i = Phi^-1( i / (n + 1) ), y_i = MSE_(i).
  3. Min-max normalise both axes to [0, 1].
  4. The knee is the point of greatest distance from the straight chord
     joining the first and last points: argmax |x_norm - y_norm| / sqrt(2).
  5. SNPs at or above the knee (lowest MSE) form the prioritization set.

Input  : --ranking CSV with columns [Feature (rsID), lasso_mse_test].
         A baseline row (Feature == 'basic') is ignored if present; it does
         not change the knee rank.
Output : <out_prefix>_prioritization_set.txt  (rsIDs, one per line)
         prints the knee rank (= prioritization-set size).

Reproduces the published prioritization-set sizes on the 2026-07-02 rankings:
  female  knee rank = 1048
  male    knee rank =  378

Usage:
  python kneedle_prioritization.py --ranking female_cv_5_lasso_aligned_merged_20260702.csv \
                                   --out-prefix female
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm


def kneedle_knee_rank(mse_values):
    """Return the 1-based knee rank on the ordered MSE distribution.

    The knee is located with the Kneedle maximum-distance criterion on the
    normal quantile-quantile representation of the ordered MSE values.
    """
    y = np.sort(np.asarray(mse_values, dtype=float))          # ascending (lowest MSE first)
    n = len(y)
    x = norm.ppf(np.arange(1, n + 1) / (n + 1))               # theoretical normal quantiles
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    distance = np.abs(x_norm - y_norm) / np.sqrt(2.0)         # distance from first-to-last chord
    return int(np.argmax(distance)) + 1                       # 1-based rank


def main():
    ap = argparse.ArgumentParser(description="ML-EAT Kneedle prioritization set")
    ap.add_argument("--ranking", required=True,
                    help="CSV with the per-SNP ranking [Feature, lasso_mse_test]")
    ap.add_argument("--id-col", default="Feature")
    ap.add_argument("--mse-col", default="lasso_mse_test")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.ranking)
    df = df[df[a.id_col].astype(str) != "basic"].dropna(subset=[a.mse_col]).copy()

    k = kneedle_knee_rank(df[a.mse_col].values)

    # prioritization set = the k SNPs with the lowest MSE (largest add-one dMSE)
    prioritization = df.nsmallest(k, a.mse_col)[[a.id_col]]
    out = f"{a.out_prefix}_prioritization_set.txt"
    prioritization.to_csv(out, index=False, header=False)

    print(f"knee rank (prioritization-set size) = {k}")
    print(f"wrote {out} ({k} variants)")


if __name__ == "__main__":
    main()
