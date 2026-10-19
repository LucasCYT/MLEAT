#!/usr/bin/env python3
"""
How many of the top-N ML_MSE ranked loci carry at least one known reference?

For each ranked list it reports, at several cut-offs, the share of loci with
RefCount > 0, the fold enrichment over the whole-list background, and a
one-sided hypergeometric p-value for that enrichment.

    python refcount_enrichment_stats.py                       # both lists
    python refcount_enrichment_stats.py -i other_list.csv -n 3000

Writes refcount_enrichment_stats.csv next to the outputs.
"""

import argparse
import os

import pandas as pd

try:
    from scipy.stats import hypergeom
except ImportError:                       # p-values are optional
    hypergeom = None

DEFAULT_INPUTS = ["Male_50wAddone_rankedList_20260903.csv",
                  "Female_50wAddone_rankedList_20260903.csv"]
DEFAULT_CUTOFFS = [100, 500, 1000, 3000, 5000, 10000]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", action="append", default=None,
                   help="ranked CSV; repeat for several (default: male + female)")
    p.add_argument("-n", "--cutoff", action="append", type=int, default=None,
                   help="top-N cut-off; repeat for several")
    p.add_argument("-c", "--column", default="RefCount", help="reference count column")
    p.add_argument("-o", "--out", default="refcount_enrichment_stats.csv")
    args = p.parse_args()

    inputs = args.input or DEFAULT_INPUTS
    cutoffs = sorted(args.cutoff or DEFAULT_CUTOFFS)

    rows = []
    for path in inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        y = pd.read_csv(path, usecols=[args.column])[args.column].fillna(0).to_numpy()
        N, K = len(y), int((y > 0).sum())          # list size, loci with any ref
        background = K / N
        print(f"\n== {name}")
        print(f"   whole list: {K:,}/{N:,} loci with {args.column} > 0 "
              f"({background * 100:.3f}%)")
        for n in cutoffs:
            if n > N:
                continue
            k = int((y[:n] > 0).sum())
            share = k / n
            pval = (hypergeom.sf(k - 1, N, K, n) if hypergeom else float("nan"))
            print(f"   top {n:>6,}: {k:>5,} ({share * 100:6.2f}%)  "
                  f"enrichment x{share / background:5.1f}"
                  + (f"  p = {pval:.3g}" if hypergeom else ""))
            rows.append(dict(list=name, cutoff=n, with_ref=k, share_pct=share * 100,
                             list_size=N, background_with_ref=K,
                             background_pct=background * 100,
                             fold_enrichment=share / background, p_value=pval))

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
