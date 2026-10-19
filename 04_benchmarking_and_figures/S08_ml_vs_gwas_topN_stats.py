#!/usr/bin/env python3
"""
Top-N comparison of the add-one (ML) ranking with the GWAS ranking.

For each sex and each cut-off N it reports
  * overlap of the two top-N lists
  * famous SNPs (Ref > 5): counts in each top-N and Fisher's exact test,
    the 2x2 quadrant split at N (both / ML-only / GWAS-only / neither),
    and how many fSNPs visible in the top-3000 union plot fall in the
    lower-left (both <= 1000) quadrant
  * known (Ref > 0) and unreported (Ref = 0) loci in each top-N, Fisher's exact
  * the ML-only and GWAS-only regions

Input: MaleCompare.txt / FemaleCompare.txt (Feature, ML_Rank, GWAS_Rank, Ref).

    python3 ml_vs_gwas_topN_stats.py            # N = 100 ... 3000
    python3 ml_vs_gwas_topN_stats.py -n 1000 -n 3000
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

FSNP = 5          # "famous SNP" = more than this many GWAS Catalog references


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--male", default="MaleCompare.txt")
    ap.add_argument("--female", default="FemaleCompare.txt")
    ap.add_argument("-n", "--top", type=int, action="append", default=None)
    ap.add_argument("--view", type=int, default=3000, help="plot window used in Fig 4B")
    ap.add_argument("-o", "--out", default="ML_vs_GWAS_topN_ref_stats.csv")
    args = ap.parse_args()
    tops = args.top or [100, 200, 500, 1000, 2000, 3000]

    rows = []
    for sex, path in (("Male", args.male), ("Female", args.female)):
        d = pd.read_csv(path, sep="\t")
        f = d[d.Ref > FSNP]
        print(f"\n==== {sex}: {len(d):,} SNPs, {int((d.Ref > 0).sum()):,} with Ref > 0, "
              f"{len(f)} famous SNPs (Ref > {FSNP})")

        # fSNPs inside the Fig 4B window and in its lower-left quadrant
        vis = f[(f.ML_Rank <= args.view) | (f.GWAS_Rank <= args.view)]
        ll = vis[(vis.ML_Rank <= 1000) & (vis.GWAS_Rank <= 1000)]
        print(f"   fSNPs in the top-{args.view} union view: {len(vis)}; "
              f"both ranks <= 1000: {len(ll)}")

        for N in tops:
            ml, gw = d[d.ML_Rank <= N], d[d.GWAS_Rank <= N]
            both = int(((d.ML_Rank <= N) & (d.GWAS_Rank <= N)).sum())
            r = dict(sex=sex, top_N=N, overlap=both, overlap_pct=both / N * 100,
                     fsnp_in_view=len(vis), fsnp_lower_left_1000=len(ll))
            for lbl, cond in (("fsnp", lambda x: x.Ref > FSNP),
                              ("ref_gt0", lambda x: x.Ref > 0),
                              ("novel", lambda x: x.Ref == 0)):
                a, b = int(cond(ml).sum()), int(cond(gw).sum())
                orr, p = fisher_exact([[a, N - a], [b, N - b]])
                r.update({f"{lbl}_ML": a, f"{lbl}_GWAS": b,
                          f"{lbl}_OR": orr, f"{lbl}_fisher_p": p})
            fq = dict(both=int(((f.ML_Rank <= N) & (f.GWAS_Rank <= N)).sum()),
                      ML_only=int(((f.ML_Rank <= N) & (f.GWAS_Rank > N)).sum()),
                      GWAS_only=int(((f.ML_Rank > N) & (f.GWAS_Rank <= N)).sum()))
            fq["neither"] = len(f) - sum(fq.values())
            r.update({f"fsnp_quadrant_{k}": v for k, v in fq.items()})
            r["fsnp_GWAS_top1000_but_ML_beyond_view"] = int(
                ((f.GWAS_Rank <= 1000) & (f.ML_Rank > args.view)).sum())
            r["fsnp_ML_top1000_but_GWAS_beyond_view"] = int(
                ((f.ML_Rank <= 1000) & (f.GWAS_Rank > args.view)).sum())
            r["sumRef_ML"], r["sumRef_GWAS"] = int(ml.Ref.sum()), int(gw.Ref.sum())
            rows.append(r)
            print(f"   top {N:>5}: overlap {both:4d} ({both/N*100:4.1f}%) | fSNP ML {r['fsnp_ML']:2d} "
                  f"GWAS {r['fsnp_GWAS']:2d} p={r['fsnp_fisher_p']:.2f} | novel ML {r['novel_ML']:4d} "
                  f"({r['novel_ML']/N*100:.1f}%) GWAS {r['novel_GWAS']:4d} ({r['novel_GWAS']/N*100:.1f}%) "
                  f"p={r['novel_fisher_p']:.2g} | fSNP quadrants both/MLonly/GWASonly/neither "
                  f"{fq['both']}/{fq['ML_only']}/{fq['GWAS_only']}/{fq['neither']}")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
