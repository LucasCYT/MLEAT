#!/usr/bin/env python3
"""
Catalogue-independent evidence for the loci ML ranks highly in regions the
GWAS Catalog has never touched.

Every "support" metric derived from the GWAS Catalog is biased towards what
GWAS already found. This script instead asks: for the ML top-N SNPs that sit
in UNEXPLORED regions -- no catalogued height SNP within a window, and not in
a catalogued height gene -- is there evidence that does NOT come from GWAS?

Evidence lines (none of them use the catalogue):
  1. add-one effect size   Delta-MSE of the SNP in the cross-validated Lasso
                           model (baseline MSE minus MSE with the SNP added),
                           compared with the catalogued SNPs in the same top-N
  2. second Lasso split    rank of the SNP in an independent split-reference
                           Lasso run (same model family, different data split)
  3. XGBoost               rank of the SNP in the XGBoost add-one run (different
                           model family, non-linear)
  4. cross-sex replication SNP in the unexplored top-N of BOTH sexes (independent
                           cohorts), vs the hypergeometric expectation
  5. marginal GWAS p       reported only to show the SNPs are invisible to
                           single-SNP testing (not as support)

Also written: a candidate table of unexplored ML loci with all of the above,
for the supplementary material.

    python3 unexplored_loci.py -n 1000 -n 3000 --window 500000
"""

import argparse
from bisect import bisect_left

import numpy as np
import pandas as pd
from scipy import stats


def load_catalog(path):
    cat = pd.read_csv(path, sep="\t")
    genes = set()
    for g in cat.Genes.dropna():
        if g not in ("-", "---"):
            genes.update(x for x in g.replace(" ", "").split(",") if x and x not in ("-", "---"))
    cat = cat.dropna(subset=["chr", "pos"])
    pos = {}
    for c, v in cat.groupby("chr")["pos"]:
        key = str(int(c)) if str(c).replace(".0", "").isdigit() else str(c)
        pos[key] = np.array(sorted(v))
    return genes, pos


def nearest(chr_, p, catpos):
    a = catpos.get(chr_)
    if a is None or p is None:
        return np.nan
    i = bisect_left(a, p)
    c = []
    if i < len(a):
        c.append(abs(a[i] - p))
    if i > 0:
        c.append(abs(a[i - 1] - p))
    return min(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="folder with the extracted tsv.gz files")
    ap.add_argument("--cmp-dir", default=".", help="folder with MaleCompare.txt / FemaleCompare.txt")
    ap.add_argument("-n", "--top", type=int, action="append", default=None)
    ap.add_argument("--window", type=int, default=500_000,
                    help="'unexplored' = no catalogued SNP within this many bp")
    ap.add_argument("-o", "--out", default="unexplored_loci_summary.csv")
    args = ap.parse_args()
    tops = args.top or [1000, 3000]
    D = args.dir

    height_genes, catpos = load_catalog(f"{D}/BHGWASList_pos.tsv.gz")
    ann = pd.read_csv(f"{D}/universe_annotation.tsv.gz", sep="\t", dtype=str).fillna("")
    ann["pos"] = pd.to_numeric(ann.pos_hg38, errors="coerce")
    ann["in_height_gene"] = ann.genes_in.apply(
        lambda s: bool(set(x for x in s.split(";") if x) & height_genes))
    ann["dist_catalog"] = [nearest(c, (int(p) if pd.notna(p) else None), catpos)
                           for c, p in zip(ann.chr, ann.pos)]

    per_sex, summary, cand = {}, [], []
    for sex in ("Male", "Female"):
        d = pd.read_csv(f"{args.cmp_dir}/{sex}Compare.txt", sep="\t")
        d = d.merge(ann[["Feature", "chr", "pos", "genes_in", "in_height_gene", "dist_catalog"]],
                    on="Feature", how="left")
        sc = pd.read_csv(f"{D}/{sex.lower()}_model_scores.tsv.gz", sep="\t")
        base = sc.loc[sc.Feature == "basic", "Lasso_MSE"].mean()
        sc = sc[sc.Feature != "basic"].drop_duplicates("Feature")
        sc["dMSE"] = base - sc.Lasso_MSE                      # positive = SNP helps
        sc["rank_split"] = sc.splitRef_lasso.rank(method="min")
        sc["rank_xgb"] = sc.XGBoots_mse.rank(method="min")
        d = d.merge(sc[["Feature", "dMSE", "rank_split", "rank_xgb"]], on="Feature", how="left")
        p = pd.read_csv(f"{D}/{sex.lower()}_gwas_p.tsv.gz", sep="\t")
        p["SNP"] = p.SNP.astype(str).str.split("_").str[0]
        d = d.merge(p.rename(columns={"SNP": "Feature", "P": "gwas_p"}), on="Feature", how="left")

        d["unexplored"] = (d.Ref == 0) & (~d.in_height_gene.fillna(False)) & (d.dist_catalog > args.window)
        n_all = len(d)
        n_unexp = int(d.unexplored.sum())
        per_sex[sex] = d
        print(f"\n==== {sex}: {n_unexp:,} of {n_all:,} SNPs ({n_unexp/n_all*100:.1f}%) lie in unexplored "
              f"regions (no catalogued SNP within {args.window/1000:.0f} kb, not in a height gene)")

        for N in tops:
            top_ml = d[d.ML_Rank <= N]
            top_gw = d[d.GWAS_Rank <= N]
            u_ml, u_gw = top_ml[top_ml.unexplored], top_gw[top_gw.unexplored]
            known_ml = top_ml[top_ml.Ref > 0]
            rest_unexp = d[d.unexplored & (d.ML_Rank > N)]

            # 1. effect size: unexplored vs catalogued SNPs in the same ML top-N
            mw_p = stats.mannwhitneyu(u_ml.dMSE.dropna(), known_ml.dMSE.dropna()).pvalue if len(u_ml) else np.nan
            # 2/3. replication in other model runs: how many of the unexplored ML
            #      top-N are also in the top-N of the split Lasso / XGBoost runs
            rep_split = int((u_ml.rank_split <= N).sum())
            rep_xgb = int((u_ml.rank_xgb <= N).sum())
            exp_split = len(u_ml) * N / n_all
            p_split = stats.binom.sf(rep_split - 1, len(u_ml), N / n_all) if len(u_ml) else np.nan
            p_xgb = stats.binom.sf(rep_xgb - 1, len(u_ml), N / n_all) if len(u_ml) else np.nan
            # same for the catalogued SNPs in the top-N, as a yardstick
            rep_xgb_known = int((known_ml.rank_xgb <= N).sum())
            # 5. GWAS visibility
            invisible = float((u_ml.gwas_p >= 1e-3).mean()) * 100 if len(u_ml) else np.nan

            row = dict(sex=sex, top_N=N, window_kb=args.window / 1000,
                       n_unexplored_genome=n_unexp, pct_unexplored_genome=n_unexp / n_all * 100,
                       ML_unexplored=len(u_ml), ML_unexplored_pct=len(u_ml) / N * 100,
                       GWAS_unexplored=len(u_gw), GWAS_unexplored_pct=len(u_gw) / N * 100,
                       ML_known=len(known_ml),
                       dMSE_median_unexplored=u_ml.dMSE.median(),
                       dMSE_median_known=known_ml.dMSE.median(),
                       dMSE_median_rest_unexplored=rest_unexp.dMSE.median(),
                       mannwhitney_p_unexp_vs_known=mw_p,
                       split_lasso_replicated=rep_split, split_lasso_expected=exp_split, split_lasso_p=p_split,
                       xgb_replicated=rep_xgb, xgb_expected=exp_split, xgb_p=p_xgb,
                       xgb_replicated_known=rep_xgb_known, xgb_known_pct=rep_xgb_known / max(len(known_ml), 1) * 100,
                       gwas_invisible_pct=invisible)
            summary.append(row)

            print(f"\n  top {N}:  unexplored loci  ML {len(u_ml)} ({len(u_ml)/N*100:.1f}%)   "
                  f"GWAS {len(u_gw)} ({len(u_gw)/N*100:.1f}%)   genome-wide {n_unexp/n_all*100:.1f}%")
            print(f"     add-one dMSE median   unexplored {u_ml.dMSE.median():.4f}   catalogued {known_ml.dMSE.median():.4f}   "
                  f"(Mann-Whitney p = {mw_p:.2g});  unexplored SNPs outside top-N: {rest_unexp.dMSE.median():.4f}")
            print(f"     split-Lasso top-{N}    {rep_split}/{len(u_ml)} ({rep_split/max(len(u_ml),1)*100:.0f}%)  expected {exp_split:.1f}  p = {p_split:.1e}")
            print(f"     XGBoost   top-{N}      {rep_xgb}/{len(u_ml)} ({rep_xgb/max(len(u_ml),1)*100:.0f}%)  expected {exp_split:.1f}  p = {p_xgb:.1e}"
                  f"   [catalogued SNPs: {rep_xgb_known}/{len(known_ml)} = {rep_xgb_known/max(len(known_ml),1)*100:.0f}%]")
            print(f"     invisible to GWAS (p >= 1e-3): {invisible:.0f}%")

            u = u_ml.copy()
            u.insert(0, "sex", sex)
            u.insert(1, "top_N", N)
            cand.append(u)

    # 4. cross-sex replication of unexplored loci
    print("\n==== cross-sex replication of unexplored loci (independent cohorts)")
    U = int((per_sex["Male"].unexplored & per_sex["Female"].unexplored).sum())   # loci unexplored in both
    for N in tops:
        for method, col in (("ML", "ML_Rank"), ("GWAS", "GWAS_Rank")):
            A = set(per_sex["Male"][per_sex["Male"].unexplored & (per_sex["Male"][col] <= N)].Feature)
            B = set(per_sex["Female"][per_sex["Female"].unexplored & (per_sex["Female"][col] <= N)].Feature)
            both = A & B
            expct = len(A) * len(B) / U
            pval = stats.hypergeom.sf(len(both) - 1, U, len(A), len(B))
            print(f"  top {N} {method:4s}: male {len(A):4d} x female {len(B):4d} -> in both {len(both):3d} "
                  f"(expected {expct:.2f}, p = {pval:.2g})  {sorted(both)[:12]}")
            summary.append(dict(sex="both", top_N=N, window_kb=args.window / 1000,
                                set=f"{method} cross-sex", male=len(A), female=len(B),
                                in_both=len(both), expected=expct, p=pval,
                                shared=";".join(sorted(both))))

    pd.DataFrame(summary).to_csv(args.out, index=False)
    cols = ["sex", "top_N", "Feature", "chr", "pos", "ML_Rank", "GWAS_Rank", "gwas_p",
            "dMSE", "rank_split", "rank_xgb", "dist_catalog", "genes_in"]
    c = pd.concat(cand)[cols].rename(columns={"pos": "pos_hg38", "dist_catalog": "dist_to_catalog_bp",
                                              "genes_in": "gene", "rank_split": "rank_split_lasso"})
    c = c.sort_values(["sex", "top_N", "ML_Rank"])
    c.to_csv(args.out.replace("_summary.csv", "_candidates.csv"), index=False)
    print(f"\nsaved: {args.out} and {args.out.replace('_summary.csv', '_candidates.csv')}")


if __name__ == "__main__":
    main()
