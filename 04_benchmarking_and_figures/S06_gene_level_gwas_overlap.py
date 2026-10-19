#!/usr/bin/env python3
"""
Gene-level overlap of the ML top-N SNPs with GWAS Catalog height genes.

Reproduces the gene-level numbers in the Results:
  * SNPs of the top-N list that fall inside a gene body (TWB array annotation)
  * unique genes containing at least one top-N SNP
  * how many of those genes are height genes in the GWAS Catalog (BHGWASList)
  * background: share of ALL genes named in the array annotation that are
    height genes, and Fisher's exact test top-N genes vs background

Inputs (all local): TWBV2_20240515.db (tables BHGWASList, TWBv2_Probeinfor)
and the per-SNP gene tables made by map_genes_from_twb_db.py
(ML_male_top3000_genes.csv / ML_female_top3000_genes.csv).

    python3 gene_level_gwas_overlap.py --db TWBV2_20240515.db
"""

import argparse
import math
import sqlite3

import pandas as pd

try:
    from scipy.stats import fisher_exact
except ImportError:                                   # exact test without scipy
    def fisher_exact(table, alternative="two-sided"):
        (a, b), (c, d) = table
        n = a + b + c + d
        r1, c1 = a + b, a + c

        def logp(x):
            return (math.lgamma(r1 + 1) - math.lgamma(x + 1) - math.lgamma(r1 - x + 1)
                    + math.lgamma(n - r1 + 1) - math.lgamma(c1 - x + 1)
                    - math.lgamma(n - r1 - c1 + x + 1)
                    - (math.lgamma(n + 1) - math.lgamma(c1 + 1) - math.lgamma(n - c1 + 1)))
        lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
        p_obs = logp(a)
        if alternative == "greater":
            p = sum(math.exp(logp(x)) for x in range(a, hi + 1))
        else:
            p = sum(math.exp(logp(x)) for x in range(lo, hi + 1) if logp(x) <= p_obs + 1e-9)
        odds = (a * d) / (b * c) if b * c else float("inf")
        return odds, min(p, 1.0)


def height_gene_set(con):
    genes = set()
    for (g,) in con.execute("select Genes from BHGWASList"):
        if g in (None, "-", "---", ""):
            continue
        genes.update(x for x in str(g).replace(" ", "").split(",") if x and x not in ("-", "---"))
    return genes


def array_gene_set(con):
    """Every gene symbol mentioned anywhere in the array annotation."""
    genes = set()
    for (a,) in con.execute("select AssociatedGene from TWBv2_Probeinfor"):
        if a in (None, "---", ""):
            continue
        for rec in str(a).split("///"):
            p = [x.strip() for x in rec.split("//")]
            if len(p) >= 6 and p[4] not in ("---", ""):
                genes.add(p[4])
    return genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="TWBV2_20240515.db")
    ap.add_argument("--male", default="ML_male_top3000_genes.csv")
    ap.add_argument("--female", default="ML_female_top3000_genes.csv")
    ap.add_argument("-o", "--out", default="gene_level_gwas_overlap.csv")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    hg = height_gene_set(con)
    allg = array_gene_set(con)
    bg_hit = len(allg & hg)
    print(f"GWAS Catalog height genes: {len(hg):,}")
    print(f"genes named in the array annotation: {len(allg):,}; "
          f"{bg_hit:,} ({bg_hit/len(allg)*100:.1f}%) are height genes  <- background\n")

    rows = []
    for sex, path in (("Male", args.male), ("Female", args.female)):
        t = pd.read_csv(path).fillna("")
        n = len(t)
        inside = t[t.genes_in != ""]
        near = t[(t.genes_in == "") & (t.nearest_gene != "")]
        genes = {g for s in inside.genes_in for g in s.split(";") if g}
        hit = genes & hg
        # background excludes the genes under test
        bg_rest_hit = bg_hit - len(hit)
        bg_rest_n = len(allg) - len(genes)
        orr, p = fisher_exact([[len(hit), len(genes) - len(hit)],
                               [bg_rest_hit, bg_rest_n - bg_rest_hit]], alternative="greater")
        snp_hit = int(inside.genes_in.apply(lambda s: any(g in hg for g in s.split(";"))).sum())
        rows.append(dict(sex=sex, n_snps=n, snps_inside_gene=len(inside),
                         snps_with_nearest_gene_only=len(near),
                         pct_remaining_assigned_nearest=len(near) / max(n - len(inside), 1) * 100,
                         snps_inside_height_gene=snp_hit,
                         genes=len(genes), height_genes=len(hit), pct=len(hit) / len(genes) * 100,
                         background_pct=bg_rest_hit / bg_rest_n * 100, OR=orr, fisher_p=p))
        print(f"== {sex}: top {n:,} SNPs")
        print(f"   inside a gene body: {len(inside):,}; nearest-gene assigned: {len(near):,} "
              f"({len(near)/max(n-len(inside),1)*100:.0f}% of the rest)")
        print(f"   unique genes: {len(genes):,}; height genes: {len(hit):,} ({len(hit)/len(genes)*100:.1f}%)"
              f"  vs background {bg_rest_hit/bg_rest_n*100:.1f}%   OR = {orr:.2f}, p = {p:.2g}")
        print(f"   SNPs inside a height gene: {snp_hit:,} ({snp_hit/len(inside)*100:.1f}% of in-gene SNPs)\n")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
