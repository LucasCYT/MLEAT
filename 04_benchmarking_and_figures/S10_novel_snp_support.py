#!/usr/bin/env python3
"""
Is there evidence that the NOVEL SNPs (Ref = 0) ranked highly by ML are real
signal rather than noise -- and how do they compare with the novel SNPs GWAS
ranks highly?

There is no ground truth for a novel locus, so four independent lines of
indirect support are measured for each set of novel SNPs:

  1. gene context   share located inside a gene already associated with height
                    in the GWAS Catalog (BHGWASList)         -- via TWB probe annotation
  2. proximity      distance to the nearest catalogued height SNP (hg38);
                    share within 50 kb / 250 kb              -- LD-neighbourhood proxy
  3. cross-sex      novel SNPs that make the top-N in BOTH the male and the
     replication    female analyses (independent cohorts), vs chance
  4. association    for ML's novel picks: their GWAS p-values, vs background

Sets compared (per sex): ML top-N novel, GWAS top-N novel, and all novel SNPs
as background.  Run from the 4B_BH_cmp folder:

    python3 novel_snp_support.py --db ../Top3000List/TWBV2_20240515.db -n 1000
"""

import argparse
import sqlite3
from bisect import bisect_left

import numpy as np
import pandas as pd
from scipy import stats

INSIDE = {"intron", "exon", "utr-3", "utr-5", "cds", "splice-site", "utr"}


def parse_inside(text):
    genes = set()
    if text in (None, "---", ""):
        return genes
    for rec in str(text).split("///"):
        p = [x.strip() for x in rec.split("//")]
        if len(p) < 6 or p[4] in ("---", ""):
            continue
        try:
            d = int(p[2])
        except ValueError:
            d = 0
        if p[1].lower() in INSIDE or d == 0:
            genes.add(p[4])
    return genes


def load_extract(ann_path, cat_path):
    """Same content as load_db, but from the tsv.gz extracts made on the Mac."""
    height_genes, cat = set(), {}
    for _, r in pd.read_csv(cat_path, sep="\t").iterrows():
        if pd.notna(r["pos"]) and pd.notna(r["chr"]):
            cat.setdefault(str(int(r["chr"])) if str(r["chr"]).replace(".0", "").isdigit()
                           else str(r["chr"]), []).append(int(r["pos"]))
        g = r["Genes"]
        if isinstance(g, str) and g not in ("-", "---", ""):
            height_genes.update(x for x in g.replace(" ", "").split(",")
                                if x and x not in ("-", "---"))
    cat = {k: np.array(sorted(v)) for k, v in cat.items()}
    a = pd.read_csv(ann_path, sep="\t", dtype=str).fillna("")
    ann = {r.Feature: (r.chr, int(float(r.pos_hg38)) if r.pos_hg38 else None,
                       set(x for x in r.genes_in.split(";") if x))
           for r in a.itertuples(index=False)}
    return height_genes, cat, ann


def load_db(db):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    height_genes = set()
    cat = {}
    for chr_, pos, g in con.execute("select chr, pos, Genes from BHGWASList"):
        cat.setdefault(str(chr_), []).append(int(pos))
        if g not in (None, "-", "---", ""):
            height_genes.update(x for x in str(g).replace(" ", "").split(",")
                                if x and x not in ("-", "---"))
    cat = {k: np.array(sorted(v)) for k, v in cat.items()}

    ann = {}
    for rs, ext, chr_, pos, a in con.execute(
            "select dbSNPRSID, ExtendedRSID, Chromosome, PhysicalPosition, AssociatedGene "
            "from TWBv2_Probeinfor"):
        key = rs if rs not in (None, "---", "") else ext
        if key in (None, "---", ""):
            continue
        genes = parse_inside(a)
        if key in ann:
            genes |= ann[key][2]
        ann[key] = (str(chr_), pos, genes)
    return height_genes, cat, ann


def nearest_catalog_distance(chr_, pos, cat):
    arr = cat.get(chr_)
    if arr is None or pos is None:
        return np.nan
    i = bisect_left(arr, pos)
    cands = []
    if i < len(arr):
        cands.append(abs(arr[i] - pos))
    if i > 0:
        cands.append(abs(arr[i - 1] - pos))
    return min(cands)


def describe(name, rs_list, ann, height_genes, cat, pvals=None):
    n = len(rs_list)
    in_gene = in_height = 0
    dists = []
    for rs in rs_list:
        chr_, pos, genes = ann.get(rs, (None, None, set()))
        if genes:
            in_gene += 1
            if genes & height_genes:
                in_height += 1
        dists.append(nearest_catalog_distance(chr_, pos, cat))
    d = np.array(dists, dtype=float)
    d = d[~np.isnan(d)]
    row = dict(set=name, n=n,
               inside_gene=in_gene, inside_gene_pct=in_gene / n * 100,
               inside_height_gene=in_height, inside_height_gene_pct=in_height / n * 100,
               median_dist_to_catalog_kb=np.median(d) / 1000,
               within_50kb_pct=(d <= 50_000).mean() * 100,
               within_250kb_pct=(d <= 250_000).mean() * 100)
    if pvals is not None:
        p = np.array([pvals.get(rs, np.nan) for rs in rs_list], dtype=float)
        p = p[~np.isnan(p)]
        row.update(gwas_p_median=np.median(p),
                   gwas_p_lt_1e3_pct=(p < 1e-3).mean() * 100,
                   gwas_p_lt_1e4_pct=(p < 1e-4).mean() * 100,
                   gwas_p_lt_1e5_pct=(p < 1e-5).mean() * 100)
    return row


def fisher(a, na, b, nb):
    return stats.fisher_exact([[a, na - a], [b, nb - b]], alternative="greater")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--male", default="MaleCompare.txt")
    ap.add_argument("--female", default="FemaleCompare.txt")
    ap.add_argument("--male-gwas", default="male_GWAS_BODY_HEIGHT_c29_80_ADD.assoc.linear")
    ap.add_argument("--female-gwas", default="female_GWAS_BODY_HEIGHT_c33_80_ADD.assoc.linear")
    ap.add_argument("--db", default="../Top3000List/TWBV2_20240515.db")
    ap.add_argument("--annotation", default=None,
                    help="universe_annotation.tsv.gz (use instead of --db)")
    ap.add_argument("--catalog", default=None, help="BHGWASList_pos.tsv.gz")
    ap.add_argument("-n", "--top", type=int, action="append", default=None)
    ap.add_argument("--bg-sample", type=int, default=50000,
                    help="random novel SNPs used as background (speed)")
    ap.add_argument("--invisible-p", type=float, default=1e-3,
                    help="GWAS p at or above which an ML pick counts as GWAS-invisible")
    ap.add_argument("-o", "--out", default="novel_snp_support.csv")
    args = ap.parse_args()
    tops = args.top or [1000, 3000]

    print("loading annotation ...")
    if args.annotation:
        height_genes, cat, ann = load_extract(args.annotation, args.catalog)
    else:
        height_genes, cat, ann = load_db(args.db)
    rng = np.random.default_rng(0)

    data, pv = {}, {}
    for sex, cmp_path, gwas_path in (("Male", args.male, args.male_gwas),
                                     ("Female", args.female, args.female_gwas)):
        data[sex] = pd.read_csv(cmp_path, sep="\t")
        sep = "\t" if gwas_path.endswith((".tsv", ".tsv.gz")) else r"\s+"
        g = pd.read_csv(gwas_path, sep=sep, usecols=["SNP", "P"])
        g["SNP"] = g["SNP"].astype(str).str.split("_").str[0]
        pv[sex] = dict(zip(g["SNP"], g["P"]))

    rows = []
    for N in tops:
        sets = {}
        for sex in ("Male", "Female"):
            d = data[sex]
            novel = d[d.Ref == 0]
            ml = novel[novel.ML_Rank <= N].Feature.tolist()
            gw = novel[novel.GWAS_Rank <= N].Feature.tolist()
            bg = novel.Feature.sample(min(args.bg_sample, len(novel)), random_state=0).tolist()
            sets[sex] = dict(ML=set(ml), GWAS=set(gw))
            # ML novel picks with no marginal GWAS signal at all ("GWAS-invisible")
            invisible = [rs for rs in ml if pv[sex].get(rs, 1.0) >= args.invisible_p]
            for lbl, lst in (("ML top novel", ml), ("GWAS top novel", gw),
                             ("ML top novel, GWAS p >= 1e-3", invisible), ("all novel (bg)", bg)):
                if not lst:
                    continue
                r = describe(lbl, lst, ann, height_genes, cat, pv[sex])
                r.update(sex=sex, top_N=N)
                rows.append(r)
            rInv = rows[-2]
            rBG0 = rows[-1]
            for key, lab in (("within_50kb_pct", "within 50 kb of a catalogued SNP"),
                             ("inside_height_gene_pct", "inside a catalogued height gene")):
                ka = int(round(rInv[key] * rInv["n"] / 100)); kc = int(round(rBG0[key] * rBG0["n"] / 100))
                orr, p = stats.fisher_exact([[ka, rInv["n"] - ka], [kc, rBG0["n"] - kc]], alternative="greater")
                rInv[f"{key}_OR_vs_bg"], rInv[f"{key}_p_vs_bg"] = orr, p
            print(f"\n   GWAS-invisible ML novel picks (p >= {args.invisible_p:g}): {rInv['n']} of {len(ml)} "
                  f"({rInv['n']/len(ml)*100:.1f}%) | within 50 kb {rInv['within_50kb_pct']:.1f}% "
                  f"(OR {rInv['within_50kb_pct_OR_vs_bg']:.2f}, p {rInv['within_50kb_pct_p_vs_bg']:.1e}) | "
                  f"inside height gene {rInv['inside_height_gene_pct']:.1f}% "
                  f"(OR {rInv['inside_height_gene_pct_OR_vs_bg']:.2f}, p {rInv['inside_height_gene_pct_p_vs_bg']:.1e})")

            # tests vs background and ML vs GWAS
            rML, rGW, rBG = rows[-4], rows[-3], rows[-1]
            print(f"\n== {sex}, top {N}: novel SNPs  ML n={rML['n']}  GWAS n={rGW['n']}  (background n={rBG['n']})")
            for key, lab in (("inside_height_gene", "inside a catalogued height gene"),
                             ("within_50kb_pct", "within 50 kb of a catalogued SNP"),
                             ("within_250kb_pct", "within 250 kb of a catalogued SNP"),
                             ("gwas_p_lt_1e4_pct", "GWAS p < 1e-4")):
                if key.endswith("_pct") or key.startswith("gwas"):
                    a, b, c = rML[key], rGW[key], rBG[key]
                    ka, kb, kc = [int(round(x * nn / 100)) for x, nn in ((a, rML["n"]), (b, rGW["n"]), (c, rBG["n"]))]
                else:
                    ka, kb, kc = rML[key], rGW[key], rBG[key]
                    a, b, c = ka / rML["n"] * 100, kb / rGW["n"] * 100, kc / rBG["n"] * 100
                _, p_ml_bg = fisher(ka, rML["n"], kc, rBG["n"])
                _, p_gw_bg = fisher(kb, rGW["n"], kc, rBG["n"])
                orr, p_ml_gw = stats.fisher_exact([[ka, rML["n"] - ka], [kb, rGW["n"] - kb]])
                print(f"   {lab:36s} ML {a:5.1f}%  GWAS {b:5.1f}%  bg {c:5.1f}%   "
                      f"ML>bg p={p_ml_bg:.1e}  GWAS>bg p={p_gw_bg:.1e}  ML vs GWAS OR={orr:.2f} p={p_ml_gw:.2g}")
            print(f"   {'median distance to catalogued SNP (kb)':36s} ML {rML['median_dist_to_catalog_kb']:5.0f}   "
                  f"GWAS {rGW['median_dist_to_catalog_kb']:5.0f}   bg {rBG['median_dist_to_catalog_kb']:5.0f}")
            print(f"   {'median GWAS p':36s} ML {rML['gwas_p_median']:.2e}  GWAS {rGW['gwas_p_median']:.2e}  bg {rBG['gwas_p_median']:.2e}")

        # cross-sex replication of novel picks
        U = len(data["Male"][data["Male"].Ref == 0])
        print(f"\n== cross-sex replication of novel SNPs, top {N} (universe of novel SNPs = {U:,})")
        for method in ("ML", "GWAS"):
            A, B = sets["Male"][method], sets["Female"][method]
            both = len(A & B)
            expected = len(A) * len(B) / U
            p = stats.hypergeom.sf(both - 1, U, len(A), len(B))
            print(f"   {method:5s} male {len(A):4d} x female {len(B):4d}: in both = {both:3d} "
                  f"(expected {expected:.1f}, x{both/expected:.0f}, p = {p:.2g})")
            rows.append(dict(set=f"{method} cross-sex overlap", sex="both", top_N=N,
                             n=both, expected=expected, fold=both / expected, p=p))
        a, b = len(sets["Male"]["ML"] & sets["Female"]["ML"]), len(sets["Male"]["GWAS"] & sets["Female"]["GWAS"])
        orr, p = stats.fisher_exact([[a, len(sets["Male"]["ML"]) - a], [b, len(sets["Male"]["GWAS"]) - b]])
        print(f"   ML vs GWAS replication rate: {a}/{len(sets['Male']['ML'])} vs {b}/{len(sets['Male']['GWAS'])}  OR={orr:.2f} p={p:.3g}")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
