#!/usr/bin/env python3
"""S09: weights file (SNP, A1, BETA) for PLINK --score from a .clumped index-SNP list and the training GWAS.
usage: python3 S09_make_weights_from_clumped.py <assoc.linear ADD file> <plink .clumped> <out weights.txt>"""
import sys, pandas as pd
assoc, clumped, out = sys.argv[1:4]
g = pd.read_csv(assoc, sep=r"\s+", usecols=["SNP", "A1", "BETA"])
c = pd.read_csv(clumped, sep=r"\s+", usecols=["SNP"])
w = g[g.SNP.isin(c.SNP)][["SNP", "A1", "BETA"]]
w.to_csv(out, sep="\t", index=False)
print(f"{out}: {len(w)} index SNPs (of {len(c)} clumps)")
