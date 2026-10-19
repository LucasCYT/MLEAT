#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-EAT downstream analysis (all steps from per-SNP test MSE onward).

Given the genome-wide add-one test-MSE ranking and the genotype-permutation
null, this module computes:
  (1) empirical false discovery rate (FDR) and the FDR-controlled set;
  (2) rank-dependent enrichment for established (Yengo 2022) height loci;
  (3) a three-tier novelty assessment (direct GWAS lookup, LD proxy, region scan).

Upstream (folder 00_baseline_addone_permutation/): the baseline LASSO model,
the add-one predictive-contribution screen, and the genotype permutation that
produce the per-SNP test-MSE files consumed below.

Author: <...>   |   License: MIT   |   Dependencies: numpy, pandas, scipy

Inputs
------
--ranking     CSV with columns [Feature (rsID), lasso_mse_test]; one row per
              screened SNP (plus a 'basic' row = baseline). Sorted or unsorted.
--perm-null   One or more per-SNP permutation files (long format) with columns
              [permutation_id, mse]; permutation_id==0 is the real (unpermuted)
              value, >0 are the genotype-permuted null values. Pool across the
              sampled SNPs / strata to estimate the null.
--yengo       Reference height GWAS summary statistics (tab-separated, gzip),
              with columns [RSID, P]; e.g. Yengo et al. 2022 (GIANT).
--geno        (optional) Genotype dosage matrix (samples x SNPs, tab-separated;
              columns 'rsID_allele', values in {0,1,2}) used to compute LD (r^2)
              for the proxy step. Autosomal LD is population-level; an
              ancestry-matched panel of either sex may be used.
--M           Number of variants screened genome-wide (default 510274).
--alpha       Target FDR (default 0.05).

Outputs (written next to --out-prefix)
-------
  *_fdr_curve.csv          FDR as a function of selection size
  *_highconf.csv           the FDR<=alpha high-confidence set (annotated)
  *_enrichment.csv         quartile enrichment for known loci + trend test
"""

import argparse, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

GW_SIG = 5e-8   # genome-wide significance threshold used to define "established"


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------
def load_ranking(path):
    """Return SNPs sorted by ascending test MSE (best predictive contribution first)."""
    r = pd.read_csv(path)
    r = r[r["Feature"] != "basic"].copy()
    r = r.sort_values("lasso_mse_test").reset_index(drop=True)
    r["rank"] = np.arange(1, len(r) + 1)
    return r


def pooled_null(paths):
    """Pool the genotype-permutation null (permutation_id != 0) across files."""
    vals = []
    for p in paths:
        d = pd.read_csv(p)
        mcols = [c for c in d.columns if c == "mse" or c.startswith("mse_")]
        a = d.loc[d["permutation_id"] != 0, mcols].to_numpy(dtype=float).ravel()
        vals.append(a[~np.isnan(a)])
    return np.sort(np.concatenate(vals))


def load_yengo(path):
    y = pd.read_csv(path, sep="\t", usecols=["RSID", "P"])
    y["P"] = pd.to_numeric(y["P"], errors="coerce")
    return dict(zip(y["RSID"], y["P"]))


# ----------------------------------------------------------------------
# (1) Empirical FDR  (SAM / Storey-style, on a statistic with no parametric null)
# ----------------------------------------------------------------------
def empirical_fdr(ranking, null_sorted, M, alpha=0.05, grid=None):
    """FDR(t) = M * p0(t) / R(t), where p0(t) = P(null MSE <= t) and
    R(t) = number of real SNPs with MSE <= t. Lower MSE = stronger contribution."""
    real = ranking["lasso_mse_test"].to_numpy(float)
    n = len(null_sorted)
    p0 = lambda t: np.searchsorted(null_sorted, t, side="right") / n

    if grid is None:
        grid = sorted(set(np.r_[np.arange(1, 201), np.arange(200, 5001, 50)]))
    rows = []
    for R in grid:
        if R > len(real):
            break
        t = real[R - 1]
        efp = M * p0(t)
        rows.append({"R": R, "threshold_mse": t, "p0": p0(t),
                     "expected_false": efp, "FDR": min(1.0, efp / R)})
    curve = pd.DataFrame(rows)

    # largest R with FDR <= alpha
    best = 0
    for R in range(1, min(len(real), 20000)):
        if M * p0(real[R - 1]) / R <= alpha:
            best = R
        else:
            break
    hc = ranking.head(best).copy()
    hc["FDR"] = [min(1.0, M * p0(t) / rk) for t, rk in zip(hc["lasso_mse_test"], hc["rank"])]
    return curve, hc


# ----------------------------------------------------------------------
# (2) Rank-dependent enrichment for established height loci
# ----------------------------------------------------------------------
def enrichment(ranking, yengo, knee, n_quartile_bins=4):
    """Fraction of variants per ranking quartile that are genome-wide significant
    in the reference GWAS, and a Cochran-Armitage trend test across quartiles.

    The test statistic is asymptotically normal; we report the two-sided trend
    P-value (not a z-score) to avoid any confusion with the parametric
    MSE-to-z transformation that this framework deliberately replaces. Run this
    once per reference (e.g. all-ancestry and East-Asian-specific summary stats)
    as an ancestry sensitivity analysis."""
    ranking = ranking.copy()
    ranking["known"] = ranking["Feature"].map(lambda s: yengo.get(s, np.nan)) < GW_SIG
    bg = ranking["known"].mean()
    q = knee // n_quartile_bins
    rows, counts = [], []
    for i in range(n_quartile_bins):
        a, b = 1 + i * q, (i + 1) * q
        s = ranking[(ranking["rank"] >= a) & (ranking["rank"] <= b)]["known"]
        rows.append({"quartile": i + 1, "rank_from": a, "rank_to": b,
                     "n": len(s), "frac_known": s.mean(),
                     "fold_over_background": (s.mean() / bg if bg else np.nan)})
        counts.append([int(s.sum()), int(len(s) - s.sum())])
    counts = np.array(counts)
    scores = np.arange(n_quartile_bins)
    N = counts.sum(); Nk = counts.sum(1); r1 = counts[:, 0]; Rt = r1.sum()
    T = np.sum(scores * (r1 - Nk * Rt / N)); pbar = Rt / N
    var = pbar * (1 - pbar) * (np.sum(Nk * scores**2) - (np.sum(Nk * scores))**2 / N)
    z = T / np.sqrt(var) if var > 0 else np.nan
    trend_p = 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    df = pd.DataFrame(rows)
    df.attrs["cochran_armitage_trend_pvalue"] = trend_p
    df.attrs["background_frac_known"] = bg
    return df, trend_p, bg


# ----------------------------------------------------------------------
# (3) Novelty assessment: (i) direct GWAS lookup, (ii) LD proxy, (iii) region scan
# ----------------------------------------------------------------------
def _r2_to_anchors(cand, anchors, geno_path):
    """max r^2 of each candidate SNP against a set of anchor SNPs, from a
    genotype dosage matrix. Returns {rsID: max_r2}. Missing dosages mean-imputed."""
    with open(geno_path, encoding="utf-8", errors="ignore") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    col = {c.split("_")[0]: c for c in header if c.startswith("rs")}
    cand_in = [s for s in cand if s in col]
    anch_in = [s for s in anchors if s in col]
    if not cand_in or not anch_in:
        return {}
    d = pd.read_csv(geno_path, sep="\t", usecols=[col[s] for s in cand_in + anch_in])

    def z(names):
        A = np.empty((len(d), len(names)), dtype=np.float32)
        for j, s in enumerate(names):
            v = pd.to_numeric(d[col[s]], errors="coerce").to_numpy(np.float32)
            v[~np.isin(v, [0, 1, 2])] = np.nan
            v[np.isnan(v)] = np.nanmean(v)
            cc = v - v.mean(); sd = cc.std(); A[:, j] = cc / (sd if sd > 0 else 1)
        return A

    Zc, Zs, n = z(cand_in), z(anch_in), len(d)
    mx = np.zeros(len(cand_in))
    for i in range(0, Zs.shape[1], 400):
        mx = np.maximum(mx, (((Zc.T @ Zs[:, i:i + 400]) / n) ** 2).max(axis=1))
    return dict(zip(cand_in, mx))


def novelty(highconf, yengo, geno_path=None, r2_thresh=0.2):
    """Annotate the high-confidence set as established (direct), established (LD proxy),
    or candidate-for-regional-scan. Regional (+-500 kb) scan is left to the caller
    (requires the reference GWAS with positions)."""
    hc = highconf.copy()
    hc["yengo_p"] = hc["Feature"].map(yengo)
    hc["known_direct"] = hc["yengo_p"] < GW_SIG
    ld = {}
    if geno_path is not None:
        cand = hc.loc[~hc["known_direct"], "Feature"].tolist()
        anchors = [s for s in hc.loc[hc["known_direct"], "Feature"]]  # + broader known set if desired
        anchors += [k for k, p in yengo.items() if p < GW_SIG]
        ld = _r2_to_anchors(cand, anchors, geno_path)
    def verdict(row):
        if row["known_direct"]:
            return "established (direct GWAS-significant)"
        r2 = ld.get(row["Feature"], np.nan)
        if pd.notna(r2) and r2 >= r2_thresh:
            return f"established (LD proxy, r2={r2:.2f})"
        return "candidate (requires +-500kb regional scan)"
    hc["max_r2_known"] = hc["Feature"].map(lambda s: ld.get(s, np.nan))
    hc["verdict"] = hc.apply(verdict, axis=1)
    return hc


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ML-EAT downstream analysis")
    ap.add_argument("--ranking", required=True)
    ap.add_argument("--perm-null", nargs="+", required=True)
    ap.add_argument("--yengo", required=True)
    ap.add_argument("--geno", default=None)
    ap.add_argument("--knee", type=int, required=True)
    ap.add_argument("--M", type=int, default=510274)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    ranking = load_ranking(a.ranking)
    null = pooled_null(a.perm_null)
    yengo = load_yengo(a.yengo)

    curve, hc = empirical_fdr(ranking, null, a.M, a.alpha)
    enr, trend_p, bg = enrichment(ranking, yengo, a.knee)
    hc = novelty(hc, yengo, a.geno)

    curve.to_csv(a.out_prefix + "_fdr_curve.csv", index=False)
    hc.to_csv(a.out_prefix + "_highconf.csv", index=False)
    enr.to_csv(a.out_prefix + "_enrichment.csv", index=False)

    print(f"[FDR<= {a.alpha:.0%}] high-confidence set: {len(hc)} variants")
    print(f"  established (direct or LD proxy): "
          f"{hc['verdict'].str.startswith('established').sum()}")
    print(f"[enrichment] Cochran-Armitage trend P = {trend_p:.2e}; background known = {bg:.3%}")
    print(f"outputs written with prefix: {a.out_prefix}")


if __name__ == "__main__":
    main()
