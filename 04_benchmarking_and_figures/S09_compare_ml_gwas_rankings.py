#!/usr/bin/env python3
"""
Cut-off-free comparison of two rankings of the same SNPs (ML vs GWAS) at
recovering literature-supported loci.

Both rankings are scored on the same universe with the same labels, so every
statistic here is paired -- no top-N threshold has to be chosen.

    python3 compare_ml_gwas_rankings.py                # MaleCompare.txt + FemaleCompare.txt
    python3 compare_ml_gwas_rankings.py --male x.txt --female y.txt --boot 500

Per sex and per label (Ref > 0 = any reference; Ref > 5 = strongly supported):
  AUROC (ML, GWAS)   + DeLong paired test of the difference
  Average precision  + bootstrap 95% CI of the difference
  BEDROC (alpha=20)  early-recognition score
  Wilcoxon signed-rank on log10 rank of the labelled SNPs (which method ranks
                     the SAME known SNPs higher), plus median ranks
  Recall at 0.1%, 1%, 5% of the list, for both
Also: a figure with recall-vs-rank curves (log x) for both methods.
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
def auc_from_ranks(rank, pos):
    """AUROC when the score is 'smaller rank = better'."""
    n = len(rank)
    n_pos = pos.sum()
    n_neg = n - n_pos
    # score = n - rank  -> larger is better
    r = stats.rankdata(n - rank)
    return (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def delong_components(score, pos):
    """DeLong structural components V10 (per positive) and V01 (per negative)."""
    x, y = score[pos], score[~pos]
    m, n = len(x), len(y)
    allv = np.concatenate([x, y])
    r_all = stats.rankdata(allv)
    r_x = stats.rankdata(x)
    r_y = stats.rankdata(y)
    v10 = (r_all[:m] - r_x) / n
    v01 = 1.0 - (r_all[m:] - r_y) / m
    return v10, v01


def delong_paired(score_a, score_b, pos):
    """Two-sided DeLong test that AUROC(a) == AUROC(b) on the same samples."""
    v10a, v01a = delong_components(score_a, pos)
    v10b, v01b = delong_components(score_b, pos)
    auc_a, auc_b = v10a.mean(), v10b.mean()
    m, n = len(v10a), len(v01a)
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    s = s10 / m + s01 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (auc_a - auc_b) / np.sqrt(var)
    return auc_a, auc_b, z, 2 * stats.norm.sf(abs(z))


def average_precision(rank, pos):
    order = np.argsort(rank)
    hits = np.cumsum(pos[order])
    prec = hits / np.arange(1, len(rank) + 1)
    return prec[pos[order]].sum() / pos.sum()


def bedroc(rank, pos, alpha=20.0):
    n = len(rank)
    n_pos = pos.sum()
    ra = n_pos / n
    idx = rank[pos]
    rie = np.exp(-alpha * idx / n).sum() / (ra * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1))
    factor = ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra))
    return rie * factor + 1 / (1 - np.exp(alpha * (1 - ra)))


def recall_at(rank, pos, frac):
    k = int(round(len(rank) * frac))
    return (pos & (rank <= k)).sum() / pos.sum()


def recall_curve(rank, pos):
    order = np.argsort(rank)
    return np.cumsum(pos[order]) / pos.sum()


# --------------------------------------------------------------------------- #
def analyse(name, df, boot, rng):
    ml = df["ML_Rank"].to_numpy(float)
    gw = df["GWAS_Rank"].to_numpy(float)
    ref = df["Ref"].to_numpy(float)
    n = len(df)
    rows, curves = [], {}

    for label, pos in (("Ref>0", ref > 0), ("Ref>5", ref > 5)):
        n_pos = int(pos.sum())
        auc_ml, auc_gw, z, p_delong = delong_paired(n - ml, n - gw, pos)
        ap_ml, ap_gw = average_precision(ml, pos), average_precision(gw, pos)

        # bootstrap over SNPs for the AP difference
        diffs = np.empty(boot)
        idx_pos = np.flatnonzero(pos)
        idx_neg = np.flatnonzero(~pos)
        for b in range(boot):
            # stratified resample keeps the prevalence, which AP depends on
            samp = np.concatenate([rng.choice(idx_pos, len(idx_pos)),
                                   rng.choice(idx_neg, len(idx_neg))])
            pb = pos[samp]
            diffs[b] = (average_precision(stats.rankdata(ml[samp]), pb)
                        - average_precision(stats.rankdata(gw[samp]), pb))
        lo, hi = np.percentile(diffs, [2.5, 97.5])

        bed_ml, bed_gw = bedroc(ml, pos), bedroc(gw, pos)
        w_stat, p_wilcox = stats.wilcoxon(np.log10(ml[pos]), np.log10(gw[pos]))
        med_ml, med_gw = np.median(ml[pos]), np.median(gw[pos])
        better_ml = int((ml[pos] < gw[pos]).sum())

        row = dict(sex=name, label=label, n_snps=n, n_labelled=n_pos,
                   auroc_ML=auc_ml, auroc_GWAS=auc_gw, delong_z=z, delong_p=p_delong,
                   ap_ML=ap_ml, ap_GWAS=ap_gw, ap_diff=ap_ml - ap_gw,
                   ap_diff_ci_lo=lo, ap_diff_ci_hi=hi,
                   bedroc20_ML=bed_ml, bedroc20_GWAS=bed_gw,
                   median_rank_ML=med_ml, median_rank_GWAS=med_gw,
                   n_ranked_higher_by_ML=better_ml,
                   n_ranked_higher_by_GWAS=n_pos - better_ml,
                   wilcoxon_p=p_wilcox)
        for f in (0.001, 0.01, 0.05):
            row[f"recall_top{f*100:g}pct_ML"] = recall_at(ml, pos, f)
            row[f"recall_top{f*100:g}pct_GWAS"] = recall_at(gw, pos, f)
        rows.append(row)
        curves[label] = (recall_curve(ml, pos), recall_curve(gw, pos))

        print(f"\n== {name} / {label}  ({n_pos:,} of {n:,} SNPs)")
        print(f"   AUROC        ML {auc_ml:.4f}  GWAS {auc_gw:.4f}   DeLong z = {z:+.2f}, p = {p_delong:.3g}")
        print(f"   Avg precision ML {ap_ml:.4f}  GWAS {ap_gw:.4f}   diff {ap_ml-ap_gw:+.4f} "
              f"(95% CI {lo:+.4f} .. {hi:+.4f})")
        print(f"   BEDROC(20)   ML {bed_ml:.3f}  GWAS {bed_gw:.3f}")
        print(f"   median rank  ML {med_ml:,.0f}  GWAS {med_gw:,.0f}   "
              f"ML higher for {better_ml}/{n_pos} SNPs, Wilcoxon p = {p_wilcox:.3g}")
        for f in (0.001, 0.01, 0.05):
            print(f"   recall@{f*100:>4g}%  ML {recall_at(ml, pos, f):.3f}  GWAS {recall_at(gw, pos, f):.3f}")
    return rows, curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--male", default="MaleCompare.txt")
    p.add_argument("--female", default="FemaleCompare.txt")
    p.add_argument("--boot", type=int, default=300, help="bootstrap replicates")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--out", default="ML_vs_GWAS_ranking_comparison.csv")
    p.add_argument("--fig", default="ML_vs_GWAS_recall_curves.png")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    rows, allcurves = [], {}
    for name, path in (("Male", args.male), ("Female", args.female)):
        df = pd.read_csv(path, sep="\t")
        r, c = analyse(name, df, args.boot, rng)
        rows += r
        allcurves[name] = (c, len(df))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nsaved: {args.out}")

    # recall-vs-rank curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (name, (curves, n)) in zip(axes, allcurves.items()):
        k = np.arange(1, n + 1)
        for label, ls in (("Ref>0", "-"), ("Ref>5", "--")):
            c_ml, c_gw = curves[label]
            ax.plot(k, c_ml, ls, color="#1f6fd0", lw=1.4, label=f"ML  ({label})")
            ax.plot(k, c_gw, ls, color="#d1495b", lw=1.4, label=f"GWAS ({label})")
        ax.plot(k, k / n, ":", color="grey", lw=1, label="random")
        ax.set_xscale("log")
        ax.set_xlim(1, n)
        ax.set_ylim(0, 1)
        ax.set_xlabel("rank cut-off k (log scale)")
        ax.set_ylabel("fraction of labelled SNPs recovered")
        ax.set_title(f"{name}: recall of referenced SNPs vs rank")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.fig, dpi=200)
    fig.savefig(args.fig.rsplit(".", 1)[0] + ".pdf")
    print(f"saved: {args.fig}")


if __name__ == "__main__":
    main()
