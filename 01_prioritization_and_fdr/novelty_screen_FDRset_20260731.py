#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
針對「新 FDR 高信賴集」做文獻(Yengo 2022)比對。
流程:
  1. 用「三層合併虛無」估 p0(t),配完整 510k 排序算 FDR(t)=M*p0/R。
  2. 取 FDR<=0.05 的高信賴集(女~84 / 男~19)。
  3. 每個 SNP 以 rsID 對位 Yengo ALL+EAS → 標 KNOWN(p<5e-8)/ 非顯著。
  4. (女性)對「非 Yengo 顯著」者,用 TWB 基因型算對已知位點的 max r²,
     >=0.2 → 已知位點 proxy;<0.2 → 待進一步(位置掃描)判定。
輸出:每性別一份 annotated CSV + 螢幕摘要。
"""
import pandas as pd, numpy as np, glob, os

RK   = "/path/to/inputs"          # ranking files + Yengo summary statistics
NEW  = "/path/to/permutation_results"   # results_filter_v4_*_NoFilter_SnpTrainShuff/
GENO = "/path/to/Female_TopSNP_240621-002-030.mtx"
M    = 510274
SIG  = 5e-8

def load_yengo(fname):
    y = pd.read_csv(f"{RK}/{fname}", sep="\t", usecols=["RSID","P"])
    y["P"] = pd.to_numeric(y["P"], errors="coerce")
    return dict(zip(y["RSID"], y["P"]))

def pooled_null(files):
    v = []
    for f in files:
        d = pd.read_csv(f); mc = [c for c in d.columns if c.startswith("mse_")]
        a = d.loc[d["permutation_id"] != 0, mc].values.flatten().astype(float)
        v.append(a[~np.isnan(a)])
    return np.sort(np.concatenate(v))

def fdr_set(rank_file, null_files, alpha=0.05, cap=4000):
    null = pooled_null(null_files)
    r = pd.read_csv(rank_file); r = r[r["Feature"] != "basic"].copy()
    r = r.sort_values("lasso_mse_test").reset_index(drop=True)
    r["rank"] = np.arange(1, len(r)+1)
    real = r["lasso_mse_test"].values.astype(float)
    p0 = lambda t: np.searchsorted(null, t, side="right")/len(null)
    best = 0
    for R in range(1, min(len(real), cap)):
        if M*p0(real[R-1])/R <= alpha: best = R
        else: break
    sub = r.head(best).copy()
    sub["FDR"] = [min(1.0, M*p0(t)/rk) for t, rk in zip(sub["lasso_mse_test"], sub["rank"])]
    return sub, null

def annotate(sub, yALL, yEAS):
    sub["yengo_ALL"] = sub["Feature"].map(yALL)
    sub["yengo_EAS"] = sub["Feature"].map(yEAS)
    sub["yengo_min"] = sub[["yengo_ALL","yengo_EAS"]].min(axis=1)
    sub["in_yengo"]  = sub["Feature"].isin(yALL) | sub["Feature"].isin(yEAS)
    sub["known"]     = sub["yengo_min"] < SIG
    return sub

def ld_proxy_female(sub, yALL):
    """對女性『非 Yengo 顯著』候選,算對已知位點 anchor 的 max r²(用 TWB 基因型)。"""
    with open(GENO, encoding="utf-8", errors="ignore") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    col = {c.split("_")[0]: c for c in header if c.startswith("rs")}
    sigset = {k for k, p in yALL.items() if p < SIG}
    cand = [s for s in sub.loc[~sub["known"], "Feature"] if s in col]
    anch = [s for s in col if s in sigset]
    if not cand:
        return {}
    use = [col[s] for s in cand] + [col[s] for s in anch]
    d = pd.read_csv(GENO, sep="\t", usecols=use)
    def prep(names):
        A = np.empty((len(d), len(names)), dtype=np.float32)
        for j, s in enumerate(names):
            v = pd.to_numeric(d[col[s]], errors="coerce").values.astype(np.float32)
            v[~np.isin(v, [0,1,2])] = np.nan
            v[np.isnan(v)] = np.nanmean(v)
            c = v - v.mean(); sd = c.std(); A[:, j] = c/(sd if sd > 0 else 1)
        return A
    Zc, Zs, n = prep(cand), prep(anch), len(d)
    mx = np.zeros(len(cand))
    for i in range(0, Zs.shape[1], 400):
        mx = np.maximum(mx, ((Zc.T @ Zs[:, i:i+400])/n)**2).max(axis=1) if False else \
             np.maximum(mx, (((Zc.T @ Zs[:, i:i+400])/n)**2).max(axis=1))
    return dict(zip(cand, mx))

def verdict(row, ld):
    if row["known"]:
        return "KNOWN (Yengo GW-sig)"
    r2 = ld.get(row["Feature"], np.nan)
    if pd.notna(r2) and r2 >= 0.2:
        return f"KNOWN proxy (r2={r2:.2f})"
    return "candidate (待位置掃描)"

def main():
    yALL = load_yengo("GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_ALL.gz")
    yEAS = load_yengo("GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_EAS.gz")
    jobs = {
        "FEMALE": (f"{RK}/female_cv_5_lasso_aligned_merged_20260702.csv",
                   [f"{NEW}/results_filter_v4_NoFilter_SnpTrainShuff/Top1000_female_snps_merged.csv",
                    f"{NEW}/results_filter_v4_Middle_NoFilter_SnpTrainShuff/Middle1000_female_snps_merged.csv",
                    f"{NEW}/results_filter_v4_Last_NoFilter_SnpTrainShuff/Last1000_female_snps_merged.csv"]),
        "MALE":   (f"{RK}/male_cv_5_lasso_aligned_merged_20260702.csv",
                   [f"{NEW}/results_filter_v4_NoFilter_SnpTrainShuff/Top300_male_snps_merged.csv",
                    f"{NEW}/results_filter_v4_Middle_NoFilter_SnpTrainShuff/Middle300_male_snps_merged.csv",
                    f"{NEW}/results_filter_v4_Last_NoFilter_SnpTrainShuff/Last300_male_snps_merged.csv"]),
    }
    for sex, (rf, nf) in jobs.items():
        sub, _ = fdr_set(rf, nf)
        sub = annotate(sub, yALL, yEAS)
        ld = ld_proxy_female(sub, yALL) if sex == "FEMALE" else {}
        sub["verdict"] = sub.apply(lambda r: verdict(r, ld), axis=1)
        n = len(sub); nk = sub["known"].sum(); nproxy = sub["verdict"].str.startswith("KNOWN proxy").sum()
        ncand = (sub["verdict"].str.startswith("candidate")).sum()
        print(f"\n===== {sex}  5% FDR 高信賴集 = {n} =====")
        print(f"  Yengo 直接顯著(KNOWN) = {nk}")
        if sex == "FEMALE":
            print(f"  已知位點 proxy (r2>=0.2) = {nproxy}")
            print(f"  candidate(待位置掃描)   = {ncand}")
        out = sub[["Feature","rank","FDR","yengo_ALL","yengo_EAS","verdict"]]
        out.to_csv(f"{RK}/FDRset_novelty_{sex}_20260731.csv", index=False)
        print(f"  → 已存 FDRset_novelty_{sex}_20260731.csv")
        # 列出非 KNOWN 的(要人工看的)
        nonk = sub.loc[~sub["known"], ["Feature","rank","verdict"]]
        if len(nonk):
            print("  非直接已知者:")
            for _, x in nonk.iterrows():
                print(f"    r{int(x['rank']):>4} {x['Feature']:>12}  {x['verdict']}")

if __name__ == "__main__":
    main()
