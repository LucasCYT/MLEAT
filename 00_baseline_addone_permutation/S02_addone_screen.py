#!/usr/bin/env python3
"""
S02_addone_screen.py
Genome-wide add-one predictive-contribution screen (paper Methods,
"Add-one SNP predictive-contribution analysis"; Figure 1, Step 2).

For every SNP column of each analysis matrix, a separate LASSO model
(alpha = 0.01) is fitted on the 80 % training partition with the sex-specific
baseline-model (BM) features plus that SNP's genotype (all coefficients
re-estimated jointly) and evaluated on the 20 % held-out test partition.
The first row of every output file (index -1, columns == 'basic_columns') is
the BM-only fit.

  lasso_mse_test : MSE on the 20 % held-out test partition   (discovery ranking)
  lasso_mse_cv   : mean 5-fold CV MSE inside the 80 % training partition
                   (training-partition PGS panel, 03_pgs/S07)
  dMSE = MSE(BM) - MSE(BM + SNP)

Processing, exactly as run on the NCHC cluster (2026-06-12):
  * analysis matrix columns: Release_No (index), then the BM block
    (BODY_HEIGHT + 32 female / 27 male BM features), then SNP genotype columns
    (rsID_allele, coded 0/1/2; -1 = missing call);
  * missing genotype calls (-1) are set to 0 (major-allele homozygote); this
    is a safeguard only - participants with no-call genotypes were removed
    upstream, so the final analysis matrices contain no -1 values;
  * any remaining missing values are replaced by the column mean;
  * single 80/20 split, train_test_split(test_size=0.2, random_state=42);
  * StandardScaler fitted on the training partition;
  * Lasso(alpha=0.01, random_state=42); 5-fold CV MSE via cross_val_score.

Input   --data-dir  directory containing height_merged_{female,male}_mtx*.csv
                    (one block of SNPs per file)
Output  --out-dir/{file}_lasso_results.csv
                    columns: index, columns, lasso_mse_cv, lasso_mse_test
        The per-block files are concatenated into
        {sex}_cv_5_lasso_aligned_merged_20260702.csv (Feature = rsID part of
        'columns'; the BM row becomes Feature == 'basic').

Usage:
  python S02_addone_screen.py --data-dir mergerdataFolder_female_1/ \
                              --out-dir  lassoLoop_RES/

Provenance: cleaned from twb_lassoLoop_0612.py (run 2026-06-12); the
computation is unchanged (verified to give identical output on test data).
The original wrote the output CSV after every SNP; here it is written once
per file.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

PHENOTYPE = 'BODY_HEIGHT'
N_BASIC = {'female': 33, 'male': 28}   # BODY_HEIGHT + 32 / 27 BM features


def addone_file(path):
    name = os.path.basename(path)
    sex = 'female' if name.startswith('height_merged_female_mtx') else 'male'
    data = pd.read_csv(path).set_index('Release_No')

    basic = data.columns.tolist()[:N_BASIC[sex]]
    snps = [c for c in data.columns if c not in basic]
    data[snps] = data[snps].replace(-1, 0)              # missing call -> major allele
    bm = [c for c in basic if c != PHENOTYPE]

    X = data.drop(columns=[PHENOTYPE])
    y = data[PHENOTYPE]
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    rows = []
    for i in range(-1, len(snps)):
        cols = bm if i == -1 else bm + [snps[i]]
        label = 'basic_columns' if i == -1 else snps[i]
        model = Lasso(alpha=0.01, random_state=42)
        mse_cv = np.mean(-cross_val_score(model, X_train_s[cols], y_train, cv=5,
                                          scoring='neg_mean_squared_error'))
        model.fit(X_train_s[cols], y_train)
        mse_test = mean_squared_error(y_test, model.predict(X_test_s[cols]))
        rows.append({'index': i, 'columns': label,
                     'lasso_mse_cv': mse_cv, 'lasso_mse_test': mse_test})
        print(f"{name} {i} {label} lasso_mse_cv={mse_cv:.6f} lasso_mse_test={mse_test:.6f}")
    print(f"{name}: {sex}, {len(bm)} BM features, {len(snps)} SNPs, {len(data)} participants")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="ML-EAT genome-wide add-one screen")
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(a.data_dir) if f.startswith('height_merged_'))
    for f in files:
        res = addone_file(os.path.join(a.data_dir, f))
        res.to_csv(os.path.join(a.out_dir, f"{f}_lasso_results.csv"), index=False)


if __name__ == '__main__':
    main()
