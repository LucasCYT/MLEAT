#!/usr/bin/env python3
"""
S01_baseline_model_alpha_scan.py
Sex-specific non-genetic baseline model (BM) for standing height
(paper Methods, "Baseline prediction model"; Figure 2).

LASSO regression on the prespecified TWB cofactors + PC1-PC20 (no SNPs):
  * single 80/20 split, train_test_split(train_size=0.8, random_state=42);
  * missing feature values imputed with the column median;
  * StandardScaler fitted on the training partition;
  * regularisation strength scanned over 25 log-spaced values (1e-5 ... 10);
    for each value: 5-fold CV R^2 within the training partition, coefficients
    of the model fitted on the full training partition, and MSE / R^2 in each
    of three non-overlapping subsets of the 20 % test partition
    (Figure 2C, D: test-subset R^2 versus lambda).
  * The coefficients at alpha = 0.01 give the factor weights of Figure 2B.

Iterative feature selection: the scan is run on the full candidate set
(female 37 / male 33 features) and re-run on the retained features
(female 32 / male 27), which form the BM used by S02/S03 (mleat_core.py).

Input   --input CSV: participant ID column(s), BODY_HEIGHT, then the candidate
        features (the first --n-id-cols columns are skipped, as in the original
        notebook; BODY_HEIGHT must not be among the feature columns).
Output  {out-prefix}_lasso_results.csv             Alpha, CV_Mean_R2, Test Set, MSE, R-squared, Test Set Size
        {out-prefix}_lasso_feature_importance.csv  Alpha, Feature, Coefficient

Usage:
  python S01_baseline_model_alpha_scan.py --input Merged_Female_Data_OnlyRetainZ-score.csv \
         --subset-size 2800 --out-prefix female_37AllFeatures       # first round
  python S01_baseline_model_alpha_scan.py --input BM_Female_Data_C33.csv \
         --subset-size 2800 --out-prefix female_32AllFeatures       # retained features
  (male: Merged_Male_Data_OnlyRetainZ-score.csv / BM_Male_Data_C27.csv, --subset-size 2190)

Provenance: cleaned from NewData/Figure2/Data/Source/LASSO_ScanAlpha_260826.ipynb
(four cells: male/female x first round/retained features); computation unchanged.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-5, 1, 25)


def main():
    ap = argparse.ArgumentParser(description="ML-EAT baseline-model LASSO alpha scan")
    ap.add_argument('--input', required=True)
    ap.add_argument('--subset-size', type=int, required=True,
                    help="size of test subsets 1 and 2 (female 2800, male 2190); subset 3 = remainder")
    ap.add_argument('--n-id-cols', type=int, default=2)
    ap.add_argument('--out-prefix', required=True)
    a = ap.parse_args()

    gtD = pd.read_csv(a.input)
    keep = gtD[gtD['BODY_HEIGHT'].notnull()].index
    X = gtD.iloc[keep, a.n_id_cols:]
    if 'BODY_HEIGHT' in X.columns:
        raise SystemExit("BODY_HEIGHT found among feature columns; check --n-id-cols")
    y = gtD.loc[keep, 'BODY_HEIGHT']
    X = X.fillna(X.median())
    print(f"{len(y)} participants, {X.shape[1]} candidate features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # three non-overlapping subsets of the 20 % test partition
    X_t1, X_tmp, y_t1, y_tmp = train_test_split(X_test_s, y_test, train_size=a.subset_size, random_state=42)
    X_t2, X_t3, y_t2, y_t3 = train_test_split(X_tmp, y_tmp, train_size=a.subset_size, random_state=42)
    test_sets = {'Test Set 1': (X_t1, y_t1), 'Test Set 2': (X_t2, y_t2), 'Test Set 3': (X_t3, y_t3)}

    results, importance = [], []
    for alpha in ALPHAS:
        model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
        cv_r2 = np.mean(cross_val_score(model, X_train_s, y_train, cv=5, scoring='r2'))
        model.fit(X_train_s, y_train)
        for feat, coef in zip(X_train.columns, model.coef_):
            importance.append({'Alpha': alpha, 'Feature': feat, 'Coefficient': coef})
        for name, (Xs, ys) in test_sets.items():
            pred = model.predict(Xs)
            results.append({'Alpha': alpha, 'CV_Mean_R2': cv_r2, 'Test Set': name,
                            'MSE': mean_squared_error(ys, pred), 'R-squared': r2_score(ys, pred),
                            'Test Set Size': len(ys)})

    pd.DataFrame(results).to_csv(f"{a.out_prefix}_lasso_results.csv", index=False)
    pd.DataFrame(importance).to_csv(f"{a.out_prefix}_lasso_feature_importance.csv", index=False)
    print(f"wrote {a.out_prefix}_lasso_results.csv / _lasso_feature_importance.csv")


if __name__ == '__main__':
    main()
