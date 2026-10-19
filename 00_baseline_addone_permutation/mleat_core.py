#!/usr/bin/env python3
"""
mleat_core.py
Shared model-fitting core for the genotype-permutation null
(S03_genotype_permutation_null.py): the observed fit (permutation_id 0) and
every permutation refit use exactly the same model.  The genome-wide add-one
screen (S02_addone_screen.py, the original cluster script) uses the same split,
scaling and alpha (Lasso(alpha=0.01)).

Model (as run on the NCHC cluster for the 2026 analysis):
  * single 80/20 split, sklearn train_test_split(test_size=0.2, random_state=42),
    applied to row positions of the per-sex analysis matrix;
  * StandardScaler fitted on the 80 % training partition only;
  * LASSO with a fixed regularisation strength alpha = 0.01
    (implemented as LassoCV(cv=5, random_state=42, alphas=[0.01]); with a single
    alpha the CV does not tune anything and the fit equals Lasso(alpha=0.01));
  * mse      = MSE on the 20 % held-out test partition   -> column lasso_mse_test
  * mse_cv   = mean 5-fold CV MSE inside the 80 % training partition
                                                         -> column lasso_mse_cv

Baseline-model (BM) features are the cofactors and genetic PCs retained by the
sex-specific LASSO baseline model (Figure 2; 32 female / 27 male).
"""
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

SPLIT_SEED = 42
TEST_SIZE = 0.2
LASSO_ALPHA = 0.01
CV_FOLDS = 5

# Baseline-model features retained by LASSO (female 32)
FEMALE_BM_FEATURES = [
    'AGE', 'EDUCATION',
    'NATIVE_MOM_F', 'NATIVE_MOM_CHINA', 'NATIVE_MOM_O',
    'NATIVE_FA_H', 'NATIVE_FA_CHINA', 'NATIVE_FA_O',
    'MENARCHE', 'MENSTRURATION_PERIOD', 'PREGNANCY', 'BREAST_MILK',
    'Z_SCORE',
    'PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA9',
    'PCA10', 'PCA11', 'PCA12', 'PCA13', 'PCA14', 'PCA15', 'PCA16',
    'PCA17', 'PCA18', 'PCA19', 'PCA20',
]

# Baseline-model features retained by LASSO (male 27)
MALE_BM_FEATURES = [
    'AGE', 'EDUCATION',
    'NATIVE_MOM_H', 'NATIVE_MOM_CHINA', 'NATIVE_MOM_O',
    'NATIVE_FA_F', 'NATIVE_FA_CHINA', 'NATIVE_FA_O',
    'Z_SCORE',
    'PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA7', 'PCA8', 'PCA9',
    'PCA11', 'PCA12', 'PCA13', 'PCA14', 'PCA15', 'PCA16', 'PCA17',
    'PCA18', 'PCA19', 'PCA20',
]

PHENOTYPE = 'BODY_HEIGHT'


def bm_features(sex):
    """Return the baseline-model feature list for 'female' or 'male'."""
    if sex == 'female':
        return list(FEMALE_BM_FEATURES)
    if sex == 'male':
        return list(MALE_BM_FEATURES)
    raise ValueError(f"sex must be 'female' or 'male', got {sex!r}")


def detect_sex(filename):
    """Infer sex from a file name ('female' is tested first: it contains 'male')."""
    name = str(filename).lower()
    if 'female' in name:
        return 'female'
    if 'male' in name:
        return 'male'
    return 'unknown'


def split_indices(n_samples):
    """Row positions of the single 80/20 split (random_state = 42)."""
    return train_test_split(np.arange(n_samples), test_size=TEST_SIZE,
                            random_state=SPLIT_SEED)


def fit_and_score(X_train, X_test, y_train, y_test):
    """Fit the LASSO model on the training partition and return (mse, mse_cv).

    mse    : held-out MSE on the 20 % test partition
    mse_cv : mean 5-fold cross-validated MSE within the 80 % training partition
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso = LassoCV(cv=CV_FOLDS, random_state=SPLIT_SEED, alphas=[LASSO_ALPHA])
    lasso.fit(X_train_scaled, y_train)
    mse = mean_squared_error(y_test, lasso.predict(X_test_scaled))

    cv_scores = cross_val_score(lasso, X_train_scaled, y_train, cv=CV_FOLDS,
                                scoring='neg_mean_squared_error')
    mse_cv = float(np.mean(-cv_scores))
    return mse, mse_cv
