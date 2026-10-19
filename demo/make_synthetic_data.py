#!/usr/bin/env python3
"""
make_synthetic_data.py
Generate a small, fully SYNTHETIC data set with the same layout as the
(non-shareable) Taiwan Biobank analysis matrices, so that the ML-EAT pipeline
can be run end to end without access to TWB data.

All values are drawn from random-number generators with a fixed seed. No real
participant, genotype or phenotype data were used, sampled or perturbed. SNP
identifiers (rsSIM00001_A, ...) are fictitious and do not correspond to dbSNP
entries.

Simulation model (per sex):
  * cofactors with the TWB names: age, education, parental ancestry (one-hot),
    bone-density Z-score, reproductive history (females), PC1-PC20;
    continuous/ordinal variables scaled to [0, 1];
  * --n-snps biallelic SNPs, minor-allele frequency U(0.05, 0.5),
    genotypes 0/1/2 drawn independently (Hardy-Weinberg);
  * the same SNPs and genetic architecture in both sexes;
  * --n-causal SNPs with additive effects of 0.8-1.6 cm per allele; three
    "proxy" SNPs are copies of causal SNPs with 5 % of genotypes resampled
    (high LD, no direct effect);
  * height = sex-specific mean + cofactor effects + SNP effects + N(0, 5^2) cm.

Outputs (in --out-dir):
  {sex}_bm_candidates.csv                     input of S01 (Release_No,
                                               BODY_HEIGHT, 37 F / 33 M candidates)
  matrices_{sex}/height_merged_{sex}_mtx_demo.mtx.csv
                                               input of S02 and S03 (Release_No,
                                               BODY_HEIGHT, BM features, SNPs)
  reference_gwas_demo.tsv.gz                   mock "established loci" reference
                                               (RSID, P) for the enrichment step
  truth_{sex}.csv                              simulated causal / proxy SNPs

Usage:  python make_synthetic_data.py --out-dir demo_data
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '00_baseline_addone_permutation'))
from mleat_core import FEMALE_BM_FEATURES, MALE_BM_FEATURES  # noqa: E402

PCS = [f'PCA{i}' for i in range(1, 21)]
MOM = ['NATIVE_MOM_F', 'NATIVE_MOM_H', 'NATIVE_MOM_CHINA', 'NATIVE_MOM_AB', 'NATIVE_MOM_O']
FA = ['NATIVE_FA_F', 'NATIVE_FA_H', 'NATIVE_FA_CHINA', 'NATIVE_FA_AB', 'NATIVE_FA_O']
REPRO = ['MENARCHE', 'MENSTRURATION_PERIOD', 'PREGNANCY', 'BREAST_MILK']


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def one_hot(rng, n, cols, p):
    k = rng.choice(len(cols), size=n, p=p)
    return pd.DataFrame({c: (k == i).astype(int) for i, c in enumerate(cols)})


def architecture(n_snps, n_causal, rng):
    """SNP architecture shared by both sexes (same SNPs, same causal effects)."""
    maf = rng.uniform(0.05, 0.5, n_snps)
    alleles = rng.choice(list('ACGT'), n_snps)
    names = [f'rsSIM{j + 1:05d}_{alleles[j]}' for j in range(n_snps)]
    causal = rng.choice(n_snps, n_causal, replace=False)
    effects = rng.uniform(0.8, 1.6, n_causal) * rng.choice([-1, 1], n_causal)
    proxies = rng.choice(np.setdiff1d(np.arange(n_snps), causal), 3, replace=False)
    maf[proxies] = maf[causal[:3]]
    return dict(maf=maf, names=names, causal=causal, effects=effects, proxies=proxies)


def simulate(sex, n, arch, rng):
    d = pd.DataFrame({'Release_No': [f'SIM{sex[0].upper()}{i:05d}' for i in range(n)]})
    age = rng.uniform(30, 70, n)
    edu = rng.integers(1, 8, n)
    d['AGE'] = minmax(age)
    d['EDUCATION'] = minmax(edu.astype(float))
    d = pd.concat([d, one_hot(rng, n, MOM, [0.70, 0.12, 0.12, 0.02, 0.04]),
                   one_hot(rng, n, FA, [0.70, 0.12, 0.12, 0.02, 0.04])], axis=1)
    if sex == 'female':
        d['MENARCHE'] = minmax(rng.normal(13, 1.3, n))
        d['MENSTRURATION_PERIOD'] = minmax(rng.normal(29, 3, n))
        d['PREGNANCY'] = rng.choice([1, 2], n, p=[0.2, 0.8])
        d['BREAST_MILK'] = rng.choice([1, 2], n, p=[0.4, 0.6])
    z = rng.normal(0, 1, n)
    d['Z_SCORE'] = minmax(z)
    for pc in PCS:
        d[pc] = rng.normal(0, 1, n) * 0.01

    # genotypes
    maf, names, causal, effects, proxies = (arch[k] for k in
                                            ('maf', 'names', 'causal', 'effects', 'proxies'))
    G = rng.binomial(2, maf, size=(n, len(maf)))
    for p_idx, c_idx in zip(proxies, causal[:3]):          # high-LD copies, no own effect
        g = G[:, c_idx].copy()
        flip = rng.random(n) < 0.05
        g[flip] = rng.binomial(2, maf[c_idx], flip.sum())
        G[:, p_idx] = g

    mean = 157.5 if sex == 'female' else 170.0
    h = (mean - 0.08 * (age - 50) + 0.4 * (edu - 4) + 1.0 * z
         + 1.5 * d['NATIVE_MOM_CHINA'] + 1.5 * d['NATIVE_FA_CHINA'])
    if sex == 'female':
        h = h + 2.0 * (d['MENARCHE'] - 0.5)
    h = h + G[:, causal] @ effects + rng.normal(0, 5.0, n)
    d['BODY_HEIGHT'] = np.round(h, 1)

    truth = pd.DataFrame({'SNP': [names[j] for j in causal], 'role': 'causal',
                          'effect_cm_per_allele': effects})
    truth = pd.concat([truth, pd.DataFrame({
        'SNP': [names[j] for j in proxies], 'role': 'proxy (high LD with causal)',
        'effect_cm_per_allele': 0.0,
        'proxy_of': [names[j] for j in causal[:3]]})], ignore_index=True)
    snps = pd.DataFrame(G, columns=names)
    return d, snps, truth


def main():
    ap = argparse.ArgumentParser(description="Synthetic demo data for ML-EAT")
    ap.add_argument('--out-dir', default='demo_data')
    ap.add_argument('--n-female', type=int, default=3000)
    ap.add_argument('--n-male', type=int, default=2500)
    ap.add_argument('--n-snps', type=int, default=600)
    ap.add_argument('--n-causal', type=int, default=10)
    ap.add_argument('--seed', type=int, default=20261019)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    arch = architecture(a.n_snps, a.n_causal, rng)
    ref = {}
    for sex, n, bm in [('female', a.n_female, FEMALE_BM_FEATURES),
                       ('male', a.n_male, MALE_BM_FEATURES)]:
        d, snps, truth = simulate(sex, n, arch, rng)
        candidates = ['AGE', 'EDUCATION'] + MOM + FA + (REPRO if sex == 'female' else []) \
            + ['Z_SCORE'] + PCS
        d[['Release_No', 'BODY_HEIGHT'] + candidates].to_csv(
            os.path.join(a.out_dir, f'{sex}_bm_candidates.csv'), index=False)
        mdir = os.path.join(a.out_dir, f'matrices_{sex}')
        os.makedirs(mdir, exist_ok=True)
        mat = pd.concat([d[['Release_No', 'BODY_HEIGHT'] + bm], snps], axis=1)
        mat.to_csv(os.path.join(mdir, f'height_merged_{sex}_mtx_demo.mtx.csv'), index=False)
        truth.to_csv(os.path.join(a.out_dir, f'truth_{sex}.csv'), index=False)
        # mock reference GWAS: 7 of the causal SNPs "established" (P = 1e-12)
        for s in truth.loc[truth.role == 'causal', 'SNP'].head(7):
            ref[s.split('_')[0]] = 1e-12
        for s in snps.columns:
            ref.setdefault(s.split('_')[0], float(rng.uniform(1e-3, 1)))
        print(f"{sex}: {n} participants, {len(bm)} BM features, {snps.shape[1]} SNPs, "
              f"{a.n_causal} causal")
    pd.DataFrame({'RSID': list(ref), 'P': list(ref.values())}).to_csv(
        os.path.join(a.out_dir, 'reference_gwas_demo.tsv.gz'), sep='\t', index=False,
        compression='gzip')
    print(f"synthetic data written to {a.out_dir}/")


if __name__ == '__main__':
    main()
