#!/usr/bin/env python3
"""
summarize_demo.py
Compare the demo results with the simulated truth: where the causal and proxy
SNPs rank, and which of them enter the Kneedle prioritization set and the
FDR-controlled high-confidence set.

Usage: python summarize_demo.py --work-dir demo_output --data-dir demo_data
"""
import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-dir', required=True)
    ap.add_argument('--data-dir', required=True)
    a = ap.parse_args()
    for sex in ['female', 'male']:
        truth = pd.read_csv(os.path.join(a.data_dir, f'truth_{sex}.csv'))
        r = pd.read_csv(os.path.join(a.work_dir, f'{sex}_ranking.csv'))
        base = r.loc[r.Feature == 'basic', 'lasso_mse_test'].iloc[0]
        r = r[r.Feature != 'basic'].sort_values('lasso_mse_test').reset_index(drop=True)
        r['rank'] = r.index + 1
        r['dMSE'] = base - r['lasso_mse_test']
        prio = set(pd.read_csv(os.path.join(a.work_dir, f'{sex}_prioritization_set.txt'),
                               header=None)[0])
        hc = pd.read_csv(os.path.join(a.work_dir, f'{sex}_highconf.csv'))
        hc_set = set(hc.Feature)
        t = truth.assign(Feature=truth.SNP.str.split('_').str[0]).merge(
            r[['Feature', 'rank', 'dMSE']], on='Feature', how='left')
        t['in_prioritization_set'] = t.Feature.isin(prio)
        t['in_high_confidence_set'] = t.Feature.isin(hc_set)
        print(f"\n===== {sex}: {len(r)} SNPs, baseline test MSE = {base:.3f} =====")
        print(f"prioritization set (Kneedle) = {len(prio)}, high-confidence set (FDR<=5%) = {len(hc)}")
        print(t[['SNP', 'role', 'effect_cm_per_allele', 'rank', 'dMSE',
                 'in_prioritization_set', 'in_high_confidence_set']].to_string(index=False))
        fp = hc_set - set(t.Feature)
        print(f"high-confidence SNPs that are neither causal nor proxy: {len(fp)}")


if __name__ == '__main__':
    main()
