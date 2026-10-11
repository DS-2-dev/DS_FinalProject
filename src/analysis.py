#!/usr/bin/env python3
"""Primary statistical analysis on the cleaned dataset.

Answers the research question: is weekly time online associated with the
ability to tell AI-generated images from human-made ones (`score_pct`)?

Writes to `outputs/analysis/`:
- `dataset_overview.csv`, `gender_counts.csv`, `numeric_summary.csv`
- `correlations.csv` (Pearson and Spearman, time vs score)
- `pooled_ols_summary.txt`, `pooled_ols_coefficients.csv` (HC3 robust SEs)
- `mixed_effects_summary.txt` (random intercept by gender; only 2 groups, so
  expect a singular fit and treat as a sensitivity check)
- `summary.json` (headline numbers, used by the website)

Run: python src/analysis.py
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import KFold

from cleaning import clean_dataframe
from common import CLEANED_CSV, OUTPUTS, RAW_CSV

FORMULA = 'score_pct ~ time_spent_hours + C(gender) + age'


def cv_rmse(df, k=5):
    errs = []
    for train, test in KFold(n_splits=k, shuffle=True, random_state=42).split(df):
        fit = smf.ols(FORMULA, data=df.iloc[train]).fit()
        pred = fit.predict(df.iloc[test])
        errs.append(((df.iloc[test]['score_pct'] - pred) ** 2).mean())
    return float(np.sqrt(np.mean(errs)))


def main():
    df = pd.read_csv(CLEANED_CSV)
    n_imputed = int(clean_dataframe(pd.read_csv(RAW_CSV))['time_spent_hours'].isna().sum())
    out = OUTPUTS / 'analysis'
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({'n_rows': [len(df)]}).to_csv(out / 'dataset_overview.csv', index=False)
    gender_counts = df['gender'].value_counts().rename_axis('gender').reset_index(name='n')
    gender_counts.to_csv(out / 'gender_counts.csv', index=False)
    df[['time_spent_hours', 'score_pct', 'age']].describe().T.to_csv(out / 'numeric_summary.csv')

    pearson = stats.pearsonr(df['time_spent_hours'], df['score_pct'])
    spearman = stats.spearmanr(df['time_spent_hours'], df['score_pct'])
    pd.DataFrame([
        {'method': 'pearson', 'r': pearson.statistic, 'pval': pearson.pvalue},
        {'method': 'spearman', 'r': spearman.statistic, 'pval': spearman.pvalue},
    ]).to_csv(out / 'correlations.csv', index=False)

    ols = smf.ols(FORMULA, data=df).fit(cov_type='HC3')
    (out / 'pooled_ols_summary.txt').write_text(str(ols.summary()))
    coefs = pd.DataFrame({'coef': ols.params, 'se': ols.bse, 'pval': ols.pvalues,
                          'ci_low': ols.conf_int()[0], 'ci_high': ols.conf_int()[1]})
    coefs.to_csv(out / 'pooled_ols_coefficients.csv')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        mixed = smf.mixedlm('score_pct ~ time_spent_hours + age', df, groups=df['gender']).fit(reml=False)
    mixed_text = str(mixed.summary())
    if caught:
        mixed_text += '\n\nWarnings:\n' + '\n'.join(sorted({str(w.message) for w in caught}))
    (out / 'mixed_effects_summary.txt').write_text(mixed_text)

    def term(name):
        return {k: float(coefs.loc[name, k]) for k in coefs.columns}

    summary = {
        'n': int(len(df)),
        'n_imputed_time': n_imputed,
        'gender_counts': dict(zip(gender_counts['gender'], gender_counts['n'].astype(int))),
        'score_pct_mean': float(df['score_pct'].mean()),
        'score_pct_sd': float(df['score_pct'].std()),
        'time_hours_median': float(df['time_spent_hours'].median()),
        'age_median': float(df['age'].median()),
        'pearson': {'r': float(pearson.statistic), 'pval': float(pearson.pvalue)},
        'spearman': {'r': float(spearman.statistic), 'pval': float(spearman.pvalue)},
        'ols': {
            'r2': float(ols.rsquared),
            'cv_rmse': cv_rmse(df),
            'time_spent_hours': term('time_spent_hours'),
            'gender_male': term('C(gender)[T.Male]'),
            'age': term('age'),
            'intercept': term('Intercept'),
        },
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('Wrote analysis outputs to', out)
    print(f"OLS R2={summary['ols']['r2']:.3f}, time coef={summary['ols']['time_spent_hours']['coef']:.2e} "
          f"(p={summary['ols']['time_spent_hours']['pval']:.3f})")


if __name__ == '__main__':
    main()
