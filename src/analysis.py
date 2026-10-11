#!/usr/bin/env python3
"""Primary statistical analysis on the cleaned dataset.

Answers the research question: is weekly time online associated with the
ability to tell AI-generated images from human-made ones (`score_pct`)?

Writes to `outputs/analysis/`:
- `dataset_overview.csv`, `gender_counts.csv`, `numeric_summary.csv`
- `correlations.csv` (Pearson and Spearman, time vs score)
- `pooled_ols_summary.txt`, `pooled_ols_coefficients.csv` (HC3 robust SEs)
- `by_gender.csv`: descriptives, correlations and a time + age OLS within
  each gender (also refit on ages <= 40 to check whether a few older
  participants drive the age effect)
- `robustness.csv`: the time effect under alternative specifications
- `summary.json` (headline numbers, used by the website)

Run: python src/analysis.py
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import KFold

from cleaning import clean_dataframe
from common import CLEANED_CSV, GENDERS, OUTPUTS, RAW_CSV

FORMULA = 'score_pct ~ time_spent_hours + C(gender) + age'
OUTLIER_HOURS = 100  # answers above this are treated as implausible in the robustness check


def cv_rmse(df, k=5):
    errs = []
    for train, test in KFold(n_splits=k, shuffle=True, random_state=42).split(df):
        fit = smf.ols(FORMULA, data=df.iloc[train]).fit()
        pred = fit.predict(df.iloc[test])
        errs.append(((df.iloc[test]['score_pct'] - pred) ** 2).mean())
    return float(np.sqrt(np.mean(errs)))


def term(fit, name):
    ci = fit.conf_int().loc[name]
    return {'coef': float(fit.params[name]), 'se': float(fit.bse[name]), 'pval': float(fit.pvalues[name]),
            'ci_low': float(ci[0]), 'ci_high': float(ci[1])}


def per_10h(fit, df, name='time_spent_hours'):
    """Change in score (percentage points) for +10 h/week, from the median, with 95% CI."""
    t = term(fit, name)
    if name == 'log_hours':
        med = df['time_spent_hours'].median()
        scale = np.log1p(med + 10) - np.log1p(med)
    else:
        scale = 10
    return {k: t[k] * scale * 100 for k in ('coef', 'ci_low', 'ci_high')} | {'pval': t['pval']}


def robustness(df, imputed):
    specs = [
        ('Main model', df, FORMULA, 'time_spent_hours'),
        (f'Without answers over {OUTLIER_HOURS} h', df[df['time_spent_hours'] <= OUTLIER_HOURS], FORMULA, 'time_spent_hours'),
        ('Log of hours', df.assign(log_hours=np.log1p(df['time_spent_hours'])),
         'score_pct ~ log_hours + C(gender) + age', 'log_hours'),
        ('Without the imputed row', df[~imputed], FORMULA, 'time_spent_hours'),
        ('Time only (no controls)', df, 'score_pct ~ time_spent_hours', 'time_spent_hours'),
    ]
    rows = []
    for label, data, formula, name in specs:
        fit = smf.ols(formula, data=data).fit(cov_type='HC3')
        eff = per_10h(fit, data, name)
        rows.append({'spec': label, 'n': int(fit.nobs), 'r2': float(fit.rsquared),
                     'effect_per_10h_pts': eff['coef'], 'ci_low_pts': eff['ci_low'],
                     'ci_high_pts': eff['ci_high'], 'pval': eff['pval']})
    return pd.DataFrame(rows)


def by_gender(df):
    rows = []
    for g in GENDERS:
        sub = df[df['gender'] == g]
        fit = smf.ols('score_pct ~ time_spent_hours + age', data=sub).fit(cov_type='HC3')
        r = stats.pearsonr(sub['time_spent_hours'], sub['score_pct'])
        rho = stats.spearmanr(sub['time_spent_hours'], sub['score_pct'])
        eff = per_10h(fit, sub)
        age = term(fit, 'age')
        young = sub[sub['age'] <= 40]
        age_young = term(smf.ols('score_pct ~ time_spent_hours + age', data=young).fit(cov_type='HC3'), 'age')
        rows.append({
            'gender': g, 'n': len(sub),
            'score_mean': sub['score_pct'].mean(), 'score_sd': sub['score_pct'].std(),
            'score_median': sub['score_pct'].median(),
            'pct_above_chance': (sub['score_pct'] > 0.5).mean(),
            'hours_median': sub['time_spent_hours'].median(), 'age_median': sub['age'].median(),
            'pearson_r': r.statistic, 'pearson_p': r.pvalue,
            'spearman_rho': rho.statistic, 'spearman_p': rho.pvalue,
            'effect_per_10h_pts': eff['coef'], 'ci_low_pts': eff['ci_low'], 'ci_high_pts': eff['ci_high'],
            'time_pval': eff['pval'], 'age_coef_pts_per_year': age['coef'] * 100, 'age_pval': age['pval'],
            'age_spearman_rho': stats.spearmanr(sub['age'], sub['score_pct']).statistic,
            'age_spearman_p': stats.spearmanr(sub['age'], sub['score_pct']).pvalue,
            'n_over_40': int((sub['age'] > 40).sum()),
            'age_coef_under40_pts': age_young['coef'] * 100, 'age_pval_under40': age_young['pval'],
            'r2': fit.rsquared,
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(CLEANED_CSV)
    imputed = clean_dataframe(pd.read_csv(RAW_CSV))['time_spent_hours'].isna().to_numpy()
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

    genders = by_gender(df)
    genders.to_csv(out / 'by_gender.csv', index=False)
    male, female = (df.loc[df['gender'] == g, 'score_pct'] for g in ('Male', 'Female'))
    welch = stats.ttest_ind(male, female, equal_var=False)
    mwu = stats.mannwhitneyu(male, female)

    robust = robustness(df, imputed)
    robust.to_csv(out / 'robustness.csv', index=False)

    summary = {
        'n': int(len(df)),
        'n_imputed_time': int(imputed.sum()),
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
            'time_spent_hours': term(ols, 'time_spent_hours'),
            'gender_male': term(ols, 'C(gender)[T.Male]'),
            'age': term(ols, 'age'),
            'intercept': term(ols, 'Intercept'),
        },
        'gender_gap': {
            'male_minus_female_pts': float((male.mean() - female.mean()) * 100),
            'welch_t': float(welch.statistic), 'welch_p': float(welch.pvalue),
            'mannwhitney_p': float(mwu.pvalue),
            'cohens_d': float((male.mean() - female.mean()) / np.sqrt((male.var() + female.var()) / 2)),
        },
        'by_gender': json.loads(genders.to_json(orient='records')),
        'robustness': json.loads(robust.to_json(orient='records')),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('Wrote analysis outputs to', out)
    print(robust.round(3).to_string(index=False))
    print(genders[['gender', 'n', 'score_mean', 'pearson_r', 'effect_per_10h_pts', 'time_pval', 'age_coef_pts_per_year', 'age_pval', 'age_pval_under40']].round(3).to_string(index=False))
    print('Gender gap:', {k: round(v, 3) for k, v in summary['gender_gap'].items()})


if __name__ == '__main__':
    main()
