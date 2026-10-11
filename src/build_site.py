#!/usr/bin/env python3
"""Export analysis results to `docs/data.json` for the GitHub Pages site.

The site itself is `docs/index.html`; it reads this file at load time.

Run: python src/build_site.py
"""
import json

import pandas as pd

from common import CLEANED_CSV, OUTPUTS, ROOT, age_to_range


def records(df):
    return json.loads(df.to_json(orient='records'))


def main():
    df = pd.read_csv(CLEANED_CSV)
    df['age_range'] = df['age'].apply(age_to_range)

    metrics = pd.read_csv(OUTPUTS / 'models' / 'subgroup_model_metrics.csv')
    curves = pd.read_csv(OUTPUTS / 'models' / 'plots' / 'subgroup_model_summary.csv')
    metrics = metrics.merge(curves[['age_range', 'gender', 'pr_auc']], on=['age_range', 'gender'], how='left')

    coefs = pd.read_csv(OUTPUTS / 'analysis' / 'pooled_ols_coefficients.csv', index_col=0)
    coefs = coefs.drop(index='Intercept').rename_axis('term').reset_index()

    group_sizes = (df.groupby(['age_range', 'gender']).size()
                   .rename('n').reset_index())

    data = {
        'summary': json.loads((OUTPUTS / 'analysis' / 'summary.json').read_text()),
        'participants': records(df[['gender', 'age', 'age_range', 'time_spent_hours', 'score', 'score_pct']]),
        'coefficients': records(coefs),
        'group_sizes': records(group_sizes),
        'subgroup_models': records(metrics),
        'overall_model': records(pd.read_csv(OUTPUTS / 'models' / 'overall_model_metrics.csv'))[0],
    }
    out = ROOT / 'docs' / 'data.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    print('Wrote site data to', out)


if __name__ == '__main__':
    main()
