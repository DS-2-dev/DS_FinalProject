#!/usr/bin/env python3
"""Descriptive plots per age-range × gender subgroup and per gender.

For each subgroup with at least `MIN_GROUP_N` participants, writes to
`outputs/plots/age_gender/{age_range}_{gender}/`:
- `hist_score_pct.png`, `hist_time_spent.png`
- `bubble.png`: time online vs score, bubble size ∝ age, with linear fit

Also writes `outputs/plots/gender/{gender}_bubble.png` (with LOWESS smoother).

Run: python src/subgroup_analysis.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

from common import AGE_ORDER, CLEANED_CSV, GENDERS, MIN_GROUP_N, OUTPUTS, age_to_range

COLORS = {'Male': '#2457d6', 'Female': '#d9480f'}


def save_plot(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def histogram(values, title, path):
    fig, ax = plt.subplots()
    sns.histplot(values, kde=True, ax=ax)
    ax.set_title(title)
    save_plot(fig, path)


def bubble(sub, gender, title, path, smooth=False):
    x = sub['time_spent_hours'].to_numpy()
    y = sub['score_pct'].to_numpy()
    sizes = (sub['age'] - sub['age'].min() + 1) * 30

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=sizes, c=COLORS.get(gender, '#2ca02c'), alpha=0.7, edgecolors='w')
    if smooth and len(x) >= 3:
        sm = lowess(y, x, frac=0.5)
        ax.plot(sm[:, 0], sm[:, 1], color='black', linewidth=1.5, label='LOWESS')
    if len(x) >= 2:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, np.polyval(coef, xs), color='k', linestyle='--', linewidth=1, label='Linear fit')
        r2 = np.corrcoef(x, y)[0, 1] ** 2 if np.std(x) and np.std(y) else 0.0
        ax.text(0.99, 0.02, f'Slope={coef[0]:.4f}, R²={r2:.2f}', ha='right', va='bottom',
                transform=ax.transAxes, fontsize=8)
    ax.set_xlabel('Time spent online (hours/week)')
    ax.set_ylabel('Score (proportion correct)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if smooth:
        ax.legend(loc='lower right')
    fig.text(0.5, 0.005, 'Bubble size ∝ participant age.', ha='center', fontsize=9)
    save_plot(fig, path)


def main():
    df = pd.read_csv(CLEANED_CSV)
    df['age_range'] = df['age'].apply(age_to_range)
    base = OUTPUTS / 'plots'

    for age_label in AGE_ORDER:
        for gender in GENDERS:
            sub = df[(df['age_range'] == age_label) & (df['gender'] == gender)]
            if len(sub) < MIN_GROUP_N:
                continue
            outdir = base / 'age_gender' / f'{age_label}_{gender}'
            histogram(sub['score_pct'], f'Score distribution — {age_label} {gender}', outdir / 'hist_score_pct.png')
            histogram(sub['time_spent_hours'], f'Time online (hours/week) — {age_label} {gender}', outdir / 'hist_time_spent.png')
            bubble(sub, gender, f'{age_label} {gender} (n={len(sub)})', outdir / 'bubble.png')

    for gender in GENDERS:
        sub = df[df['gender'] == gender]
        bubble(sub, gender, f'{gender} (n={len(sub)})', base / 'gender' / f'{gender}_bubble.png', smooth=True)

    print('Wrote subgroup plots to', base)


if __name__ == '__main__':
    main()
