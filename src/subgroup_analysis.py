#!/usr/bin/env python3
"""Split data by age-range and gender, produce subgroup plots, and save subset CSVs.

Creates folders under `outputs/plots/age_gender/{age_range}/{gender}/` with:
- histogram of `score_pct`
- histogram of `time_spent_hours`
- scatter `time_spent_hours` vs `score_pct`
- boxplot `score_pct`

Also writes subset CSVs to `outputs/subsets/{age_range}_{gender}.csv`.

Run: python src/subgroup_analysis.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import CLEANED_CSV, GENDERS, OUTPUTS, age_to_range


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_subgroup_plots(df_sub: pd.DataFrame, outdir: str):
    # Score histogram
    if 'score_pct' in df_sub.columns:
        fig, ax = plt.subplots()
        sns.histplot(df_sub['score_pct'].dropna(), kde=True, ax=ax)
        ax.set_title('Score proportion distribution')
        save_plot(fig, os.path.join(outdir, 'hist_score_pct.png'))

    # Time histogram
    if 'time_spent_hours' in df_sub.columns:
        fig, ax = plt.subplots()
        sns.histplot(df_sub['time_spent_hours'].dropna(), kde=True, ax=ax)
        ax.set_title('Time spent (hours/week) distribution')
        save_plot(fig, os.path.join(outdir, 'hist_time_spent.png'))

    # Scatter
    if 'time_spent_hours' in df_sub.columns and 'score_pct' in df_sub.columns:
        fig, ax = plt.subplots()
        sns.regplot(x='time_spent_hours', y='score_pct', data=df_sub, scatter_kws={'alpha':0.6}, ax=ax)
        ax.set_title('Time spent vs score')
        save_plot(fig, os.path.join(outdir, 'scatter_time_vs_score.png'))

    # Boxplot score
    if 'score_pct' in df_sub.columns:
        fig, ax = plt.subplots()
        sns.boxplot(y='score_pct', data=df_sub, ax=ax)
        ax.set_title('Score distribution (boxplot)')
        save_plot(fig, os.path.join(outdir, 'box_score.png'))


def main():
    df = pd.read_csv(CLEANED_CSV)

    # Create age_range column
    df['age_range'] = df['age'].apply(age_to_range)

    out_base = str(OUTPUTS)
    plots_base = os.path.join(out_base, 'plots', 'age_gender')
    subsets_base = os.path.join(out_base, 'subsets')
    ensure_dir(plots_base)
    ensure_dir(subsets_base)

    # For each age_range and gender, save subset CSV and plots
    for age_label in sorted(df['age_range'].dropna().unique(), key=lambda x: x):
        for gender in GENDERS:
            mask = (df['age_range'] == age_label) & (df['gender'].str.lower() == gender.lower())
            sub = df[mask]
            # Skip empty
            if sub.empty:
                continue

            outdir = os.path.join(plots_base, age_label.replace(' ', '_'), gender)
            ensure_dir(outdir)

            # Save subset CSV
            csv_path = os.path.join(subsets_base, f'{age_label.replace(" ", "_")}_{gender}.csv')
            sub.to_csv(csv_path, index=False)

            # Make plots for subgroup
            make_subgroup_plots(sub, outdir)

    print('Wrote subgroup plots to', plots_base)
    print('Wrote subset CSVs to', subsets_base)


if __name__ == '__main__':
    main()
