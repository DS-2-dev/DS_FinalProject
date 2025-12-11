#!/usr/bin/env python3
"""Generate EDA plots from the cleaned dataset and save to outputs/plots.

Produces:
- histogram of `score_pct`
- histogram of `time_spent_hours`
- boxplot of `score_pct` by `gender`
- scatter `time_spent_hours` vs `score_pct` with regression line
- correlation heatmap for numeric features
- pairwise scatter matrix for numeric features (saved as PNG)
- interactive Plotly scatter HTML for exploration

Run: python src/plotting.py
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.express as px
except Exception:
    px = None


def ensure_outdir():
    outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'plots')
    os.makedirs(outdir, exist_ok=True)
    return outdir


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'data', 'cleaned_dataset.csv')
    df = pd.read_csv(path)

    outdir = ensure_outdir()

    # Numeric features to consider
    numeric_cols = ['score_pct', 'time_spent_hours', 'age']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # 1) Histogram of score_pct
    fig, ax = plt.subplots()
    sns.histplot(df['score_pct'].dropna(), kde=True, ax=ax)
    ax.set_title('Score proportion distribution')
    savefig(fig, os.path.join(outdir, 'hist_score_pct.png'))

    # 2) Histogram of time_spent_hours
    if 'time_spent_hours' in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df['time_spent_hours'].dropna(), kde=True, ax=ax)
        ax.set_title('Time spent (hours per week) distribution')
        savefig(fig, os.path.join(outdir, 'hist_time_spent.png'))

    # 3) Boxplot score by gender
    if 'gender' in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x='gender', y='score_pct', data=df, ax=ax)
        ax.set_title('Score by gender')
        savefig(fig, os.path.join(outdir, 'box_score_by_gender.png'))

    # 4) Scatter with regression
    if 'time_spent_hours' in df.columns:
        fig, ax = plt.subplots()
        sns.regplot(x='time_spent_hours', y='score_pct', data=df, scatter_kws={'alpha':0.6}, ax=ax)
        ax.set_title('Time spent vs score')
        savefig(fig, os.path.join(outdir, 'scatter_time_vs_score.png'))

    # 5) Correlation heatmap for numeric
    if numeric_cols:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu', center=0, ax=ax)
        ax.set_title('Correlation matrix (numeric)')
        savefig(fig, os.path.join(outdir, 'corr_numeric.png'))

    # 6) Pairwise scatter matrix (pairplot) - limit to at most 6 columns
    pair_cols = numeric_cols[:6]
    if len(pair_cols) >= 2:
        pp = sns.pairplot(df[pair_cols].dropna(), diag_kind='kde', plot_kws={'alpha': 0.6})
        pp.fig.suptitle('Pairwise relationships', y=1.02)
        pairpath = os.path.join(outdir, 'pairplot_numeric.png')
        pp.savefig(pairpath, dpi=200)
        plt.close('all')

    # 7) Interactive plotly scatter (.html)
    if px is not None and 'time_spent_hours' in df.columns:
        fig = px.scatter(df, x='time_spent_hours', y='score_pct', color='gender' if 'gender' in df.columns else None,
                         hover_data=['age','participant_id'] if 'participant_id' in df.columns else None,
                         title='Interactive: Time spent vs score')
        html_path = os.path.join(outdir, 'interactive_time_vs_score.html')
        fig.write_html(html_path)

    print('Saved plots to', outdir)


if __name__ == '__main__':
    main()
