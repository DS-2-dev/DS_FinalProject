#!/usr/bin/env python3
"""Create a single interactive HTML dashboard summarizing all main features.

Generates an HTML file with multiple Plotly figures: histograms, boxplot,
scatter, correlation heatmap, and parallel coordinates for numeric features.

Run: python src/dashboard.py
Output: outputs/plots/dashboard.html
"""
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np


def make_figures(df):
    figs = {}

    # Ensure numeric columns
    numeric_cols = [c for c in ['score_pct', 'time_spent_hours', 'age'] if c in df.columns]

    # Histogram: score_pct
    if 'score_pct' in df.columns:
        figs['hist_score'] = px.histogram(df, x='score_pct', nbins=20, title='Score proportion distribution')

    # Histogram: time_spent_hours
    if 'time_spent_hours' in df.columns:
        figs['hist_time'] = px.histogram(df, x='time_spent_hours', nbins=30, title='Time spent (hours/week)')

    # Boxplot: score by gender
    if 'gender' in df.columns and 'score_pct' in df.columns:
        # Plotly uses 'suspectedoutliers' as the keyword
        figs['box_gender'] = px.box(df, x='gender', y='score_pct', points='suspectedoutliers', title='Score by gender')

    # Scatter: time vs score colored by gender
    if 'time_spent_hours' in df.columns and 'score_pct' in df.columns:
        color = 'gender' if 'gender' in df.columns else None
        figs['scatter_time_score'] = px.scatter(df, x='time_spent_hours', y='score_pct', color=color,
                                               hover_data=[c for c in ['age','participant_id'] if c in df.columns],
                                               title='Time spent vs Score (interactive)')

    # Correlation heatmap for numeric
    if numeric_cols:
        corr = df[numeric_cols].corr()
        figs['corr'] = px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1, title='Correlation matrix (numeric)')

    # Parallel coordinates for numeric features
    if len(numeric_cols) >= 2:
        # drop rows with NaN in these cols
        pc_df = df[numeric_cols].dropna()
        # normalize numeric columns for better visualization
        norm = (pc_df - pc_df.min()) / (pc_df.max() - pc_df.min())
        norm.columns = pc_df.columns
        figs['parallel'] = px.parallel_coordinates(pc_df, color=numeric_cols[0], labels={c: c for c in pc_df.columns},
                                                  title='Parallel coordinates (numeric)')

    return figs


def assemble_html(figs, outpath):
    # Build HTML with CDN plotly, stacking figure divs
    fragments = []
    for key, fig in figs.items():
        fragments.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # Full HTML wrapper with Plotly JS from CDN
    html = """
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <title>EDA Dashboard</title>
</head>
<body>
  <h1>EDA Dashboard</h1>
"""

    for frag in fragments:
        html += '<div style="margin-bottom:40px">' + frag + '</div>\n'

    html += '</body>\n</html>'

    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'data', 'cleaned_dataset.csv')
    df = pd.read_csv(path)

    figs = make_figures(df)
    outdir = os.path.join(root, 'outputs', 'plots')
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'dashboard.html')
    assemble_html(figs, outpath)
    print('Wrote dashboard to', outpath)


if __name__ == '__main__':
    main()
