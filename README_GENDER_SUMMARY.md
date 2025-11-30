Gender summary — README

This project now includes a gender-aggregated summary (notebook `notebooks/summary_2.ipynb`).

Files created/used
- `outputs/models/gender_summary_metrics.csv`: Aggregated metrics by `gender` (weighted averages by subgroup `n`). Columns include `gender`, `n`, `pos`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`.
- `outputs/models/plots/gender/{gender}/bubble.png`: Bubble plot (time spent online vs score) per gender. Bubble size ∝ participant age.

Purpose and caveats
- These gender-level metrics are descriptive summaries built from subgroup results (age×gender subgroups). They are NOT causal estimates.
- Small subgroups (n < 10) are unstable and can bias aggregated metrics. Prefer `n >= 10` when making descriptive comparisons; treat smaller groups as exploratory only.

How to regenerate
From the repository root (use the venv Python at `.venv`):

```bash
/workspaces/DS_FinalProject/.venv/bin/python src/cleaning.py
/workspaces/DS_FinalProject/.venv/bin/python src/subgroup_modeling.py --min-n 1
/workspaces/DS_FinalProject/.venv/bin/python src/plot_models.py
# Then open and run the notebook:
/workspaces/DS_FinalProject/.venv/bin/python -m nbconvert --to notebook --execute notebooks/summary_2.ipynb --output summary_2_executed.ipynb --output-dir notebooks
```

Notes
- The notebook writes `outputs/models/gender_summary_metrics.csv` when run; it also saves per-gender bubble plots to `outputs/models/plots/gender/`.
- If you change age binning or subgroup logic, re-run the subgroup modeling and plotting scripts before re-executing this notebook.
