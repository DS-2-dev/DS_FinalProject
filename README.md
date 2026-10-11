# Does time online affect spotting AI images?

DS Final Project. Survey participants judged 20 images as AI-generated or human-made and reported how many hours a week they spend online. We tested whether internet exposure predicts their accuracy.

**Results site:** https://ds-2-dev.github.io/DS_FinalProject/ (source in `docs/`)

## Key findings (n = 102)

- **No meaningful link between time online and accuracy.** Pearson r = 0.03 (p = 0.76). In the regression, the time coefficient is about zero (95% CI −0.27 to +0.25 percentage points per hour, p = 0.95).
- Average accuracy was 58%, against 50% for chance. Median time online was 21 hours a week.
- Men scored about 4.8 percentage points higher than women (p = 0.046, borderline). Age showed no clear effect (p = 0.17).
- The model explains little variance (R² = 0.064).
- Subgroup (age × gender) classifiers perform near chance, and most groups have n < 10. Treat them as exploratory.

These results are observational and describe associations, not causes. Full write-up: `notebooks/findings.ipynb`.

## Repository layout

```
data/
  raw/survey_responses.csv   Google Forms export (input)
  cleaned_dataset.csv        cleaned, anonymized data (generated)
src/
  run_all.py                 runs the whole pipeline below
  common.py                  shared paths, age bins, helpers
  cleaning.py                parse raw CSV, hash names, impute missing hours
  analysis.py                OLS (HC3), correlations, mixed model -> outputs/analysis/
  plotting.py                EDA figures -> outputs/plots/
  dashboard.py               interactive EDA dashboard -> outputs/plots/dashboard.html
  subgroup_analysis.py       age × gender subsets and plots
  modeling.py                overall classifiers -> models/
  subgroup_modeling.py       per-subgroup logistic models -> models/subgroups/
  plot_models.py             ROC / PR / confusion / coefficient plots
  build_site.py              exports results to docs/data.json
notebooks/
  01_cleaning.ipynb          cleaning walkthrough
  02_modeling.ipynb          overall model training
  03_analysis.ipynb          EDA and regression
  04_summary.ipynb           subgroup metrics and bubble plots
  summary_2.ipynb            gender-level summary
  findings.ipynb             findings and conclusions
outputs/, models/            generated (rebuilt by run_all.py)
docs/                        GitHub Pages site
```

## Reproduce

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py
```

`run_all.py` regenerates `data/cleaned_dataset.csv`, `models/`, `outputs/` and `docs/data.json` from the raw CSV. It also executes `04_summary.ipynb` and `summary_2.ipynb`, which write the bubble plots and gender summary CSVs. Notebooks are committed without outputs; open them in Jupyter from the `notebooks/` folder.

## Method notes

- **Cleaning:** names are replaced by a 12-character SHA-256 hash. Hours are parsed from free text (ranges are averaged, per-day answers are multiplied by 7, minutes are converted to hours). One missing hours value is imputed with the median (21 h).
- **Primary model:** `score_pct ~ time_spent_hours + C(gender) + age`, OLS with HC3 robust standard errors.
- **Mixed model:** random intercept by gender. With only two groups the fit is singular, so it is a sensitivity check only.
- **Classifiers:** label = low performance (`score_pct <= 0.6`); predictors are hours and age (plus gender in the overall model). Evaluated with stratified CV; groups too small for CV fall back to in-sample metrics and are flagged as optimistic.

## Website

`docs/index.html` is a static page that reads `docs/data.json` and draws its charts with Plotly. To publish it, open repository **Settings → Pages**, choose *Deploy from a branch*, then select branch `main` and folder `/docs`. To preview it locally:

```bash
cd docs && python -m http.server
```
