#!/usr/bin/env python3
"""Train a logistic model per age-range × gender subgroup.

- Label `low_performance`: `score_pct <= LABEL_THRESH`.
- Predictors: `time_spent_hours`, `age`.
- Only subgroups with at least `--min-n` rows (default `MIN_GROUP_N`) and at
  least 2 members in each class are modeled, so stratified CV is always possible.

Outputs:
- `models/subgroups/subgroup_{age}_{gender}_logistic.joblib`
- `outputs/models/subgroup_model_metrics.csv`

Run: python src/subgroup_modeling.py
"""
import argparse

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (AGE_ORDER, CLEANED_CSV, GENDERS, LABEL_THRESH, MIN_GROUP_N, MODELS,
                    OUTPUTS, age_to_range, prepare_X)


def make_pipeline():
    return Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500))])


def cv_splitter(y, max_splits=5):
    """Stratified folds, capped by the size of the smaller class."""
    return StratifiedKFold(n_splits=min(max_splits, int(y.value_counts().min())), shuffle=True, random_state=42)


def eval_model(X, y):
    skf = cv_splitter(y)
    y_pred = cross_val_predict(make_pipeline(), X, y, cv=skf, method='predict')
    y_proba = cross_val_predict(make_pipeline(), X, y, cv=skf, method='predict_proba')[:, 1]
    return {
        'n': int(len(y)),
        'pos': int(y.sum()),
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, y_proba)),
    }


def main():
    parser = argparse.ArgumentParser(description='Train subgroup models')
    parser.add_argument('--min-n', type=int, default=MIN_GROUP_N, help='Minimum subgroup size to train')
    args = parser.parse_args()

    df = pd.read_csv(CLEANED_CSV)
    df['age_range'] = df['age'].apply(age_to_range)
    df['low_performance'] = (df['score_pct'] <= LABEL_THRESH).astype(int)

    models_dir = MODELS / 'subgroups'
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for age_label in AGE_ORDER:
        for gender in GENDERS:
            sub = df[(df['age_range'] == age_label) & (df['gender'] == gender)]
            y = sub['low_performance']
            if len(sub) < args.min_n:
                continue
            if y.nunique() < 2 or y.value_counts().min() < 2:
                print(f'Skipping {age_label} {gender}: too few of one class (n={len(sub)}, pos={int(y.sum())})')
                continue

            X = prepare_X(sub)
            m = eval_model(X, y)
            m.update({'age_range': age_label, 'gender': gender})
            metrics.append(m)

            pipe = make_pipeline().fit(X, y)
            model_path = models_dir / f'subgroup_{age_label}_{gender}_logistic.joblib'
            joblib.dump({'pipeline': pipe, 'age_range': age_label, 'gender': gender, 'label': 'low_performance'}, model_path)
            print(f'Trained {age_label} {gender} (n={len(sub)}, ROC AUC={m["roc_auc"]:.2f})')

    out_metrics = OUTPUTS / 'models' / 'subgroup_model_metrics.csv'
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(out_metrics, index=False)
    print('Wrote metrics to', out_metrics)


if __name__ == '__main__':
    main()
