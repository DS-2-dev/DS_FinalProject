#!/usr/bin/env python3
"""Train models per age-range × gender subgroup.

Behavior:
- Runs the cleaning script to ensure `data/cleaned_dataset.csv` is up to date.
- Builds label `low_performance` where `score_pct <= 0.6` (consistent with earlier work).
- For each subgroup (age_range × gender) with at least `MIN_N` rows, trains a logistic
  regression using `time_spent_hours` and `age` as predictors.
- Evaluates with stratified 5-fold CV and saves metrics and trained model.

Outputs:
- `models/subgroup_{age}_{gender}_logistic.joblib`
- `outputs/models/subgroup_model_metrics.csv`

Run: python src/subgroup_modeling.py
"""
import os
import runpy
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MIN_N = 20  # minimum subgroup size to train on (tunable)
LABEL_THRESH = 0.6  # score_pct <= this -> low_performance


def make_label(df):
    return (df['score_pct'] <= LABEL_THRESH).astype(int)


def prepare_X(df):
    X = df[['time_spent_hours', 'age']].copy()
    X['time_spent_hours'] = X['time_spent_hours'].fillna(X['time_spent_hours'].median())
    X['age'] = X['age'].fillna(X['age'].median())
    return X


def eval_model(estimator, X, y, cv=5):
    # Adjust number of splits to the smallest class count (StratifiedKFold requirement)
    class_counts = np.array(list(pd.Series(y).value_counts()))
    min_class_count = int(class_counts.min()) if len(class_counts) > 0 else 0

    if min_class_count < 2:
        # Not enough members in one of the classes for cross-val; fallback to training-set metrics
        estimator.fit(X, y)
        y_pred = estimator.predict(X)
        try:
            y_proba = estimator.predict_proba(X)[:, 1]
        except Exception:
            y_proba = np.zeros(len(y))
        try:
            roc = float(roc_auc_score(y, y_proba))
        except Exception:
            roc = float('nan')
        return {
            'n': int(len(y)),
            'pos': int(y.sum()),
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1': float(f1_score(y, y_pred, zero_division=0)),
            'roc_auc': roc,
            'cv_fallback': True,
        }

    n_splits = min(cv, min_class_count)
    if n_splits < 2:
        # cannot perform stratified CV
        return {
            'n': int(len(y)),
            'pos': int(y.sum()),
            'accuracy': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'roc_auc': float('nan'),
            'cv_fallback': True,
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(estimator, X, y, cv=skf, method='predict')
    y_proba = cross_val_predict(estimator, X, y, cv=skf, method='predict_proba')[:, 1]
    try:
        roc_auc = float(roc_auc_score(y, y_proba))
    except Exception:
        roc_auc = float('nan')
    return {
        'n': int(len(y)),
        'pos': int(y.sum()),
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'roc_auc': roc_auc,
        'cv_fallback': False,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train subgroup models')
    parser.add_argument('--min-n', type=int, default=MIN_N, help='Minimum subgroup size to train')
    args = parser.parse_args()
    min_n = int(args.min_n)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Ensure cleaned dataset is current
    runpy.run_path(os.path.join(root, 'src', 'cleaning.py'))

    df = pd.read_csv(os.path.join(root, 'data', 'cleaned_dataset.csv'))
    # recreate age_range mapping used earlier
    def age_to_range(a):
        try:
            a = float(a)
        except Exception:
            return None
        if a <= 10:
            return '0-10'
        if 10 < a <= 15:
            return '10-15'
        if 15 < a <= 20:
            return '16-20'
        if 20 < a <= 25:
            return '20-25'
        if 25 < a <= 30:
            return '26-30'
        if 30 < a <= 40:
            return '31-40'
        if 40 < a <= 50:
            return '40-50'
        if 50 < a <= 60:
            return '51-60'
        if 60 < a <= 80:
            return '61-80'
        return None

    df['age_range'] = df['age'].apply(age_to_range)
    df['low_performance'] = make_label(df)

    models_dir = os.path.join(root, 'models', 'subgroups')
    os.makedirs(models_dir, exist_ok=True)

    metrics = []

    genders = ['Male', 'Female']
    age_labels = sorted(df['age_range'].dropna().unique())
    for age_label in age_labels:
        for gender in genders:
            sub = df[(df['age_range'] == age_label) & (df['gender'].str.lower() == gender.lower())]
            if len(sub) < min_n:
                print(f"Skipping {age_label} {gender}: n={len(sub)} < min_n={min_n}")
                continue

            X = prepare_X(sub)
            y = sub['low_performance']

            # require both classes present to train meaningfully
            if y.nunique() < 2:
                print(f"Skipping {age_label} {gender}: only one class present (pos={int(y.sum())})")
                continue

            pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500))])
            m = eval_model(pipe, X, y, cv=5)
            m.update({'age_range': age_label, 'gender': gender})
            metrics.append(m)

            # retrain on full subgroup and save model
            pipe.fit(X, y)
            model_path = os.path.join(models_dir, f'subgroup_{age_label}_{gender}_logistic.joblib')
            joblib.dump({'pipeline': pipe, 'age_range': age_label, 'gender': gender, 'label': 'low_performance'}, model_path)
            print(f"Trained and saved model for {age_label} {gender} -> {model_path}")

    # Save metrics
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        out_metrics = os.path.join(root, 'outputs', 'models', 'subgroup_model_metrics.csv')
        os.makedirs(os.path.dirname(out_metrics), exist_ok=True)
        metrics_df.to_csv(out_metrics, index=False)
        print('Wrote metrics to', out_metrics)
    else:
        print('No subgroup models were trained (no groups met MIN_N)')


if __name__ == '__main__':
    main()
