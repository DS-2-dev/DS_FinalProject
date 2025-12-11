#!/usr/bin/env python3
"""Train and evaluate classifiers to predict an 'impaired' label.

This script:
- Loads `data/cleaned_dataset.csv` produced by `src/cleaning.py`.
- Builds candidate binary labels for 'impaired' using configurable thresholds.
- Trains Logistic Regression and RandomForest classifiers with stratified CV.
- Reports metrics (accuracy, precision, recall, f1, ROC AUC) and saves the best model.

Run: python src/modeling.py
"""
import os
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_label(df, method='threshold', threshold=0.5):
    """Create a binary label 'impaired' using method.

    Methods:
    - 'threshold': impaired if score_pct <= threshold (default 0.5)
    - 'percentile': impaired if score_pct <= given percentile (0-100)
    """
    s = df['score_pct']
    if method == 'threshold':
        return (s <= threshold).astype(int)
    elif method == 'percentile':
        pct = np.nanpercentile(s.dropna(), threshold)
        return (s <= pct).astype(int)
    else:
        raise ValueError('Unknown method')


def prepare_features(df):
    df2 = df.copy()
    # Basic features: time_spent_hours, age, gender
    X = df2[['time_spent_hours', 'age', 'gender']].copy()
    # Fill missing numeric with median
    X['time_spent_hours'] = X['time_spent_hours'].fillna(X['time_spent_hours'].median())
    X['age'] = X['age'].fillna(X['age'].median())
    return X


def evaluate_model(estimator, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    y_pred = cross_val_predict(estimator, X, y, cv=skf, method='predict')
    y_proba = cross_val_predict(estimator, X, y, cv=skf, method='predict_proba')[:, 1]

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba),
    }
    return metrics, y_pred, y_proba


def build_pipelines():
    numeric_features = ['time_spent_hours', 'age']
    cat_features = ['gender']

    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    cat_transformer = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])

    preproc = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', cat_transformer, cat_features),
    ])

    pipe_lr = Pipeline([
        ('pre', preproc),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    pipe_rf = Pipeline([
        ('pre', preproc),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    return {'logistic': pipe_lr, 'random_forest': pipe_rf}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'data', 'cleaned_dataset.csv')
    df = pd.read_csv(path)

    # Candidate label definitions to try
    label_defs = [
        ('threshold_0.5', ('threshold', 0.5)),
        ('threshold_0.6', ('threshold', 0.6)),
        ('bottom_quartile', ('percentile', 25)),
    ]

    results = {}
    pipelines = build_pipelines()

    for name, (method, param) in label_defs:
        print(f'\nEvaluating label: {name} (method={method}, param={param})')
        y = make_label(df, method=method, threshold=param)
        X = prepare_features(df)

        # check class balance
        counts = y.value_counts().to_dict()
        print('Label counts:', counts)

        # skip if too few positives
        if counts.get(1, 0) < 5:
            print('Too few positive cases, skipping this label.')
            continue

        for model_name, pipe in pipelines.items():
            print(' Training', model_name)
            metrics, y_pred, y_proba = evaluate_model(pipe, X, y, cv=5)
            print('  metrics:', metrics)
            results[(name, model_name)] = {'metrics': metrics, 'y_pred': y_pred, 'y_proba': y_proba}

    # choose best by roc_auc
    best = None
    best_auc = -1
    for k, v in results.items():
        auc_val = v['metrics']['roc_auc']
        if auc_val > best_auc:
            best_auc = auc_val
            best = k

    if best is not None:
        print('\nBest model:', best, 'AUC=', best_auc)
        # retrain best on full data and save
        label_name, model_name = best
        # Reconstruct the label used for best
        if label_name.startswith('threshold_'):
            try:
                th = float(label_name.split('_', 1)[1])
            except Exception:
                th = 0.5
            y_full = make_label(df, method='threshold', threshold=th)
        elif label_name == 'bottom_quartile' or label_name.startswith('percentile'):
            y_full = make_label(df, method='percentile', threshold=25)
        else:
            y_full = make_label(df, method='threshold', threshold=0.5)

        X_full = prepare_features(df)
        best_pipe = pipelines[model_name]
        best_pipe.fit(X_full, y_full)

        model_dir = os.path.join(root, 'models')
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, f'best_model_{label_name}_{model_name}.joblib')
        joblib.dump({'pipeline': best_pipe, 'label_name': label_name, 'model_name': model_name}, save_path)
        print('Saved best model to', save_path)
    else:
        print('No valid model results to save.')


if __name__ == '__main__':
    main()
