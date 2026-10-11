#!/usr/bin/env python3
"""Overall classifier: can time online, age and gender flag low scorers?

- Label `low_performance`: `score_pct <= LABEL_THRESH` (same as subgroup models).
- Logistic regression on `time_spent_hours`, `age` (scaled) and `gender` (one-hot).
- Stratified 5-fold CV metrics are written to `outputs/models/overall_model_metrics.csv`;
  the model refit on all data is saved to `models/overall_logistic.joblib`.

Run: python src/modeling.py
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CLEANED_CSV, LABEL_THRESH, MODELS, OUTPUTS, prepare_X


def build_pipeline():
    preproc = ColumnTransformer([
        ('num', StandardScaler(), ['time_spent_hours', 'age']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['gender']),
    ])
    return Pipeline([('pre', preproc), ('clf', LogisticRegression(max_iter=1000))])


def main():
    df = pd.read_csv(CLEANED_CSV)
    X = prepare_X(df).assign(gender=df['gender'])
    y = (df['score_pct'] <= LABEL_THRESH).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(build_pipeline(), X, y, cv=skf, method='predict')
    y_proba = cross_val_predict(build_pipeline(), X, y, cv=skf, method='predict_proba')[:, 1]
    metrics = {
        'n': int(len(y)),
        'pos': int(y.sum()),
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_proba),
    }
    out = OUTPUTS / 'models' / 'overall_model_metrics.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out, index=False)
    print('CV metrics:', {k: round(v, 3) for k, v in metrics.items()})

    MODELS.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline().fit(X, y)
    joblib.dump({'pipeline': pipe, 'label': 'low_performance', 'threshold': LABEL_THRESH},
                MODELS / 'overall_logistic.joblib')
    print('Saved model to', MODELS / 'overall_logistic.joblib')


if __name__ == '__main__':
    main()
