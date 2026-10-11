#!/usr/bin/env python3
"""Plot evaluation results for subgroup models.

Loads models from `models/subgroups/` and the cleaned dataset, runs stratified
CV predictions (same folds as `subgroup_modeling.py`) to compute ROC and PR curves, confusion
matrices, and saves coefficient plots for logistic models retrained on full
subgroup. Outputs are saved to `outputs/models/plots/{age}_{gender}/`.

Run: python src/plot_models.py
"""
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

from common import CLEANED_CSV, LABEL_THRESH, MODELS, OUTPUTS, age_to_range, prepare_X
from subgroup_modeling import cv_splitter, make_pipeline


MODEL_DIR = MODELS / 'subgroups'


def parse_model_filename(fname):
    # Expect pattern: subgroup_{age_label}_{gender}_logistic.joblib
    m = re.match(r'subgroup_(.+)_(Male|Female)_logistic\.joblib', fname)
    if not m:
        return None
    return m.group(1), m.group(2)


def plot_roc_pr_confusion(age_label, gender, X, y, save_dir):
    skf = cv_splitter(y)
    y_proba = cross_val_predict(make_pipeline(), X, y, cv=skf, method='predict_proba')[:, 1]
    y_pred = cross_val_predict(make_pipeline(), X, y, cv=skf, method='predict')

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr, tpr, roc_auc = [0, 1], [0, 1], float('nan')
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC: {age_label} {gender}')
    ax.legend()
    fig.savefig(save_dir / 'roc.png', dpi=150)
    plt.close(fig)

    # Precision-Recall
    try:
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = auc(recall, precision)
    except Exception:
        precision, recall, pr_auc = [1, 0], [0, 1], float('nan')
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall: {age_label} {gender}')
    ax.legend()
    fig.savefig(save_dir / 'pr.png', dpi=150)
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    ax.set_title(f'Confusion Matrix: {age_label} {gender}')
    fig.savefig(save_dir / 'confusion_matrix.png', dpi=150)
    plt.close(fig)

    return {'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc)}


def coef_plot(age_label, gender, X, y, save_dir):
    # Fit on full subgroup to get coefficients
    pipe = make_pipeline().fit(X, y)
    coef = pipe.named_steps['clf'].coef_.ravel()
    features = X.columns.tolist()
    fig, ax = plt.subplots()
    sns.barplot(x=coef, y=features, orient='h', ax=ax)
    ax.set_title(f'Logistic coefficients (trained on full subgroup): {age_label} {gender}')
    fig.savefig(save_dir / 'coefficients.png', dpi=150)
    plt.close(fig)


def main():
    cleaned = pd.read_csv(CLEANED_CSV)

    cleaned['age_range'] = cleaned['age'].apply(age_to_range)

    results = []
    if not MODEL_DIR.exists():
        print('No subgroup models directory found at', MODEL_DIR)
        return

    for fname in sorted(p.name for p in MODEL_DIR.iterdir()):
        parsed = parse_model_filename(fname)
        if parsed is None:
            continue
        age_label, gender = parsed
        sub = cleaned[(cleaned['age_range'] == age_label) & (cleaned['gender'].str.lower() == gender.lower())]
        if sub.empty:
            print('No data for', age_label, gender)
            continue

        X = prepare_X(sub)
        y = (sub['score_pct'] <= LABEL_THRESH).astype(int)

        save_dir = OUTPUTS / 'models' / 'plots' / f'{age_label}_{gender}'
        save_dir.mkdir(parents=True, exist_ok=True)

        print('Plotting for', age_label, gender, 'n=', len(sub))
        metrics = plot_roc_pr_confusion(age_label, gender, X, y, save_dir)
        coef_plot(age_label, gender, X, y, save_dir)

        metrics.update({'age_range': age_label, 'gender': gender, 'n': len(sub)})
        results.append(metrics)

    if results:
        dfm = pd.DataFrame(results)
        outpath = OUTPUTS / 'models' / 'plots' / 'subgroup_model_summary.csv'
        dfm.to_csv(outpath, index=False)
        print('Wrote subgroup model summary to', outpath)


if __name__ == '__main__':
    main()
