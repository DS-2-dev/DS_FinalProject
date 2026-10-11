"""Shared paths and helpers used by the pipeline scripts."""
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / 'data' / 'raw' / 'survey_responses.csv'
CLEANED_CSV = ROOT / 'data' / 'cleaned_dataset.csv'
OUTPUTS = ROOT / 'outputs'
MODELS = ROOT / 'models'

LABEL_THRESH = 0.6  # score_pct <= this -> low_performance
GENDERS = ['Male', 'Female']


def age_to_range(age):
    """Map numeric age to an age-range label (upper bounds inclusive)."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    if math.isnan(a):
        return None
    if a <= 10:
        return '0-10'
    if a <= 15:
        return '10-15'
    if a <= 20:
        return '16-20'
    if a <= 25:
        return '20-25'
    if a <= 30:
        return '26-30'
    if a <= 40:
        return '31-40'
    if a <= 50:
        return '40-50'
    if a <= 60:
        return '51-60'
    if a <= 80:
        return '61-80'
    return None


def prepare_X(df):
    """Numeric predictors with median fill."""
    X = df[['time_spent_hours', 'age']].copy()
    X['time_spent_hours'] = X['time_spent_hours'].fillna(X['time_spent_hours'].median())
    X['age'] = X['age'].fillna(X['age'].median())
    return X
