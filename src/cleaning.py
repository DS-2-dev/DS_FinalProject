#!/usr/bin/env python3
"""Cleaning utilities for the survey CSV.

Reads the Google Forms export CSV, normalizes columns, parses the score and
time-spent fields, anonymizes names into `participant_id`, and writes a
cleaned CSV to `data/cleaned_dataset.csv`. Missing `time_spent_hours` values
are imputed with the sample median.

Run: python src/cleaning.py
"""
import hashlib
import re
from statistics import mean

import pandas as pd

from common import CLEANED_CSV, RAW_CSV


def parse_score(score_raw):
    """Extract numeric score from strings like '8 / 20' or '12 / 20'."""
    if pd.isna(score_raw):
        return None
    s = str(score_raw)
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return None


def parse_time_hours(s):
    """Parse a messy time string into hours (weekly basis where possible).

    Heuristics implemented:
    - If a range like '20-30' or '20-30 hours' -> take the mean
    - If contains 'day' or 'per day' -> multiply numeric by 7 (to weekly)
    - If contains 'minute' or 'min' -> convert minutes to hours (/60)
    - If contains 'hr' or 'hour' or 'hours' or contains only a number -> use as hours
    - If contains ambiguous text, attempt to extract numbers and use first or mean of range
    Returns float (hours) or None if parsing failed.
    """
    if pd.isna(s):
        return None
    raw = str(s).strip()
    raw = raw.replace('\u200b', '')
    # find all numbers (ints or decimals)
    nums = re.findall(r"\d+(?:\.\d+)?", raw)
    if not nums:
        return None

    # handle ranges like '20-30' or '30-35'
    if re.search(r"\d+\s*[-–—]\s*\d+", raw):
        numbers = [float(n) for n in nums[:2]]
        val = mean(numbers)
    else:
        val = float(nums[0])

    low = raw.lower()
    # minutes -> hours
    if 'min' in low or 'minute' in low:
        # assume the number is minutes (weekly or per instance unclear); treat as hours
        val = val / 60.0
    # per day -> convert to weekly
    if 'day' in low or 'per day' in low or 'a day' in low:
        val = val * 7.0
    # if it explicitly mentions 'week' or 'weekly' or 'per week' we keep as-is
    # if it contains 'hr' or 'hour' or 'hours' keep as hours

    return float(val)


def anonymize_name(name):
    if pd.isna(name):
        return None
    h = hashlib.sha256(str(name).strip().encode('utf-8')).hexdigest()
    return h[:12]


def clean_dataframe(df):
    # standardize column names
    df = df.rename(columns=lambda s: str(s).strip())

    # map known columns
    # Score column sometimes 'Score' with value '8 / 20'
    if 'Score' in df.columns:
        df['score_raw'] = df['Score']
    elif 'score' in df.columns:
        df['score_raw'] = df['score']

    # Name column
    name_cols = [c for c in df.columns if 'name' in c.lower()]
    if name_cols:
        df['name_raw'] = df[name_cols[0]]

    # Age column
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    if age_cols:
        df['age_raw'] = df[age_cols[0]]

    # Time spent column
    time_cols = [c for c in df.columns if 'hour' in c.lower() or 'internet' in c.lower() or 'how many hours' in c.lower()]
    if not time_cols:
        # fallback: look for numeric-like columns besides score
        time_cols = [c for c in df.columns if 'time' in c.lower() or 'hour' in c.lower()]
    if time_cols:
        df['time_raw'] = df[time_cols[0]]

    # Gender
    gender_cols = [c for c in df.columns if 'gender' in c.lower()]
    if gender_cols:
        df['gender_raw'] = df[gender_cols[0]]

    # Parse score
    df['score'] = df['score_raw'].apply(parse_score)
    df['score_pct'] = df['score'].apply(lambda x: float(x) / 20.0 if x is not None else None)

    # Parse time
    df['time_spent_hours'] = df['time_raw'].apply(parse_time_hours)

    # Anonymize
    df['participant_id'] = df['name_raw'].apply(anonymize_name)

    # Parse age to numeric
    if 'age_raw' in df.columns:
        df['age'] = pd.to_numeric(df['age_raw'], errors='coerce')

    # Keep relevant columns (include age)
    out_cols = ['participant_id', 'gender_raw', 'age', 'time_spent_hours', 'score', 'score_pct']
    existing = [c for c in out_cols if c in df.columns]
    cleaned = df[existing].copy()
    cleaned = cleaned.rename(columns={'gender_raw': 'gender'})

    return cleaned


def main():
    if not RAW_CSV.exists():
        raise SystemExit(f'Survey CSV not found at {RAW_CSV}')

    df = pd.read_csv(RAW_CSV)
    cleaned = clean_dataframe(df)

    n_missing = int(cleaned['time_spent_hours'].isna().sum())
    if n_missing:
        median = cleaned['time_spent_hours'].median()
        cleaned['time_spent_hours'] = cleaned['time_spent_hours'].fillna(median)
        print(f'Imputed {n_missing} missing time_spent_hours with median={median}')

    CLEANED_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEANED_CSV, index=False)
    print(f'Wrote cleaned dataset to: {CLEANED_CSV} (n={len(cleaned)})')


if __name__ == '__main__':
    main()
