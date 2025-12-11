"""
Reassign participant IDs to sequential integers for readability.
- Reads `data/cleaned_dataset.csv`.
- Adds `participant_old_id` with the previous value (if present).
- Creates `participant_id` as sequential integers starting from 1, ordered by existing row order (or timestamp if available).
- Writes updated CSVs:
  - `data/cleaned_dataset_with_old_ids.csv` (backup)
  - overwrites `data/cleaned_dataset.csv` with the new IDs
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path('/workspaces/DS_FinalProject')
DATA_PATH = REPO_ROOT / 'data' / 'cleaned_dataset.csv'
BACKUP_PATH = REPO_ROOT / 'data' / 'cleaned_dataset_with_old_ids.csv'

if not DATA_PATH.exists():
    raise SystemExit(f'Cleaned dataset not found at {DATA_PATH}')

print('Reading', DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Preserve original participant_id if exists
if 'participant_id' in df.columns:
    df['participant_old_id'] = df['participant_id'].astype(str)
else:
    df['participant_old_id'] = ''

# If timestamp exists, sort by it to keep "first respondent -> id 1" semantics
if 'timestamp' in df.columns:
    try:
        df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp_parsed').reset_index(drop=True)
        df.drop(columns=['timestamp_parsed'], inplace=True)
        print('Sorted by timestamp for sequential IDs')
    except Exception:
        print('Could not parse timestamp; using current row order')
else:
    print('No timestamp column found; using current row order')

# Assign sequential numeric IDs (1-based)
df['participant_id'] = (df.reset_index().index + 1).astype(int)

# Backup original CSV
df.to_csv(BACKUP_PATH, index=False)
print('Wrote backup with old IDs to', BACKUP_PATH)

# Overwrite cleaned dataset with new IDs
df.to_csv(DATA_PATH, index=False)
print('Overwrote cleaned dataset with sequential participant_id at', DATA_PATH)

print('\nExample head:')
print(df[['participant_id','participant_old_id']].head(10))
