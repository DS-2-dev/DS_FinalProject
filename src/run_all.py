#!/usr/bin/env python3
"""Regenerate every data file, model and figure from the raw survey CSV.

Run from anywhere: python src/run_all.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    ['cleaning.py'],
    ['analysis.py'],
    ['plotting.py'],
    ['dashboard.py'],
    ['subgroup_analysis.py'],
    ['modeling.py'],
    ['subgroup_modeling.py', '--min-n', '1'],
    ['plot_models.py'],
]
# Notebooks that write bubble plots and summary CSVs into outputs/models/
NOTEBOOKS = ['04_summary.ipynb', 'summary_2.ipynb']

for script, *args in STEPS:
    print(f'\n=== {script} {" ".join(args)}', flush=True)
    subprocess.run([sys.executable, str(SRC / script), *args], check=True)

for nb in NOTEBOOKS:
    print(f'\n=== notebooks/{nb}', flush=True)
    subprocess.run([sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute', '--stdout',
                    str(SRC.parent / 'notebooks' / nb)], check=True, stdout=subprocess.DEVNULL)

print('\n=== build_site.py', flush=True)
subprocess.run([sys.executable, str(SRC / 'build_site.py')], check=True)
