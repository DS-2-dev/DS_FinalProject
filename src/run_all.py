#!/usr/bin/env python3
"""Regenerate every data file, model and figure from the raw survey CSV.

Run from anywhere: python src/run_all.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    'cleaning.py',
    'analysis.py',
    'plotting.py',
    'subgroup_analysis.py',
    'modeling.py',
    'subgroup_modeling.py',
    'plot_models.py',
    'build_site.py',
]

for script in STEPS:
    print(f'\n=== {script}', flush=True)
    subprocess.run([sys.executable, str(SRC / script)], check=True)
