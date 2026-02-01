#!/usr/bin/env python3
"""Aggregate all evaluation results into a single CSV."""
import pandas as pd
import glob
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "eval_results"

def main():
    csv_files = list(RESULTS_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} result files")
    
    if not csv_files:
        print("No results found!")
        return
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not dfs:
        print("No valid results!")
        return
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(['Task', 'Variant', 'Seed'])
    
    output = RESULTS_DIR / "all_results.csv"
    combined.to_csv(output, index=False)
    
    print(f"\n{'='*80}")
    print("AGGREGATED RESULTS")
    print('='*80)
    print(combined.to_string(index=False))
    print(f"\nSaved to: {output}")
    
    # Summary stats by variant
    print(f"\n{'='*80}")
    print("SUMMARY BY VARIANT (mean ± std)")
    print('='*80)
    
    summary = combined.groupby(['Task', 'Variant']).agg({
        'MeanReward': ['mean', 'std', 'count']
    }).round(2)
    summary.columns = ['Mean', 'Std', 'Count']
    print(summary)

if __name__ == '__main__':
    main()
