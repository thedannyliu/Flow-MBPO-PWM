#!/usr/bin/env python3
"""Aggregate all evaluation CSV results into a single master CSV."""
import os, glob, pandas as pd
from pathlib import Path

RESULTS_DIR = Path("/storage/scratch1/9/eliu354/flow_mbpo/eval_results")

def main():
    csv_files = list(RESULTS_DIR.glob("eval_*.csv"))
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
    
    # Remove duplicates (keep best result per Task/Variant/Seed)
    combined = combined.drop_duplicates(subset=['Task', 'Variant', 'Seed'], keep='first')
    
    output = RESULTS_DIR / "final_eval_results.csv"
    combined.to_csv(output, index=False)
    
    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION RESULTS ({len(combined)} experiments)")
    print('='*80)
    print(combined.to_string(index=False))
    print(f"\nSaved to: {output}")
    
    # Summary stats by variant
    print(f"\n{'='*80}")
    print("SUMMARY BY TASK AND VARIANT")
    print('='*80)
    
    summary = combined.groupby(['Task', 'Variant']).agg({
        'MeanReward': ['mean', 'std', 'count'],
        'MeanLength': 'mean'
    }).round(2)
    summary.columns = ['AvgReward', 'StdReward', 'NumSeeds', 'AvgLength']
    print(summary)

if __name__ == '__main__':
    main()
