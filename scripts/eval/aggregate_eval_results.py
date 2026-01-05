#!/usr/bin/env python3
"""Aggregate all evaluation CSV results into a single master CSV using standard libraries."""
import os
import glob
import csv
import statistics
from pathlib import Path

RESULTS_DIR = Path("/storage/scratch1/9/eliu354/flow_mbpo/eval_results")
OUTPUT_CSV = RESULTS_DIR / "final_eval_results.csv"

def main():
    if not RESULTS_DIR.exists():
        print(f"Results directory {RESULTS_DIR} does not exist.")
        return

    csv_files = list(RESULTS_DIR.glob("eval_*.csv"))
    print(f"Found {len(csv_files)} evaluation CSVs.")

    all_results = []
    
    for f in csv_files:
        try:
            with open(f, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                # Check for empty file
                rows = list(reader)
                if not rows:
                    continue
                
                # Assume columns: episode_reward, episode_length
                rewards = []
                lengths = []
                for row in rows:
                    if 'episode_reward' in row:
                        rewards.append(float(row['episode_reward']))
                    if 'episode_length' in row:
                        lengths.append(float(row['episode_length']))
                
                if rewards:
                    mean_reward = statistics.mean(rewards)
                    mean_length = statistics.mean(lengths) if lengths else 0
                    
                    # Extract checkpoint ID from filename
                    filename = f.name
                    ckpt_id = filename.replace('eval_', '').replace('.csv', '')
                    
                    # Try to parse Task/Variant/Seed from ckpt_id or path if possible
                    # But for now just store ID and metrics
                    
                    all_results.append({
                        'CheckpointID': ckpt_id,
                        'MeanReward': mean_reward,
                        'MeanLength': mean_length,
                        'Path': str(f)
                    })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Write aggregated results
    if all_results:
        # Sort by CheckpointID
        all_results.sort(key=lambda x: x['CheckpointID'])
        
        with open(OUTPUT_CSV, 'w') as csvfile:
            fieldnames = ['CheckpointID', 'MeanReward', 'MeanLength', 'Path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        
        print(f"\nAggregated results written to {OUTPUT_CSV}")
        print(f"{'='*80}")
        print(f"{'CheckpointID':<40} | {'Reward':<10} | {'Length':<10}")
        print(f"{'-'*80}")
        for r in all_results:
            print(f"{r['CheckpointID']:<40} | {r['MeanReward']:.2f}       | {r['MeanLength']:.2f}")
        print(f"{'='*80}")
    else:
        print("No results to aggregate.")

if __name__ == "__main__":
    main()
