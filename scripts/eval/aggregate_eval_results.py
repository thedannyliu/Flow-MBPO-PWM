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
                
                # Check columns
                keys = rows[0].keys()
                
                if 'MeanReward' in keys:
                    # Summary format
                    mean_reward = float(rows[0]['MeanReward'])
                    mean_length = float(rows[0]['MeanLength']) if 'MeanLength' in keys else 0
                    
                    # Extract metadata if available
                    task = rows[0].get('Task', 'Unknown')
                    variant = rows[0].get('Variant', 'Unknown')
                    seed = rows[0].get('Seed', 'Unknown')
                    
                    # Construct ID from metadata if possible, else filename
                    if task != 'Unknown':
                        ckpt_id = f"{task}_{variant}_s{seed}"
                    else:
                        filename = f.name
                        ckpt_id = filename.replace('eval_', '').replace('.csv', '')

                    all_results.append({
                        'CheckpointID': ckpt_id,
                        'MeanReward': mean_reward,
                        'MeanLength': mean_length,
                        'Path': str(f)
                    })
                
                else: 
                    # Raw format (episode_reward) - legacy/fallback
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
                        
                        filename = f.name
                        ckpt_id = filename.replace('eval_', '').replace('.csv', '')
                        
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
