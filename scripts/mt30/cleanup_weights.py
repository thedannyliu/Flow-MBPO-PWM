import pandas as pd
import glob
import os
from pathlib import Path

def main():
    csv_file = "mt30_all_runs.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    completed = df[df["Status"] == "COMPLETED"]
    
    deleted_count = 0
    reclaimed_space = 0
    
    print(f"Found {len(completed)} completed runs. Checking for weights to delete...")

    for _, row in completed.iterrows():
        run_dir = Path(row["RunDir"]) / "logs"
        if not run_dir.exists():
            continue
            
        # Find all .pt files
        pt_files = list(run_dir.glob("*.pt"))
        
        for pt_file in pt_files:
            size = pt_file.stat().st_size
            try:
                os.remove(pt_file)
                deleted_count += 1
                reclaimed_space += size
                print(f"Deleted: {pt_file} ({size/1024/1024:.2f} MB)")
            except Exception as e:
                print(f"Failed to delete {pt_file}: {e}")

    print(f"\nCleanup Complete.")
    print(f"Deleted {deleted_count} files.")
    print(f"Reclaimed {reclaimed_space/1024/1024/1024:.2f} GB.")

if __name__ == "__main__":
    main()
