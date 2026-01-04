import glob
import os
from pathlib import Path
import re

def main():
    # Base directory to search (outputs directory)
    base_dir = Path("/home/hice1/eliu354/scratch/Projects/Flow-MBPO-PWM/outputs")
    
    print(f"Scanning {base_dir} for weight files...")
    
    deleted_count = 0
    reclaimed_space = 0
    kept_count = 0
    
    # Walk through all directories
    for root, dirs, files in os.walk(base_dir):
        pt_files = [f for f in files if f.endswith(".pt")]
        if not pt_files:
            continue
            
        root_path = Path(root)
        
        # Classify files
        model_best = None
        model_last = None
        numbered_models = []
        others = []
        
        for f in pt_files:
            if f == "model_best.pt":
                model_best = root_path / f
            elif f == "model_last.pt":
                model_last = root_path / f
            elif f == "model_final.pt":
                # Treat final as a "last" candidate if last doesn't exist
                others.append(root_path / f)
            else:
                # Check for model_{N}.pt
                match = re.match(r"model_(\d+)\.pt", f)
                if match:
                    numbered_models.append((int(match.group(1)), root_path / f))
                else:
                    others.append(root_path / f)
        
        # Determine what to keep
        keep_files = set()
        
        if model_best:
            keep_files.add(model_best)
            
        if model_last:
            keep_files.add(model_last)
        elif numbered_models:
            # If no model_last, keep the highest numbered model as "last"
            numbered_models.sort(key=lambda x: x[0], reverse=True)
            highest_model = numbered_models[0][1]
            keep_files.add(highest_model)
            # Remove it from the list of delete candidates (implicitly handled by exclusion)
            
        # Also keep model_final.pt if it exists (usually redundant if last exists, but safe to keep)
        for f in others:
            if f.name == "model_final.pt":
                keep_files.add(f)

        # Delete everything else
        for f in pt_files:
            full_path = root_path / f
            if full_path not in keep_files:
                size = full_path.stat().st_size
                try:
                    os.remove(full_path)
                    deleted_count += 1
                    reclaimed_space += size
                    print(f"Deleted: {full_path} ({size/1024/1024:.2f} MB)")
                except Exception as e:
                    print(f"Error deleting {full_path}: {e}")
            else:
                kept_count += 1
                
    print(f"\nCleanup Complete.")
    print(f"Deleted {deleted_count} intermediate files.")
    print(f"Kept {kept_count} best/last files.")
    print(f"Reclaimed {reclaimed_space/1024/1024/1024:.2f} GB.")

if __name__ == "__main__":
    main()
