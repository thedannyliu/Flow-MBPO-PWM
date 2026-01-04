import os
import glob
import pandas as pd
import yaml
from pathlib import Path

def parse_overrides(overrides_path):
    config = {}
    with open(overrides_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Hydra overrides in yaml typically start with "- key=value"
            if not line.startswith("-"):
                continue
            
            # Remove leading "- "
            content = line[2:].strip()
            
            if "=" in content:
                key, value = content.split("=", 1)
                config[key] = value
    return config

def main():
    base_dirs = [Path("outputs/2026-01-03"), Path("outputs/mt30")]
    results = []
    
    print(f"Searching for results in {base_dirs}...")
    
    # 1. Find all *_results.csv files
    csv_files = []
    for base_dir in base_dirs:
        if base_dir.exists():
            csv_files.extend(list(base_dir.rglob("*_results.csv")))
            
    print(f"Found {len(csv_files)} CSV files.")

    for csv_path in csv_files:
        run_dir = csv_path.parent.parent
        overrides_path = run_dir / ".hydra" / "overrides.yaml"
        
        try:
            # Parse Config
            if overrides_path.exists():
                config = parse_overrides(overrides_path)
                alg = config.get("alg", "unknown")
                task = config.get("task", "unknown")
                seed = config.get("general.seed", "unknown")
            else:
                # Infer from path or log file content if possible
                # e.g. outputs/mt30/4011713/8_s456/logs
                path_parts = csv_path.parts
                alg = "unknown"
                task = csv_path.name.replace("_results.csv", "")
                seed = "unknown"
                
                # Heuristic: Check path for algorithm clues
                if "baseline" in str(csv_path).lower():
                    alg = "pwm_48M_mt_baseline"
                elif "flowpolicy" in str(csv_path).lower():
                    alg = "pwm_48M_mt_flowpolicy"
                
                # Try to extract seed from path sXXX
                import re
                match = re.search(r"_s(\d+)", str(run_dir))
                if match:
                    seed = match.group(1)

            # Check CSV content
            try:
                df = pd.read_csv(csv_path)
            except:
                print(f"Skipping empty/corrupt CSV: {csv_path}")
                continue

            if not df.empty:
                last_row = df.iloc[-1]
                iteration = last_row.get("iteration", 0)
                status = "COMPLETED" if iteration >= 9900 else "IN_PROGRESS/FAILED"
                
                results.append({
                    "Algorithm": alg,
                    "Task": task,
                    "Seed": seed,
                    "Status": status,
                    "Iteration": iteration,
                    "Episode Reward": last_row.get("episode_reward", float("nan")),
                    "Planning Reward": last_row.get("episode_reward_planning", float("nan")),
                    "Success": last_row.get("episode_success", float("nan")),
                    "Planning Success": last_row.get("episode_success_planning", float("nan")),
                    "Path": str(csv_path),
                    "RunDir": str(run_dir)
                })
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

    if not results:
        print("No results found.")
        return

    # Create DataFrame
    df_raw = pd.DataFrame(results)
    
    # Save Raw for Cleanup
    df_raw.to_csv("mt30_all_runs.csv", index=False)
    print(f"Saved raw list of {len(df_raw)} runs to mt30_all_runs.csv")

    # Deduplicate for Summary
    # Prioritize: Status=COMPLETED, then Latest Run (heuristic: mt30 path > date path), then Iteration
    def rank_run(row):
        score = 0
        if row["Status"] == "COMPLETED": score += 1000
        if "outputs/mt30" in row["Path"]: score += 100
        score += row["Iteration"]
        return score

    df_raw["Score"] = df_raw.apply(rank_run, axis=1)
    df_raw = df_raw.sort_values(by=["Score"], ascending=False)
    
    # Drop duplicates, keeping best score
    df_clean = df_raw.drop_duplicates(subset=["Algorithm", "Task", "Seed"], keep="first")
    
    # Sort for display
    df_clean = df_clean.sort_values(by=["Algorithm", "Task", "Seed"])
    
    # Save Clean Summary
    output_file = "mt30_results_summary.csv"
    columns = ["Algorithm", "Task", "Seed", "Status", "Iteration", "Episode Reward", "Planning Reward", "Success", "Planning Success", "Path", "RunDir"]
    df_clean[columns].to_csv(output_file, index=False)
    
    print(f"\nSaved consolidated summary to {output_file} ({len(df_clean)} rows)")
    print("\nSummary:")
    print(df_clean[columns].to_string(index=False))

    # Calculate mean rewards per task/alg
    print("\nMean Rewards:")
    mean_df = df_clean[df_clean["Status"] == "COMPLETED"].groupby(["Algorithm", "Task"])[["Episode Reward", "Planning Reward"]].mean()
    print(mean_df)

if __name__ == "__main__":
    main()
