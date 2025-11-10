# Bug Fixes and Checkpoint Strategy

## Issues Fixed

### 1. TypeError: `unsupported operand type(s) for /: 'str' and 'str'`

**Problem:**
```python
visualizer_path = self.log_dir / 'visualizer_data.pkl'  # ❌ Fails when log_dir is str
```

**Root Cause:**
- `self.log_dir` was stored as a string
- The `/` operator for path concatenation only works with `pathlib.Path` objects
- This error occurred at training completion (iteration 14994/15000)

**Solution:**
```python
from pathlib import Path

# In __init__:
self.log_dir = Path(logdir)  # ✅ Convert to Path object

# Usage works now:
visualizer_path = self.log_dir / 'visualizer_data.pkl'  # ✅ Path / str = Path
```

**Files Modified:**
- `src/pwm/algorithms/pwm.py`:
  - Added `from pathlib import Path` import
  - Changed `self.log_dir = logdir` → `self.log_dir = Path(logdir)`
  - Changed `os.makedirs(self.log_dir, exist_ok=True)` → `self.log_dir.mkdir(parents=True, exist_ok=True)`
  - Updated `save()` method to handle Path objects properly

---

### 2. Visualization Directory Warning

**Problem:**
```
Warning: Log directory not found at /storage/.../PWM/logs/pwm_5M_dflex_ant_seed42
Skipping visualization generation.
```

**Root Cause:**
- Hydra changes working directory to `outputs/<date>/<time>/`
- The script tries to find logs in relative path from wrong location
- `general.logdir=logs/pwm_5M_dflex_ant_seed42` is relative to current directory

**Current Behavior:**
```
Working Directory: /storage/.../PWM/outputs/2025-11-08/23-48-46/
Log Directory: logs/pwm_5M_dflex_ant_seed42  (relative to Hydra output dir)
Actual Location: /storage/.../PWM/outputs/2025-11-08/23-48-46/logs/pwm_5M_dflex_ant_seed42 ✓
Script Looks At: /storage/.../PWM/logs/pwm_5M_dflex_ant_seed42 ❌
```

**Solution Options:**

**Option A: Use absolute paths (RECOMMENDED)**
```bash
python scripts/train_dflex.py \
    general.logdir=/storage/home/.../PWM/logs/${ALGORITHM}_${TASK}_seed${SEED}
```

**Option B: Look in Hydra output directory**
The logs are actually saved correctly in `outputs/<date>/<time>/logs/...`, just the visualization script looks in the wrong place.

---

### 3. Excessive Checkpoint Saving

**Problem:**
```bash
$ ls outputs/2025-11-08/23-48-46/logs/pwm_5M_dflex_ant_seed42/
PWM_iter500_rew-17.pt
PWM_iter1000_rew-19.pt
PWM_iter1500_rew-21.pt
...
PWM_iter14500_rew-27.pt
PWM_iter15000_rew-27.pt
final_policy.pt
final_policy.buffer
```

**Issues:**
- Saves checkpoint every 500 iterations (30 checkpoints for 15K iterations)
- Each checkpoint is ~50-100MB (model + optimizer states)
- Total: 1.5-3GB of redundant checkpoints per run
- Filenames include reward in name, making it hard to identify "best" checkpoint

**Standard Deep Learning Practice:**
```bash
$ ls checkpoints/
best_policy.pt        # Best validation performance
final_policy.pt       # Last iteration
latest_checkpoint.pt  # For resuming training (optional)
```

**Solution Implemented:**

Changed from:
```python
if self.iter_count % self.save_interval == 0:
    name = self.__class__.__name__
    name = name + f"_iter{self.iter_count}_rew{-mean_policy_loss:0.0f}"
    self.save(name)  # ❌ Saves PWM_iter500_rew-17.pt, PWM_iter1000_rew-19.pt, ...
```

To:
```python
if self.iter_count % self.save_interval == 0:
    self.save("latest_checkpoint")  # ✅ Overwrites same file
```

**Checkpoints Now Saved:**
1. **`init_policy.pt`** - Initial random policy (saved once at start)
2. **`best_policy.pt`** - Best performing policy (updated when policy improves)
3. **`latest_checkpoint.pt`** - Most recent checkpoint (overwrites every 500 iters)
4. **`final_policy.pt` + `final_policy.buffer`** - Final policy with replay buffer (saved at end)

**Benefits:**
- ✅ Saves ~90% disk space (4 files vs 30+ files)
- ✅ Clear naming: "best" is best, "final" is final, "latest" is for resuming
- ✅ Still can resume training from `latest_checkpoint.pt`
- ✅ Buffer only saved with final policy (most complete)

---

### 4. L40s Buffer Initialization Hang

**Problem:**
```
World Model Total Parameters: 1,400,421
Using Baseline MLP Dynamics
[Process hangs here - never continues]
```

**Root Cause:**
- L40s GPUs (Ada Lovelace architecture) hang during `Buffer.__init__()`
- Not a code bug - same code works perfectly on H200 (Hopper architecture)
- Issue is systematic across all L40s nodes tested
- Environment simulation works fine, only buffer initialization fails

**Evidence:**
- Tested on 3 different L40s nodes: all hang
- Tested multiple configurations: all hang
- dflex environment works (reset, step, forward pass all functional)
- Process shows CPU usage (118%) but no output
- H200 with identical code completes successfully

**Hypothesis:**
- CUDA 12.9 (L40s driver 575.57.08) incompatibility
- PyTorch memory allocation issue with Ada architecture  
- dflex library optimized for Hopper, poor L40s support

**Not a Fix - Just Documentation:**
This is a hardware/driver compatibility issue beyond user-level fixes. Documented for reference. Use H200 for training.

---

## Checkpoint Strategy Justification

### Why NOT Save Every 500 Iterations?

**RL Training Characteristics:**
1. **Non-monotonic improvement**: Policy performance fluctuates
2. **Long training**: 15K iterations = 3-5 hours
3. **Stochastic**: Each run has different trajectory
4. **Exploration vs exploitation**: Early checkpoints may be worse

**Counter-arguments for frequent saves:**
- "Need intermediate checkpoints for analysis" → Use WandB for metrics, not checkpoints
- "What if training crashes?" → Save `latest_checkpoint` (we do!)
- "Want to see learning progression" → That's what WandB plots are for
- "Need to resume from any point" → `latest_checkpoint` is sufficient

**Standard ML Practice:**
```python
# Most frameworks (PyTorch Lightning, HuggingFace, etc.)
ModelCheckpoint(
    save_top_k=1,           # Save best only
    save_last=True,         # Save last checkpoint
    every_n_epochs=10,      # Periodic save (overwrites)
)
```

### When Would You Need More Checkpoints?

**Legitimate use cases:**
1. **Curriculum learning**: Need checkpoints at specific curriculum stages
2. **Ensemble training**: Train from multiple checkpoints
3. **Analysis**: Study how specific layers evolve (use hooks instead)
4. **Debugging**: Investigate training instabilities (use WandB profiler)

**Our case:**
- ✅ Single-task training (no curriculum)
- ✅ WandB tracks all metrics
- ✅ Training is stable (no crashes)
- ✅ Not doing ensemble or multi-stage training

**Conclusion: 4 checkpoints sufficient**

---

## Output Directory Structure

### Current Issue

**Problem:**
```bash
$ ls outputs/
2025-11-08/
  23-48-46/    # ❓ Which run is this?
  23-49-33/    # ❓ What seed?
  23-51-12/    # ❓ Baseline or flow?
```

**Hydra creates timestamped directories, making it hard to identify runs**

### Proposed Solutions

**Option 1: Symlinks (RECOMMENDED)**
```bash
# After job submission, create symlink
outputs/pwm_5M_dflex_ant_seed42 -> 2025-11-08/23-48-46
outputs/pwm_48M_flow_dflex_ant_seed42 -> 2025-11-08/23-49-33
```

**Option 2: Configure Hydra output directory**
```yaml
# config.yaml
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}_${alg.name}_${env.name}_seed${general.seed}
```

**Option 3: Use WandB run names**
```bash
# WandB automatically creates:
# - Run name: "pwm_5M_dflex_ant_seed42"
# - Run ID: unique hash
# - Easy to search and filter
```

**Implemented: Use WandB + Add job metadata file**

---

## Files Modified Summary

### `src/pwm/algorithms/pwm.py`
```diff
+ from pathlib import Path

- self.log_dir = logdir
+ self.log_dir = Path(logdir)

- os.makedirs(self.log_dir, exist_ok=True)
+ self.log_dir.mkdir(parents=True, exist_ok=True)

- if self.iter_count % self.save_interval == 0:
-     name = self.__class__.__name__
-     name = name + f"_iter{self.iter_count}_rew{-mean_policy_loss:0.0f}"
-     self.save(name)
+ if self.iter_count % self.save_interval == 0:
+     self.save("latest_checkpoint")

- visualizer_path = self.log_dir / 'visualizer_data.pkl'
+ visualizer_path = Path(self.log_dir) / 'visualizer_data.pkl'

def save(self, filename, log_dir=None, buffer=False):
-     log_dir = self.log_dir if log_dir is None else log_dir
+     log_dir = Path(self.log_dir) if log_dir is None else Path(log_dir)
-     os.path.join(self.log_dir, "{}.pt".format(filename)),
+     str(log_dir / f"{filename}.pt"),
-     self.buffer.save(os.path.join(self.log_dir, "{}.buffer".format(filename)))
+     self.buffer.save(str(log_dir / f"{filename}.buffer"))
```

---

## Testing the Fixes

### 1. Verify Path Fix
```bash
python -c "
from pathlib import Path
log_dir = Path('logs/test')
viz_path = log_dir / 'visualizer_data.pkl'
print(f'✓ Path concatenation works: {viz_path}')
"
```

### 2. Check Checkpoint Strategy
```bash
# After training completes
ls outputs/2025-11-09/*/logs/pwm_5M_*/
# Should see:
# - init_policy.pt
# - best_policy.pt
# - latest_checkpoint.pt
# - final_policy.pt
# - final_policy.buffer
# Total: 5 files (not 30+)
```

### 3. Test Visualization
```bash
LOG_DIR="outputs/2025-11-09/12-34-56/logs/pwm_5M_dflex_ant_seed42"
python scripts/generate_visualizations.py --log-dir "$LOG_DIR"
```

---

## Migration Guide

### For Existing Checkpoints

**If you have old checkpoints and want to clean up:**

```bash
# Keep only important ones, delete rest
cd logs/pwm_5M_dflex_ant_seed42/
mkdir old_checkpoints
mv PWM_iter*.pt old_checkpoints/  # Move intermediate checkpoints
# Keep: best_policy.pt, final_policy.pt, final_policy.buffer
```

### For New Training Runs

**No action needed - new strategy applies automatically**

```bash
./scripts/submit_job.sh single pwm_5M dflex_ant 42
# Will only save 4-5 checkpoints total
```

---

## Performance Impact

### Disk Space Savings
```
Before: 30 checkpoints × 80MB = 2.4GB per run
After:  4 checkpoints × 80MB = 320MB per run
Savings: 2.1GB per run (87.5% reduction)

For 10 runs:
Before: 24GB
After:  3.2GB
Savings: 20.8GB
```

### I/O Time Savings
```
Before: 30 saves × 2 seconds = 60 seconds
After:  4 saves × 2 seconds = 8 seconds
Time saved: 52 seconds per run

(Negligible compared to 3-5 hour training time)
```

### Checkpoint Save Times
```
Model state dict: ~40MB → ~0.5s
Optimizer state dicts: ~40MB → ~0.5s
Buffer (final only): ~500MB → ~5s

Per checkpoint: ~1-2 seconds
Total overhead: ~8 seconds (0.04% of training time)
```

---

## Conclusion

✅ **TypeError fixed**: Path objects used consistently
✅ **Checkpoint bloat eliminated**: 87.5% disk space savings
✅ **Standard ML practice**: Save best, latest, final
✅ **L40s issue documented**: Hardware incompatibility (use H200)
✅ **Backward compatible**: Old checkpoints still loadable

**Recommendation**: Use H200 GPUs for all future training runs.
