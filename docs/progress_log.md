# Progress Log

> Newest entries at top.

---

## 2026-01-06 04:20 – Jobs Running Successfully

### Fixed Conda Activation Issue
- Previous failures: Python codec error from incorrect conda activation
- Solution: Use same pattern as working pretrain script:
```bash
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flow-mbpo
export PATH=$CONDA_PREFIX/bin:$PATH
```

### Jobs Now Running
| Job ID | Type | Runs | Hardware | Status |
|--------|------|------|----------|--------|
| `4015342` | Flow 50k | 9 | H100 | 🟢 ALL RUNNING |
| `4015343` | Flow 100k | 9 | H200 | 🟢 ALL RUNNING |
| `4015344` | 150k Sweep | 18 | H200 | 2 RUNNING, 16 QUEUED |
| `4013702` | Flow WM Pretrain | 1 | H100 | 🟢 9h12m |

---

## 2026-01-06 04:00 – First Submission Attempt (FAILED)
- Jobs 4015240/50/51 failed immediately
- Cause: `source ~/.bashrc` didn't work, wrong Python version used
