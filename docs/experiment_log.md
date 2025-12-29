# Experiment Log

Purpose: Registry for all training jobs with Slurm IDs, configs, seeds, wandb links, and status.

---

## Status Summary (Dec 29, 4:22 PM)

### ANYMAL (25 COMPLETED, 1 RUNNING)
All Anymal experiments nearly complete! Only 1 job still running.

### ANT (13 COMPLETED, 8 RUNNING)
Most Ant jobs completed or running. Resubmitted 2 failed Full Flow jobs.

---

# ANT EXPERIMENTS

## WandB: flow-mbpo-single-task-ant

### Smoke Tests ✅
| Job ID | Variant | Status |
|--------|---------|--------|
| 3093139 | Baseline | ✅ COMPLETED |
| 3093140 | Flow WM | ✅ COMPLETED |
| 3093141 | Flow Policy | ✅ COMPLETED |
| 3093142 | Full Flow | ✅ COMPLETED |

### Training Results

#### Baseline ✅ COMPLETED
| Job ID | Seed | Runtime |
|--------|------|---------|
| 3093165 | 42 | 2:06:32 |
| 3093166 | 123 | 1:58:33 |
| 3093167 | 456 | 2:06:23 |

#### Flow WM K=4 Heun ✅ COMPLETED
| Job ID | Seed | Runtime |
|--------|------|---------|
| 3093168 | 42 | 4:37:40 |
| 3093169 | 123 | 4:39:13 |
| 3093170 | 456 | 4:40:16 |

#### Flow WM K=2 Heun ✅ COMPLETED
| Job ID | Seed | Runtime |
|--------|------|---------|
| 3093171 | 42 | 3:25:42 |
| 3093172 | 123 | 3:20:58 |
| 3093173 | 456 | 3:25:20 |

#### Flow WM K=8 Euler 🔄 (1 COMPLETED, 2 RUNNING)
| Job ID | Seed | Status |
|--------|------|--------|
| 3093174 | 42 | ✅ COMPLETED (4:40:09) |
| 3093175 | 123 | 🔄 RUNNING |
| 3093176 | 456 | 🔄 RUNNING |

#### Flow Policy 🔄 (3 RUNNING)
| Job ID | Seed | Status |
|--------|------|--------|
| 3093177 | 42 | 🔄 RUNNING |
| 3093178 | 123 | 🔄 RUNNING |
| 3093179 | 456 | 🔄 RUNNING |

#### Full Flow 🔄 (1 RUNNING, 2 RESUBMITTED)
| Job ID | Seed | Status |
|--------|------|--------|
| 3093180 | 42 | 🔄 RUNNING |
| 3099532 | 123 | 🔄 RESUBMITTED |
| 3099534 | 456 | 🔄 RESUBMITTED |

---

# ANYMAL EXPERIMENTS

## WandB: flow-mbpo-single

### All Completed ✅ (25 jobs)

| Variant | Job IDs | Status |
|---------|---------|--------|
| Baseline | 3080227-3080229 | ✅ |
| Flow WM K=4 | 3082681-3082683 | ✅ |
| Flow WM K=2 | 3082684-3082686 | ✅ |
| Flow WM K=8 | 3084837-3084839 | ✅ |
| Flow Policy | 3084840-3084842 | ✅ |
| Full Flow | 3084843, 3087706-3087707 | ✅ |
| Flow WM H=8 | 3087697-3087699 | ✅ |
| Flow Policy H=8 | 3087703-3087705 | ✅ |
| Flow WM LowLR | 3087700-3087702 | ✅ |
| Full Flow H=8 | 3091993, 3091995-3091996 | ✅ |
| Flow WM HighLR | 3091997-3091998 | ✅ |
| Flow WM StrongReg | 3093114-3093116 | ✅ |

### Still Running (1 job)
| Job ID | Variant | Status |
|--------|---------|--------|
| 3091999 | Flow WM HighLR s456 | 🔄 RUNNING (~11h) |

---

## Resource Allocation
- **Memory**: 400GB
- **Time**: 40 hours
- **Partition**: gpu-l40s
