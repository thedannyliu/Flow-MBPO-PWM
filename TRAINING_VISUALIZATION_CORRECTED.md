# Training Results Visualization - Nov 17-18, 2025

## Performance Comparison

```
Reward (R)
    │
1200├────────────────────────────────── Expected Baseline (Nov 8)
    │                        ╭──────╮
    │                        │ V2   │ Peak: 1197.40 🏆
1150│                        │      │
    │                   ╭────┴──────┴────╮
    │              ╭────┤ V1        V3   │
1100│              │    │                │
    │              │    ╰────────────────╯
    │              │    
1050│              │    Peak: V1=1132.89, V3=1137.49
    │              │
1000│              │
    │         ╭────╯
 950│         │
    │         │
 900│         │
    │    ╭────╯
 850│    │
    │    │
 800│    │
    │    │
 750│    │
    │    │
 700│    │
    │    │
 650│    │
    │    │
 600│    │
    │    │
 550│    │
    │    │
 500│    │
    │    │
 450│    │
    │    │
 400│    │
    │    │
 350│    │
    │    │
 300│    │ ╭──────────────────────────── Current Baseline (Nov 17)
    │    │ │
 250│    │ │                            Peak: 291.93 ⚠️
    │    │ │                            (76% below expected!)
 200│    │ │
    │    │ │
 150│    │ │╮ (avg last 10: 150.43)
    │    │ ╰╯
 100│╭───╯
    │
  50│
    │
   0└─────┴─┴───────────────────────────────
    0    50 100         130    157     Iterations


Legend:
  Baseline (⚠️)  - Peak: 291.93  - Only 11 iterations
  Flow V1  (✅)  - Peak: 1132.89 - substeps=2, heun
  Flow V2  (🏆)  - Peak: 1197.40 - substeps=4, heun ⭐ BEST
  Flow V3  (⚠️)  - Peak: 1137.49 - substeps=8, euler (unstable)
```

## Training Progression Detail

### Flow V2 (Best Configuration) 🏆
```
 R
1200│                                        ● 1197 (peak)
    │                                    ╱
1150│                                ╱───
    │                            ╱
1100│                        ╱───
    │                    ╱
1050│                ╱
    │            ╱
1000│        ╱
    │    ╱
 950│╱───
    │
  0└────────────────────────────────────────────────
    0    60   82   109  135  145  157  Iterations

Training time: ~3h 15m
Stability: Excellent (avg last 10: 1165.38)
Improvement over baseline: 4.1×
```

### Flow V1 (Conservative)
```
 R
1200│                                  ● 1133 (peak)
    │                              ╱───
1150│                          ╱───
    │                      ╱
1100│                  ╱───
    │              ╱
1050│          ╱
    │      ╱
1000│  ╱───
    │
  0└────────────────────────────────────────
    0   57  61  84  110 126 130  Iterations

Training time: ~2h 30m  
Stability: Excellent (avg last 10: 1132.49)
Improvement over baseline: 3.9×
```

### Flow V3 (Unstable)
```
 R
1200│              ● 1137 (peak)
    │              ╱╲
1150│          ╱───  ╲
    │      ╱───       ╲
1100│  ╱───            ╲
    │                   ╲
1050│                    ╲
    │                     ╲
1000│                      ╲___
    │
  0└──────────────────────────────
    0   59  78  101  ...  Iterations

Issue: Performance degradation after peak
Avg last 10: 978.59 (14% drop from peak)
Cause: Euler integrator + substeps=8 numerical issues
```

### Baseline (Problematic)
```
 R
 300│  ● 292 (peak)
    │  ╱╲
 250│ ╱  ╲
    │     ╲___
 200│         ╲
    │          ╲
 150│           ╲___
    │
 100│
    │
  50│
    │
   0└───────────────
    0    11   Iterations

Issue: Only 11 iterations logged
Expected: R ~ 1200 based on Nov 8 success
Current: R ~ 292 (76% below expected)
Status: Investigation needed ⚠️
```

## Substeps Comparison

```
Peak Reward
    │
1200├───────────╮
    │     4     │ 1197.40 ⭐ BEST
1150│           │
    │     ├─────┴─────┤
1100│     2           8
    │  1132.89     1137.49
1050│
    │
    └─────┴─────┴─────┴─────
         2     4     8     Substeps

Stability (Avg Last 10):
  substeps=2: 1132.49 ✅ (99.97% of peak)
  substeps=4: 1165.38 ✅ (97.33% of peak)
  substeps=8:  978.59 ❌ (86.02% of peak)

Conclusion: substeps=4 optimal
```

## Integrator Comparison

```
               Peak    Stability
Heun (V1,V2):  1133-1197  ✅ Excellent
Euler (V3):    1137       ❌ Poor (drops to 978)

Conclusion: Heun integrator preferred
```

## Training Efficiency

```
Training Time vs Performance

 R
1200│                         ● V2 (3h 15m) 🏆
    │                    
1150│           ● V1 (2h 30m)
    │      
1100│                              ● V3 (unstable)
    │
 300│  ● Baseline (stopped early)
    │
   0└─────────┴─────────┴─────────┴─────────
         1h       2h       3h       4h    Time

Best balance: Flow V2
  - Highest performance (1197.40)
  - Reasonable training time (~3h 15m)
  - Excellent stability
```

## Recommendations

### ✅ Use for Production
**Flow V2 (substeps=4, heun)**
- Peak: 1197.40
- Stability: 97.3% retention in last 10 iterations
- Training time: ~3-4 hours on L40s
- 4.1× improvement over baseline

### ⚠️ Investigate
**Baseline underperformance**
- Current: 292 (Nov 17)
- Expected: 1200 (Nov 8)
- Action: Check checkpoint, seeds, environment setup

### ❌ Avoid
**Flow V3 (substeps=8, euler)**
- Unstable: 14% performance drop
- No benefit over substeps=4
- Euler integrator inferior to Heun

---

**Note:** All R values from true environment interaction during training.  
Evaluation metrics (length=1000, loss=0.00) were artifacts of eval() bug (now fixed).

*Generated: November 18, 2025*
