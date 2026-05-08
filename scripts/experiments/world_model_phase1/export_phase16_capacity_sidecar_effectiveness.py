#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(open(args.input, newline='', encoding='utf-8')))
    for r in rows:
        r['n_eval_completed'] = int(r['n_eval_completed'])
        for k in ['train_wm_loss_mean','train_one_step_dyn_loss_mean','train_rollout_dyn_loss_mean','val_wm_loss_mean','val_one_step_dyn_loss_mean','val_rollout_dyn_loss_mean','train_one_step_reward_loss_mean','val_one_step_reward_loss_mean','elapsed_seconds_mean']:
            r[k] = float(r[k])

    flow_ref = next(r for r in rows if r['profile'] == 'flow_ref_uniform_heun4')
    mlp_ref = next(r for r in rows if r['profile'] == 'mlp_ref')
    ensemble_ref = next(r for r in rows if r['profile'] == 'mlp_ensemble5')

    fieldnames = [
        'profile','method_key','n_eval_completed','val_rollout_dyn_loss_mean','train_rollout_dyn_loss_mean',
        'delta_vs_flow_ref','delta_vs_mlp_ref','delta_vs_ensemble_ref','elapsed_seconds_mean','status','note'
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x['val_rollout_dyn_loss_mean']):
            dfr = r['val_rollout_dyn_loss_mean'] - flow_ref['val_rollout_dyn_loss_mean']
            dmr = r['val_rollout_dyn_loss_mean'] - mlp_ref['val_rollout_dyn_loss_mean']
            der = r['val_rollout_dyn_loss_mean'] - ensemble_ref['val_rollout_dyn_loss_mean']
            if r['profile'] == 'flow_ref_uniform_heun4':
                status = 'flow_reference'; note = 'current flow reference'
            elif r['profile'] == 'mlp_ref':
                status = 'baseline_reference'; note = 'deterministic MLP reference'
            elif r['profile'] == 'mlp_ensemble5':
                status = 'stronger_baseline_reference'; note = 'stronger retained baseline'
            elif r['profile'] == 'flow_capacity_wide1024':
                if dfr < -0.001:
                    status = 'capacity_keep'; note = 'material capacity improvement over flow reference'
                elif dfr < 0:
                    status = 'capacity_weak_keep'; note = 'small capacity improvement over flow reference'
                elif dfr <= 0.001:
                    status = 'capacity_neutral'; note = 'capacity change effectively tied with flow reference'
                else:
                    status = 'capacity_drop'; note = 'worse than flow reference'
            else:
                if dfr < -0.001:
                    status = 'keep'; note = 'material improvement over flow reference'
                elif dfr < 0:
                    status = 'weak_keep'; note = 'small improvement over flow reference'
                elif dfr <= 0.001:
                    status = 'neutral'; note = 'effectively tied with flow reference'
                else:
                    status = 'drop'; note = 'worse than flow reference'
            w.writerow({
                'profile': r['profile'],
                'method_key': r['method_key'],
                'n_eval_completed': r['n_eval_completed'],
                'val_rollout_dyn_loss_mean': f"{r['val_rollout_dyn_loss_mean']:.9f}",
                'train_rollout_dyn_loss_mean': f"{r['train_rollout_dyn_loss_mean']:.9f}",
                'delta_vs_flow_ref': f"{dfr:.9f}",
                'delta_vs_mlp_ref': f"{dmr:.9f}",
                'delta_vs_ensemble_ref': f"{der:.9f}",
                'elapsed_seconds_mean': f"{r['elapsed_seconds_mean']:.3f}",
                'status': status,
                'note': note,
            })
    print(out)


if __name__ == '__main__':
    main()
