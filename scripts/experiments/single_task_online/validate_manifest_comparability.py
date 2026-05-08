#!/usr/bin/env python3
"""Validate paired-method comparability constraints for a manifest CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-methods",
        required=True,
        help="Comma-separated method keys expected in each comparison group.",
    )
    parser.add_argument(
        "--group-by",
        default="stage,suite,task_key,seed,hparam_profile",
        help="Comma-separated columns defining one comparable group.",
    )
    parser.add_argument(
        "--const-cols",
        default=(
            "env,episode_length,max_epochs,num_envs,eval_runs,"
            "rollout_episodes,rollout_max_steps,wandb_project"
        ),
        help="Comma-separated columns that must be identical across methods in a group.",
    )
    args = parser.parse_args()

    rows = parse_csv(args.manifest)
    if not rows:
        raise SystemExit(f"Manifest is empty: {args.manifest}")

    group_cols = [c.strip() for c in args.group_by.split(",") if c.strip()]
    const_cols = [c.strip() for c in args.const_cols.split(",") if c.strip()]
    expected_methods = {m.strip() for m in args.expected_methods.split(",") if m.strip()}

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(col, "") for col in group_cols)
        groups[key].append(row)

    errors: list[str] = []
    ok_groups = 0
    for key, grp in groups.items():
        methods = {row["method_key"] for row in grp}
        if methods != expected_methods:
            errors.append(
                f"group={key} methods={sorted(methods)} expected={sorted(expected_methods)}"
            )
            continue

        for col in const_cols:
            vals = {row.get(col, "") for row in grp}
            if len(vals) != 1:
                errors.append(f"group={key} column={col} values={sorted(vals)}")
        ok_groups += 1

    print(f"manifest={args.manifest}")
    print(f"rows={len(rows)} groups={len(groups)} comparable_groups={ok_groups}")
    if errors:
        print("RESULT=FAIL")
        print("mismatches:")
        for msg in errors[:100]:
            print(f"  - {msg}")
        if len(errors) > 100:
            print(f"  ... and {len(errors)-100} more")
        raise SystemExit(1)

    print("RESULT=PASS")


if __name__ == "__main__":
    main()
