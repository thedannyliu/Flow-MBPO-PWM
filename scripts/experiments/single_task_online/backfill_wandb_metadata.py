#!/usr/bin/env python3
"""Backfill WandB tags/config metadata for single-task online experiments."""

from __future__ import annotations

import argparse
import re
from typing import Any, Dict, Iterable, Set, Tuple


VALIDATION_STAGES = {"smoke", "sanity", "validation", "verify", "verification"}


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _get_first_nonempty(config: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = config.get(key, "")
        if isinstance(value, (str, int, float)):
            text = _as_str(value)
            if text:
                return text
    return ""


def _extract_metadata(run) -> Dict[str, str]:
    cfg = dict(run.config or {})
    exp = cfg.get("experiment", {})
    if not isinstance(exp, dict):
        exp = {}

    meta: Dict[str, str] = {}
    for field in ("stage", "suite", "task", "method", "hparam_profile", "gpu_type", "run_key"):
        meta[field] = _get_first_nonempty(
            cfg,
            (
                field,
                f"experiment.{field}",
            ),
        ) or _as_str(exp.get(field, ""))

    if not meta["run_key"]:
        meta["run_key"] = _get_first_nonempty(cfg, ("run_key",))
    return meta


def _infer_job_type(run, tags: Set[str]) -> str:
    current = _as_str(getattr(run, "job_type", ""))
    if current:
        return current
    name = _as_str(getattr(run, "name", ""))
    if name.endswith("_eval") or "job_eval" in tags:
        return "eval"
    if "job_validation" in tags:
        return "validation"
    return "train"


def _infer_purpose(stage: str, job_type: str) -> str:
    stage_norm = stage.lower()
    job_type_norm = job_type.lower()
    if job_type_norm == "eval":
        return "eval"
    if stage_norm in VALIDATION_STAGES or job_type_norm == "validation":
        return "validation"
    return "train"


def _seed_tag(run_key: str, run_name: str) -> str:
    text = run_key or run_name
    m = re.search(r"_s(\d+)_", text)
    if m:
        return f"seed_{m.group(1)}"
    m = re.search(r"_s(\d+)$", text)
    if m:
        return f"seed_{m.group(1)}"
    return ""


def _desired_tags(meta: Dict[str, str], run, existing_tags: Set[str]) -> Set[str]:
    tags = set(existing_tags)
    job_type = _infer_job_type(run, tags)
    purpose = _infer_purpose(meta.get("stage", ""), job_type)

    for key in ("stage", "suite", "task", "method", "hparam_profile", "gpu_type"):
        value = _as_str(meta.get(key, ""))
        if value:
            tags.add(f"{key}_{value}")

    seed = _seed_tag(meta.get("run_key", ""), _as_str(getattr(run, "name", "")))
    if seed:
        tags.add(seed)

    tags.add(f"job_{job_type}")
    tags.add(f"purpose_{purpose}")
    tags.add("single_task_online")
    tags.add("online_rl")
    return tags


def _should_process_run(run, include_archived: bool) -> bool:
    _ = include_archived
    return True


def _update_run(run, tags: Set[str], job_type: str, purpose: str, dry_run: bool) -> Tuple[bool, str]:
    old_tags = set(run.tags or [])
    changed = old_tags != tags

    if dry_run:
        return changed, "dry-run"

    if changed:
        run.tags = sorted(tags)

    cfg_updates = {
        "tracking_job_type": job_type,
        "tracking_purpose": purpose,
    }
    try:
        run.config.update(cfg_updates, allow_val_change=True)
    except Exception:
        pass

    try:
        if _as_str(getattr(run, "job_type", "")) != job_type:
            run.job_type = job_type
            changed = True
    except Exception:
        # Some API versions may not allow mutating job_type; tags/config still work.
        pass

    if changed:
        run.update()
    return changed, "updated" if changed else "no-change"


def process_project(entity: str, project: str, max_runs: int, dry_run: bool, include_archived: bool) -> None:
    import wandb

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    total = 0
    changed = 0

    print(f"\n[project] {entity}/{project}")
    for run in runs:
        if max_runs > 0 and total >= max_runs:
            break
        if not _should_process_run(run, include_archived):
            continue

        total += 1
        meta = _extract_metadata(run)
        old_tags = set(run.tags or [])
        new_tags = _desired_tags(meta, run, old_tags)
        job_type = _infer_job_type(run, new_tags)
        purpose = _infer_purpose(meta.get("stage", ""), job_type)

        did_change, status = _update_run(run, new_tags, job_type, purpose, dry_run=dry_run)
        if did_change:
            changed += 1
        print(
            f"  - {run.id} name='{run.name}' state={run.state} "
            f"job_type={job_type} purpose={purpose} status={status}"
        )

    print(
        f"[summary] {entity}/{project}: scanned={total}, "
        f"changed={changed}, dry_run={dry_run}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill WandB tags/metadata for past runs.")
    parser.add_argument("--entity", required=True, help="WandB entity/user/org.")
    parser.add_argument(
        "--project",
        action="append",
        required=True,
        help="WandB project (repeat --project for multiple projects).",
    )
    parser.add_argument("--max-runs", type=int, default=0, help="Limit runs per project (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying.")
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Keep default behavior; reserved for compatibility.",
    )
    args = parser.parse_args()

    for project in args.project:
        process_project(
            entity=args.entity,
            project=project,
            max_runs=args.max_runs,
            dry_run=args.dry_run,
            include_archived=args.include_archived,
        )


if __name__ == "__main__":
    main()
