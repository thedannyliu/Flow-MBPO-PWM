#!/usr/bin/env python3
"""
Checkpoint management utility for PWM training.

Features:
- List all checkpoints with metadata
- Find latest checkpoint for a specific run
- Compare checkpoint sizes and training progress
- Clean up old checkpoints (keep best and latest)
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import torch
from typing import List, Dict, Optional


def format_size(bytes_size: int) -> str:
    """Format bytes into human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict:
    """Load minimal metadata from checkpoint without loading full model."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        metadata = {
            'iter_count': checkpoint.get('iter_count', 'N/A'),
            'step_count': checkpoint.get('step_count', 'N/A'),
            'best_policy_loss': checkpoint.get('best_policy_loss', 'N/A'),
            'mean_horizon': checkpoint.get('mean_horizon', 'N/A'),
        }
        return metadata
    except Exception as e:
        return {'error': str(e)}


def list_checkpoints(root_dir: str = "outputs", pattern: str = "*.pt") -> List[Dict]:
    """List all checkpoints in the outputs directory."""
    root = Path(root_dir)
    if not root.exists():
        print(f"Error: Directory not found: {root_dir}")
        return []
    
    checkpoints = []
    for ckpt_path in root.rglob(pattern):
        if ckpt_path.name.endswith('.pt'):
            stat = ckpt_path.stat()
            
            # Extract run info from path
            parts = ckpt_path.parts
            try:
                log_idx = parts.index('logs')
                run_dir = parts[log_idx + 1] if log_idx + 1 < len(parts) else 'unknown'
            except ValueError:
                run_dir = 'unknown'
            
            info = {
                'path': str(ckpt_path),
                'name': ckpt_path.name,
                'run_dir': run_dir,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'metadata': load_checkpoint_metadata(ckpt_path)
            }
            checkpoints.append(info)
    
    return sorted(checkpoints, key=lambda x: x['modified'], reverse=True)


def find_latest_checkpoint(run_pattern: str, root_dir: str = "outputs") -> Optional[Path]:
    """Find the latest checkpoint for a specific run."""
    root = Path(root_dir)
    matching_checkpoints = []
    
    for ckpt_path in root.rglob("*.pt"):
        if run_pattern in str(ckpt_path):
            stat = ckpt_path.stat()
            matching_checkpoints.append((ckpt_path, stat.st_mtime))
    
    if not matching_checkpoints:
        return None
    
    # Sort by modification time and return latest
    matching_checkpoints.sort(key=lambda x: x[1], reverse=True)
    return matching_checkpoints[0][0]


def print_checkpoint_info(checkpoints: List[Dict], verbose: bool = False):
    """Pretty print checkpoint information."""
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    print(f"\n{'='*120}")
    print(f"Found {len(checkpoints)} checkpoint(s)")
    print(f"{'='*120}\n")
    
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"[{i}] {ckpt['name']}")
        print(f"    Run: {ckpt['run_dir']}")
        print(f"    Path: {ckpt['path']}")
        print(f"    Size: {format_size(ckpt['size'])}")
        print(f"    Modified: {ckpt['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        metadata = ckpt['metadata']
        if 'error' in metadata:
            print(f"    Status: Error loading - {metadata['error']}")
        else:
            print(f"    Progress: Epoch {metadata['iter_count']}, Step {metadata['step_count']}")
            print(f"    Best Loss: {metadata['best_policy_loss']}")
            print(f"    Mean Horizon: {metadata['mean_horizon']}")
        
        if verbose:
            # Check for buffer file
            buffer_path = Path(ckpt['path'].replace('.pt', '.buffer'))
            if buffer_path.exists():
                buffer_size = buffer_path.stat().st_size
                print(f"    Buffer: {format_size(buffer_size)}")
            else:
                print(f"    Buffer: Not found")
        
        print()


def clean_old_checkpoints(run_dir: str, keep_latest: int = 3, keep_best: bool = True, dry_run: bool = True):
    """
    Clean up old checkpoints, keeping only latest N and optionally the best one.
    
    Args:
        run_dir: Directory containing checkpoints
        keep_latest: Number of latest checkpoints to keep
        keep_best: Whether to keep the best_policy checkpoint
        dry_run: If True, only print what would be deleted
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Error: Directory not found: {run_dir}")
        return
    
    # Find all checkpoint files
    checkpoints = []
    for ckpt_path in run_path.glob("checkpoint_*.pt"):
        metadata = load_checkpoint_metadata(ckpt_path)
        checkpoints.append((ckpt_path, ckpt_path.stat().st_mtime, metadata))
    
    if not checkpoints:
        print(f"No checkpoints found in {run_dir}")
        return
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    
    # Determine which to keep
    to_keep = set()
    to_delete = []
    
    # Keep latest N
    for ckpt, _, _ in checkpoints[:keep_latest]:
        to_keep.add(ckpt)
    
    # Keep best policy
    best_policy_path = run_path / "best_policy.pt"
    if keep_best and best_policy_path.exists():
        to_keep.add(best_policy_path)
    
    # Mark rest for deletion
    for ckpt, _, _ in checkpoints[keep_latest:]:
        if ckpt not in to_keep:
            to_delete.append(ckpt)
    
    # Print summary
    print(f"\nCheckpoint cleanup for: {run_dir}")
    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Keeping: {len(to_keep)}")
    print(f"Deleting: {len(to_delete)}")
    
    if to_delete:
        print("\nFiles to delete:")
        total_size = 0
        for ckpt_path in to_delete:
            size = ckpt_path.stat().st_size
            total_size += size
            print(f"  - {ckpt_path.name} ({format_size(size)})")
            
            # Also count buffer if exists
            buffer_path = Path(str(ckpt_path).replace('.pt', '.buffer'))
            if buffer_path.exists():
                buffer_size = buffer_path.stat().st_size
                total_size += buffer_size
                print(f"    + {buffer_path.name} ({format_size(buffer_size)})")
        
        print(f"\nTotal space to free: {format_size(total_size)}")
        
        if not dry_run:
            confirm = input("\nProceed with deletion? (yes/no): ")
            if confirm.lower() == 'yes':
                for ckpt_path in to_delete:
                    ckpt_path.unlink()
                    print(f"Deleted: {ckpt_path}")
                    
                    buffer_path = Path(str(ckpt_path).replace('.pt', '.buffer'))
                    if buffer_path.exists():
                        buffer_path.unlink()
                        print(f"Deleted: {buffer_path}")
                print("\nCleanup complete!")
            else:
                print("Cancelled.")
        else:
            print("\n(Dry run - no files deleted. Use --no-dry-run to actually delete)")
    else:
        print("\nNo checkpoints to delete.")


def main():
    parser = argparse.ArgumentParser(description="PWM Checkpoint Management Tool")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all checkpoints')
    list_parser.add_argument('--root', default='outputs', help='Root directory to search')
    list_parser.add_argument('--pattern', default='*.pt', help='File pattern to match')
    list_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed information')
    
    # Find command
    find_parser = subparsers.add_parser('find', help='Find latest checkpoint for a run')
    find_parser.add_argument('pattern', help='Pattern to match in checkpoint path')
    find_parser.add_argument('--root', default='outputs', help='Root directory to search')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean up old checkpoints')
    clean_parser.add_argument('run_dir', help='Run directory to clean')
    clean_parser.add_argument('--keep-latest', type=int, default=3, help='Number of latest checkpoints to keep')
    clean_parser.add_argument('--no-keep-best', action='store_true', help='Do not keep best_policy checkpoint')
    clean_parser.add_argument('--no-dry-run', action='store_true', help='Actually delete files (default is dry run)')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        checkpoints = list_checkpoints(args.root, args.pattern)
        print_checkpoint_info(checkpoints, verbose=args.verbose)
    
    elif args.command == 'find':
        ckpt_path = find_latest_checkpoint(args.pattern, args.root)
        if ckpt_path:
            print(f"Latest checkpoint: {ckpt_path}")
        else:
            print(f"No checkpoints found matching: {args.pattern}")
            sys.exit(1)
    
    elif args.command == 'clean':
        clean_old_checkpoints(
            args.run_dir,
            keep_latest=args.keep_latest,
            keep_best=not args.no_keep_best,
            dry_run=not args.no_dry_run
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
