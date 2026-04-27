#!/usr/bin/env python3
"""
Alias entrypoint for online training across multiple simulators.

Keeps backward-compatible behavior by delegating to `scripts/train_dflex.py`.
"""

from train_dflex import train


if __name__ == "__main__":
    train()
