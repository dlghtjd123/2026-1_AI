"""
Shared launcher for one-method CICIDS2017 Bot augmentation training scripts.
"""

from __future__ import annotations

import sys

from bot_augmentation_experiment import parse_args, run_training


def main(default_augment: str) -> None:
    args = parse_args()
    if "--augments" not in sys.argv:
        args.augments = [default_augment]
    run_training(args)
