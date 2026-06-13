"""
Training entrypoint for the CICIDS2017 Bot-only augmentation experiment.

This is the main CLI entrypoint. The shared automation loop lives in
bot_augmentation_experiment.py, while preprocessing, AE training,
augmentation methods, and evaluation are split into focused modules.
"""

from __future__ import annotations

from bot_augmentation_experiment import parse_args, run_training


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
