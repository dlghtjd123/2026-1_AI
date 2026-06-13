"""
Training entrypoint for the CICIDS2017 Bot-only augmentation experiment.

This is the main CLI entrypoint. The shared automation loop lives in
run_cicids2017_ae_cgan_bot_multiclass.py, while preprocessing, AE training,
augmentation methods, and evaluation are split into focused modules.
"""

from __future__ import annotations

from run_cicids2017_ae_cgan_bot_multiclass import parse_args, run_training


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
