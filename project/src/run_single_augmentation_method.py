"""
증강 방식 하나만 실행할 때 사용하는 공통 실행 파일.

각 augment_*.py 파일에서 직접 실행할 때 기본 증강 방식을 지정하고,
전체 실험 루프는 bot_augmentation_experiment.py에 위임한다.
"""

from __future__ import annotations

import sys

from bot_augmentation_experiment import parse_args, run_training


def main(default_augment: str) -> None:
    """
    특정 증강 방식 하나만 실행하기 위한 공통 실행기.

    각 augment_*.py 파일에서 직접 실행할 때 --augments 옵션이 없으면
    default_augment 하나만 선택하여 전체 실험 루프를 호출한다.
    """
    args = parse_args()
    if "--augments" not in sys.argv:
        args.augments = [default_augment]
    run_training(args)
