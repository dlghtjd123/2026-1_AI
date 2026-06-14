"""
CICIDS2017 Bot-only 증강 실험을 실행하는 메인 파일.

명령행에서 실행되는 진입점이며, 실제 자동화 루프는
bot_augmentation_experiment.py에 있다. 전처리, AE 학습,
증강 방식, 평가는 기능별 모듈로 분리되어 있다.
"""

from __future__ import annotations

from bot_augmentation_experiment import parse_args, run_training


def main() -> None:
    """
    CICIDS2017 Bot-only 증강 실험의 메인 명령행 진입점.

    명령행 인자를 읽고 bot_augmentation_experiment.run_training()에 전달한다.
    """
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
