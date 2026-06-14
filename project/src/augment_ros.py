"""
Random Oversampling(ROS)으로 Bot 클래스를 증강하는 파일.

새로운 값을 합성하지 않고 기존 Bot 샘플을 복원추출하여 목표 개수까지 늘린다.
"""

from __future__ import annotations

import sys

import numpy as np
from imblearn.over_sampling import RandomOverSampler


def augment_ros(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_count: int,
    class_id: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bot 클래스에 해당하는 샘플을 복원추출하여 target_count까지 늘린다.

    이미 현재 Bot 개수가 target_count 이상이면 원본 학습 세트를 그대로 반환한다.
    """
    if target_count <= int((y_train == class_id).sum()):
        return X_train, y_train
    sampler = RandomOverSampler(
        sampling_strategy={class_id: target_count},
        random_state=random_state,
    )
    return sampler.fit_resample(X_train, y_train)


def main() -> None:
    """
    이 파일을 직접 실행했을 때 ROS만 수행하도록 공통 실행기를 호출한다.
    """
    from run_single_augmentation_method import main as run_method

    if "--augments" not in sys.argv:
        sys.argv.extend(["--augments", "ros"])
    run_method("ros")


if __name__ == "__main__":
    main()
