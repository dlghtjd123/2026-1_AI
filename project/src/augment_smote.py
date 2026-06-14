"""
SMOTE로 Bot 클래스를 증강하는 파일.

기존 Bot 샘플 사이를 보간하여 생성 Bot 샘플을 만든다.
"""

from __future__ import annotations

import sys

import numpy as np
from imblearn.over_sampling import SMOTE


def augment_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_count: int,
    class_id: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SMOTE를 사용해 Bot class_id의 샘플 수를 target_count까지 늘린다.

    SMOTE는 이웃 샘플 간 보간을 사용하므로 Bot 샘플이 최소 2개 이상 필요하다.
    """
    current = int((y_train == class_id).sum())
    if target_count <= current:
        return X_train, y_train
    if current < 2:
        raise ValueError("SMOTE는 대상 클래스 샘플이 최소 2개 이상 필요하다.")
    sampler = SMOTE(
        sampling_strategy={class_id: target_count},
        random_state=random_state,
        k_neighbors=min(5, current - 1),
    )
    return sampler.fit_resample(X_train, y_train)


def main() -> None:
    """이 파일을 직접 실행했을 때 SMOTE만 수행하도록 공통 실행기를 호출한다."""
    from run_single_augmentation_method import main as run_method

    if "--augments" not in sys.argv:
        sys.argv.extend(["--augments", "smote"])
    run_method("smote")


if __name__ == "__main__":
    main()
