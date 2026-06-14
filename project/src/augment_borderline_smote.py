"""
Borderline-SMOTE로 Bot 클래스를 증강하는 파일.

일반 SMOTE와 달리 결정 경계 근처의 소수 클래스 샘플을 중심으로
생성 Bot 샘플을 만든다.
"""

from __future__ import annotations

import sys

import numpy as np
from imblearn.over_sampling import BorderlineSMOTE


def augment_borderline_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_count: int,
    class_id: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Borderline-SMOTE를 사용해 Bot 클래스의 샘플 수를 target_count까지 늘린다.

    Bot과 다른 클래스가 섞이는 경계 부근의 Bot 샘플을 중심으로 보간하기 때문에
    Bot Recall 향상 여부를 확인하기 위한 비교 증강 기법으로 사용한다.
    """
    current = int((y_train == class_id).sum())
    if target_count <= current:
        return X_train, y_train
    if current < 2:
        raise ValueError("Borderline-SMOTE는 대상 클래스 샘플이 최소 2개 이상 필요하다.")
    sampler = BorderlineSMOTE(
        sampling_strategy={class_id: target_count},
        random_state=random_state,
        k_neighbors=min(5, current - 1),
        m_neighbors=min(10, current - 1),
        kind="borderline-1",
    )
    return sampler.fit_resample(X_train, y_train)


def main() -> None:
    """
    이 파일을 직접 실행했을 때 Borderline-SMOTE만 수행하도록 공통 실행기를 호출한다.
    """
    from run_single_augmentation_method import main as run_method

    if "--augments" not in sys.argv:
        sys.argv.extend(["--augments", "borderline_smote"])
    run_method("borderline_smote")


if __name__ == "__main__":
    main()
