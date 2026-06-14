"""
ADASYN으로 Bot 클래스를 증강하는 파일.

ADASYN은 분류가 어려운 소수 클래스 샘플 주변에 생성 샘플을 더 많이 만든다.
"""

from __future__ import annotations

import sys

import numpy as np
from imblearn.over_sampling import ADASYN


def augment_adasyn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_count: int,
    class_id: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ADASYN을 사용해 Bot 클래스의 샘플 수를 target_count 근처까지 늘린다.

    ADASYN은 난이도 기반으로 생성량을 조절하므로 최종 Bot 개수가 target_count와
    정확히 일치하지 않을 수 있다.
    """
    current = int((y_train == class_id).sum())
    if target_count <= current:
        return X_train, y_train
    if current < 2:
        raise ValueError("ADASYN은 대상 클래스 샘플이 최소 2개 이상 필요하다.")
    sampler = ADASYN(
        sampling_strategy={class_id: target_count},
        random_state=random_state,
        n_neighbors=min(5, current - 1),
    )
    return sampler.fit_resample(X_train, y_train)


def main() -> None:
    """이 파일을 직접 실행했을 때 ADASYN만 수행하도록 공통 실행기를 호출한다."""
    from run_single_augmentation_method import main as run_method

    if "--augments" not in sys.argv:
        sys.argv.extend(["--augments", "adasyn"])
    run_method("adasyn")


if __name__ == "__main__":
    main()
