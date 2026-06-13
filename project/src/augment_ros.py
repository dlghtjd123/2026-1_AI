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
    if target_count <= int((y_train == class_id).sum()):
        return X_train, y_train
    sampler = RandomOverSampler(
        sampling_strategy={class_id: target_count},
        random_state=random_state,
    )
    return sampler.fit_resample(X_train, y_train)


def main() -> None:
    from run_single_augmentation_method import main as run_method

    if "--augments" not in sys.argv:
        sys.argv.extend(["--augments", "ros"])
    run_method("ros")


if __name__ == "__main__":
    main()
