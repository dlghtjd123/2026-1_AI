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
    current = int((y_train == class_id).sum())
    if target_count <= current:
        return X_train, y_train
    if current < 2:
        raise ValueError("SMOTE requires at least 2 target-class samples.")
    sampler = SMOTE(
        sampling_strategy={class_id: target_count},
        random_state=random_state,
        k_neighbors=min(5, current - 1),
    )
    return sampler.fit_resample(X_train, y_train)


def main() -> None:
    from train_cicids2017_bot_method import main as run_method

    if "--augments" not in sys.argv:
        sys.argv.extend(["--augments", "smote"])
    run_method("smote")


if __name__ == "__main__":
    main()
