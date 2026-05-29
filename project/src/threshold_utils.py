from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_threshold(y_true, y_prob, mode: str = "fixed") -> float:
    if mode == "fixed":
        return 0.5
    if mode != "f1_opt":
        raise ValueError(f"Unknown threshold mode: {mode}")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx])


def threshold_label(threshold: float, mode: str = "fixed") -> str:
    return "argmax" if mode == "fixed" and float(threshold) == 0.5 else f"{float(threshold):.4f}"
