from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

DATASETS = ["cicids2017", "cicids2018", "ctu13"]
AUGMENTS = ["none", "smote", "gan", "wcgan_gp"]
MODELS = ["rf", "xgb", "cnn_lstm", "gru", "cnn_gru"]
MODEL_LABELS = {
    "rf": "RF",
    "xgb": "XGBoost",
    "cnn_lstm": "CNN-LSTM",
    "gru": "GRU",
    "cnn_gru": "CNN-GRU",
}


def result_dir(dataset: str, augment: str) -> Path:
    suffix = "" if augment == "none" else f"_{augment}"
    return ARTIFACTS / f"results_{dataset}{suffix}"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def metric_value(value):
    if isinstance(value, dict):
        return value.get("mean")
    return value


def rows_from_kfold() -> list[dict]:
    rows: list[dict] = []
    for dataset in DATASETS:
        for augment in AUGMENTS:
            base = result_dir(dataset, augment)
            for model in MODELS:
                path = base / f"{model}_flow_kfold_results.json"
                data = load_json(path)
                if not data:
                    continue
                summary = data.get("summary", {})
                rows.append(
                    {
                        "stage": "kfold",
                        "dataset": dataset,
                        "augment": augment,
                        "model": MODEL_LABELS[model],
                        "f1": metric_value(summary.get("f1")),
                        "recall": metric_value(summary.get("recall")),
                        "precision": metric_value(summary.get("precision")),
                        "roc_auc": metric_value(summary.get("roc_auc")),
                        "accuracy": metric_value(summary.get("accuracy")),
                        "source": str(path.relative_to(ROOT)),
                    }
                )
    return rows


def rows_from_eval() -> list[dict]:
    rows: list[dict] = []
    for dataset in DATASETS:
        for augment in AUGMENTS:
            path = result_dir(dataset, augment) / "eval_results.json"
            data = load_json(path)
            if not data:
                continue
            for model, metrics in data.get("test_results", {}).items():
                rows.append(
                    {
                        "stage": "holdout",
                        "dataset": dataset,
                        "augment": augment,
                        "model": MODEL_LABELS.get(model, model),
                        "f1": metrics.get("f1"),
                        "recall": metrics.get("recall"),
                        "precision": metrics.get("precision"),
                        "roc_auc": metrics.get("roc_auc"),
                        "accuracy": metrics.get("accuracy"),
                        "source": str(path.relative_to(ROOT)),
                    }
                )
    return rows


def fmt(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "stage",
        "dataset",
        "augment",
        "model",
        "f1",
        "recall",
        "precision",
        "roc_auc",
        "accuracy",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Experiment Summary",
        "",
        "This file is generated from JSON artifacts under `artifacts/results_*`.",
        "",
    ]
    for stage in ["kfold", "holdout"]:
        stage_rows = [r for r in rows if r["stage"] == stage]
        lines.extend([f"## {stage.upper()} Results", ""])
        if not stage_rows:
            lines.extend(["No results found yet.", ""])
            continue
        for dataset in DATASETS:
            dataset_rows = [r for r in stage_rows if r["dataset"] == dataset]
            if not dataset_rows:
                continue
            lines.extend([f"### {dataset}", ""])
            lines.append("| Augment | Model | F1 | Recall | Precision | ROC-AUC | Accuracy |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for augment in AUGMENTS:
                for model in [MODEL_LABELS[m] for m in MODELS]:
                    match = [
                        r
                        for r in dataset_rows
                        if r["augment"] == augment and r["model"] == model
                    ]
                    if not match:
                        continue
                    r = match[0]
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                augment,
                                model,
                                fmt(r["f1"]),
                                fmt(r["recall"]),
                                fmt(r["precision"]),
                                fmt(r["roc_auc"]),
                                fmt(r["accuracy"]),
                            ]
                        )
                        + " |"
                    )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_cicids2017_deep(min_f1: float) -> int:
    failed = []
    for model in ["cnn_lstm", "gru", "cnn_gru"]:
        path = result_dir("cicids2017", "none") / f"{model}_flow_kfold_results.json"
        data = load_json(path)
        f1 = None
        if data:
            f1 = metric_value(data.get("summary", {}).get("f1"))
        if f1 is None or float(f1) < min_f1:
            failed.append((model, f1))
    if failed:
        for model, f1 in failed:
            print(f"FAIL {model}: f1={f1}")
        return 1
    print("OK cicids2017 deep baseline")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-cicids2017-deep", action="store_true")
    parser.add_argument("--min-f1", type=float, default=0.2)
    args = parser.parse_args()

    if args.validate_cicids2017_deep:
        return validate_cicids2017_deep(args.min_f1)

    rows = rows_from_kfold() + rows_from_eval()
    write_csv(rows, ARTIFACTS / "paper_summary.csv")
    write_markdown(rows, ARTIFACTS / "paper_summary.md")
    print(f"Wrote {ARTIFACTS / 'paper_summary.csv'}")
    print(f"Wrote {ARTIFACTS / 'paper_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
