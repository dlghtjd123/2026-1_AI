from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from cicids2017_bot_config import OUT_DIR


AUGMENT_ORDER = [
    "none",
    "ros",
    "smote",
    "borderline_smote",
    "adasyn",
    "gan",
    "wgan_gp",
]

AUGMENT_LABELS = {
    "none": "None",
    "ros": "ROS",
    "smote": "SMOTE",
    "borderline_smote": "Borderline-SMOTE",
    "adasyn": "ADASYN",
    "gan": "GAN",
    "wgan_gp": "WGAN-GP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create bar charts from CICIDS2017 Bot augmentation summary.csv."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Run directory containing summary.csv. Defaults to the latest run_* directory.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="Direct path to summary.csv. Overrides --run_dir.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory where figures are saved. Defaults to <run_dir>/figures.",
    )
    parser.add_argument("--model", default="rf", help="Model name to plot.")
    parser.add_argument(
        "--feature_space",
        default="scaled_original",
        help="Feature space to plot, e.g. scaled_original or ae_latent.",
    )
    return parser.parse_args()


def find_latest_run_dir() -> Path:
    if not OUT_DIR.exists():
        raise FileNotFoundError(f"Result directory not found: {OUT_DIR}")
    run_dirs = sorted(
        [path for path in OUT_DIR.iterdir() if path.is_dir() and path.name.startswith("run_")],
        key=lambda path: path.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {OUT_DIR}")
    return run_dirs[-1]


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.summary_csv is not None:
        summary_csv = args.summary_csv
        run_dir = summary_csv.parent
    else:
        run_dir = args.run_dir or find_latest_run_dir()
        summary_csv = run_dir / "summary.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    out_dir = args.out_dir or (run_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, summary_csv, out_dir


def prepare_summary(summary_csv: Path, model: str, feature_space: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if "model" in df.columns:
        df = df[df["model"] == model]
    if "feature_space" in df.columns:
        df = df[df["feature_space"] == feature_space]

    if df.empty:
        raise ValueError(
            "No rows matched the selected filters. "
            f"model={model}, feature_space={feature_space}"
        )

    df = df.copy()
    df["augment_order"] = df["augment"].map(
        {augment: idx for idx, augment in enumerate(AUGMENT_ORDER)}
    )
    df["augment_order"] = df["augment_order"].fillna(len(AUGMENT_ORDER))
    df = df.sort_values(["augment_order", "augment"])
    df["augment_label"] = df["augment"].map(AUGMENT_LABELS).fillna(df["augment"])
    return df


def annotate_bars(ax: plt.Axes) -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            labels.append(f"{height:.3f}" if pd.notna(height) else "")
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)


def save_single_metric_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(df["augment_label"], df[metric], color=color)
    ax.set_title(title)
    ax.set_xlabel("Augmentation")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=25)
    annotate_bars(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_grouped_metric_bar(
    df: pd.DataFrame,
    metrics: list[str],
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    x = range(len(df))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        offsets = [value + (idx - (len(metrics) - 1) / 2) * width for value in x]
        ax.bar(offsets, df[metric], width=width, label=label)

    ax.set_title(title)
    ax.set_xlabel("Augmentation")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["augment_label"], rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    annotate_bars(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_figures(df: pd.DataFrame, out_dir: Path, model: str, feature_space: str) -> list[Path]:
    prefix = f"{model}_{feature_space}"
    saved_paths = []

    jobs = [
        (
            "macro_f1",
            "Macro F1 by Augmentation",
            "Macro F1",
            out_dir / f"{prefix}_macro_f1.png",
            "#4C78A8",
        ),
        (
            "bot_f1",
            "Bot F1 by Augmentation",
            "Bot F1",
            out_dir / f"{prefix}_bot_f1.png",
            "#F58518",
        ),
        (
            "bot_recall",
            "Bot Recall by Augmentation",
            "Bot Recall",
            out_dir / f"{prefix}_bot_recall.png",
            "#54A24B",
        ),
        (
            "bot_fnr",
            "Bot FNR by Augmentation",
            "Bot FNR",
            out_dir / f"{prefix}_bot_fnr.png",
            "#E45756",
        ),
    ]
    for metric, title, ylabel, out_path, color in jobs:
        save_single_metric_bar(df, metric, title, ylabel, out_path, color)
        saved_paths.append(out_path)

    grouped_path = out_dir / f"{prefix}_bot_precision_recall_f1.png"
    save_grouped_metric_bar(
        df,
        metrics=["bot_precision", "bot_recall", "bot_f1"],
        labels=["Bot Precision", "Bot Recall", "Bot F1"],
        title="Bot Precision / Recall / F1 by Augmentation",
        out_path=grouped_path,
    )
    saved_paths.append(grouped_path)

    overview_path = out_dir / f"{prefix}_macro_f1_bot_f1_fnr.png"
    save_grouped_metric_bar(
        df,
        metrics=["macro_f1", "bot_f1", "bot_fnr"],
        labels=["Macro F1", "Bot F1", "Bot FNR"],
        title="Macro F1 / Bot F1 / Bot FNR by Augmentation",
        out_path=overview_path,
    )
    saved_paths.append(overview_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    run_dir, summary_csv, out_dir = resolve_paths(args)
    df = prepare_summary(summary_csv, model=args.model, feature_space=args.feature_space)
    saved_paths = save_figures(df, out_dir, model=args.model, feature_space=args.feature_space)

    print(f"[SUMMARY] {summary_csv}")
    print(f"[FILTER] model={args.model} feature_space={args.feature_space}")
    print(f"[OUTPUT] {out_dir}")
    for path in saved_paths:
        print(f"[SAVED] {path}")


if __name__ == "__main__":
    main()
