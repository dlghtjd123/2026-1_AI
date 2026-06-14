"""
CICIDS2017 Bot 증강 실험 결과를 막대그래프로 시각화하는 파일.

summary.csv를 읽어 Macro F1, Bot F1, Bot Recall, Bot FNR 등을
증강 방식별 막대그래프로 저장한다.
"""

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
    """
    시각화 대상 실행 폴더, summary.csv, 출력 폴더, 모델, feature 공간 옵션을 정의한다.
    """
    parser = argparse.ArgumentParser(
        description="CICIDS2017 Bot 증강 실험 summary.csv로 막대그래프를 생성한다."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="summary.csv가 들어 있는 실행 폴더. 기본값은 가장 최근 run_* 폴더.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="summary.csv의 직접 경로. 이 값이 있으면 --run_dir보다 우선한다.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="그래프를 저장할 폴더. 기본값은 <run_dir>/figures.",
    )
    parser.add_argument("--model", default="rf", help="그래프로 그릴 모델 이름.")
    parser.add_argument(
        "--feature_space",
        default="scaled_original",
        help="그래프로 그릴 feature 공간. 예: scaled_original 또는 ae_latent.",
    )
    return parser.parse_args()


def find_latest_run_dir() -> Path:
    """
    artifacts 결과 폴더에서 가장 최근 run_* 디렉터리를 찾는다.

    --run_dir나 --summary_csv를 지정하지 않았을 때 기본 입력으로 사용한다.
    """
    if not OUT_DIR.exists():
        raise FileNotFoundError(f"결과 폴더를 찾을 수 없다: {OUT_DIR}")
    run_dirs = sorted(
        [path for path in OUT_DIR.iterdir() if path.is_dir() and path.name.startswith("run_")],
        key=lambda path: path.stat().st_mtime,
    )
    if not run_dirs:
        raise FileNotFoundError(f"{OUT_DIR} 안에서 run_* 폴더를 찾을 수 없다.")
    return run_dirs[-1]


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """
    입력 summary.csv 경로와 그래프 출력 폴더를 결정한다.

    summary_csv를 직접 지정하면 그 파일을 우선하고, 그렇지 않으면 run_dir 안의
    summary.csv를 사용한다.
    """
    if args.summary_csv is not None:
        summary_csv = args.summary_csv
        run_dir = summary_csv.parent
    else:
        run_dir = args.run_dir or find_latest_run_dir()
        summary_csv = run_dir / "summary.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv를 찾을 수 없다: {summary_csv}")

    out_dir = args.out_dir or (run_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, summary_csv, out_dir


def prepare_summary(summary_csv: Path, model: str, feature_space: str) -> pd.DataFrame:
    """
    summary.csv에서 지정한 model과 feature_space에 해당하는 행만 추출한다.

    그래프가 항상 같은 순서로 나오도록 증강 방식 순서도 함께 정렬한다.
    """
    df = pd.read_csv(summary_csv)
    if "model" in df.columns:
        df = df[df["model"] == model]
    if "feature_space" in df.columns:
        df = df[df["feature_space"] == feature_space]

    if df.empty:
        raise ValueError(
            "선택한 조건에 맞는 행이 없다. "
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
    """막대그래프 각 막대 위에 소수점 3자리 값을 표시한다."""
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
    """하나의 지표를 증강 방식별 단일 막대그래프로 저장한다."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(df["augment_label"], df[metric], color=color)
    ax.set_title(title)
    ax.set_xlabel("증강 방식")
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
    """여러 지표를 한 그림에서 비교할 수 있는 묶음 막대그래프로 저장한다."""
    x = range(len(df))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        offsets = [value + (idx - (len(metrics) - 1) / 2) * width for value in x]
        ax.bar(offsets, df[metric], width=width, label=label)

    ax.set_title(title)
    ax.set_xlabel("증강 방식")
    ax.set_ylabel("점수")
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
    """
    제출용 주요 그래프들을 한 번에 생성하고 저장 경로 목록을 반환한다.

    단일 지표 그래프와 Bot 정밀도/재현율/F1 비교 그래프를 함께 만든다.
    """
    prefix = f"{model}_{feature_space}"
    saved_paths = []

    jobs = [
        (
            "macro_f1",
            "증강 방식별 Macro F1",
            "Macro F1",
            out_dir / f"{prefix}_macro_f1.png",
            "#4C78A8",
        ),
        (
            "bot_f1",
            "증강 방식별 Bot F1",
            "Bot F1",
            out_dir / f"{prefix}_bot_f1.png",
            "#F58518",
        ),
        (
            "bot_recall",
            "증강 방식별 Bot Recall",
            "Bot Recall",
            out_dir / f"{prefix}_bot_recall.png",
            "#54A24B",
        ),
        (
            "bot_fnr",
            "증강 방식별 Bot FNR",
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
        title="증강 방식별 Bot Precision / Recall / F1",
        out_path=grouped_path,
    )
    saved_paths.append(grouped_path)

    overview_path = out_dir / f"{prefix}_macro_f1_bot_f1_fnr.png"
    save_grouped_metric_bar(
        df,
        metrics=["macro_f1", "bot_f1", "bot_fnr"],
        labels=["Macro F1", "Bot F1", "Bot FNR"],
        title="증강 방식별 Macro F1 / Bot F1 / Bot FNR",
        out_path=overview_path,
    )
    saved_paths.append(overview_path)

    return saved_paths


def main() -> None:
    """명령행 인자를 받아 summary.csv를 읽고 모든 그래프를 생성한다."""
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
