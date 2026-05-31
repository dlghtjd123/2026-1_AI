"""
run_experiments.py

논문 실험 자동 실행 스크립트.

기본 흐름:
  dataset x augmentation 조합마다
    1) RF / XGBoost / CNN-LSTM / GRU / CNN-GRU 학습
    2) 최종 holdout test 평가

예시:
  python run_experiments.py --datasets cicids2017 --augments none smote
  python run_experiments.py --datasets cicids2017 --augments gan --fold_gan_epochs 100
  python run_experiments.py --datasets cicids2017 --augments smote --threshold_mode f1_opt
  python run_experiments.py --datasets cicids2017 --augments smote --augment_multiplier 5
  python run_experiments.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DATASETS = ["cicids2017", "cicids2018", "ctu13"]
AUGMENTS = ["none", "smote", "gan", "wcgan_gp"]
TRAIN_SCRIPTS = {
    "rf": "train_rf.py",
    "xgb": "train_xgb.py",
    "cnn_lstm": "train_cnn_lstm.py",
    "gru": "train_gru.py",
    "cnn_gru": "train_cnn_gru.py",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cicids2017"], choices=DATASETS)
    parser.add_argument("--augments", nargs="+", default=["none", "smote", "gan", "wcgan_gp"],
                        choices=AUGMENTS)
    parser.add_argument("--models", nargs="+", default=list(TRAIN_SCRIPTS.keys()),
                        choices=list(TRAIN_SCRIPTS.keys()))
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_normal", type=int, default=500_000)
    parser.add_argument("--max_mismatch", type=float, default=None,
                        help="기본값: cicids2017=5, cicids2018/ctu13=2")
    parser.add_argument("--threshold_mode", type=str, default="fixed",
                        choices=["fixed", "f1_opt"],
                        help="fixed=0.5 기준, f1_opt=fold validation F1 기준 threshold 선택")
    parser.add_argument("--augment_multiplier", type=float, default=2.0,
                        help="증강 후 Bot 수 목표 배수 (기본값: 2)")
    parser.add_argument("--fold_gan_epochs", type=int, default=None,
                        help="fold-local GAN epoch override")
    parser.add_argument("--fold_wcgan_epochs", type=int, default=None,
                        help="fold-local WCGAN-GP epoch override")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_evaluate", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def project_paths():
    src_dir = Path(__file__).resolve().parent
    project_dir = src_dir.parent
    root_dir = project_dir.parent
    return src_dir, project_dir, root_dir


def run_command(cmd: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> dict:
    started_at = datetime.now().isoformat(timespec="seconds")
    printable = " ".join(cmd)
    print(f"\n[RUN] {printable}")

    if dry_run:
        return {
            "command": printable,
            "returncode": 0,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": True,
        }

    result = subprocess.run(cmd, cwd=cwd, env=env)
    finished_at = datetime.now().isoformat(timespec="seconds")
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"[{status}] returncode={result.returncode}")

    return {
        "command": printable,
        "returncode": result.returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "dry_run": False,
    }


def resolve_max_mismatch(dataset: str, args) -> float:
    if args.max_mismatch is not None:
        return float(args.max_mismatch)
    if dataset == "cicids2017":
        return 5.0
    return 2.0


def train_command(src_dir: Path, script: str, dataset: str, augment: str, args) -> list[str]:
    max_mismatch = resolve_max_mismatch(dataset, args)
    cmd = [
        sys.executable,
        str(src_dir / script),
        "--dataset", dataset,
        "--augment", augment,
        "--n_folds", str(args.n_folds),
        "--max_normal", str(args.max_normal),
        "--max_mismatch", str(max_mismatch),
        "--threshold_mode", args.threshold_mode,
        "--augment_multiplier", str(args.augment_multiplier),
    ]
    if args.debug:
        cmd.append("--debug")
    return cmd


def evaluate_command(src_dir: Path, dataset: str, augment: str, args) -> list[str]:
    return [
        sys.executable,
        str(src_dir / "evaluate.py"),
        "--dataset", dataset,
        "--augment", augment,
        "--augment_multiplier", str(args.augment_multiplier),
    ]


def selected_models_cover_evaluation(models: list[str]) -> bool:
    return set(models) == set(TRAIN_SCRIPTS.keys())


def main():
    args = parse_args()
    src_dir, _project_dir, root_dir = project_paths()

    env = os.environ.copy()
    if args.fold_gan_epochs is not None:
        env["FOLD_GAN_EPOCHS"] = str(args.fold_gan_epochs)
    if args.fold_wcgan_epochs is not None:
        env["FOLD_WCGAN_EPOCHS"] = str(args.fold_wcgan_epochs)

    logs_dir = root_dir / "artifacts" / "experiment_runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{run_id}.json"

    print("=" * 72)
    print("  Automated Experiments")
    print("=" * 72)
    print(f"  datasets         : {args.datasets}")
    print(f"  augments         : {args.augments}")
    print(f"  models           : {args.models}")
    print(f"  n_folds          : {args.n_folds}")
    print(f"  max_normal       : {args.max_normal}")
    print(f"  max_mismatch     : {args.max_mismatch if args.max_mismatch is not None else 'dataset default (cicids2017=5, cicids2018/ctu13=2)'}")
    print(f"  threshold_mode   : {args.threshold_mode}")
    print(f"  augment_multiplier: {args.augment_multiplier:g}x")
    print(f"  FOLD_GAN_EPOCHS  : {env.get('FOLD_GAN_EPOCHS', '500')}")
    print(f"  FOLD_WCGAN_EPOCHS: {env.get('FOLD_WCGAN_EPOCHS', '500')}")
    print(f"  dry_run          : {args.dry_run}")
    print(f"  log              : {log_path}")
    print("=" * 72)

    run_log = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "commands": [],
    }

    try:
        for dataset in args.datasets:
            for augment in args.augments:
                print(f"\n{'=' * 72}")
                print(f"[EXPERIMENT] dataset={dataset} augment={augment}")
                print(f"{'=' * 72}")

                failed = False
                for model_name in args.models:
                    cmd = train_command(src_dir, TRAIN_SCRIPTS[model_name], dataset, augment, args)
                    entry = run_command(cmd, root_dir, env, args.dry_run)
                    entry.update({"dataset": dataset, "augment": augment, "stage": f"train_{model_name}"})
                    run_log["commands"].append(entry)
                    if entry["returncode"] != 0:
                        failed = True
                        if not args.continue_on_error:
                            raise RuntimeError(f"Failed: {entry['command']}")
                        break

                if failed:
                    continue

                if args.skip_evaluate:
                    continue

                if not selected_models_cover_evaluation(args.models):
                    print("[SKIP] evaluate.py는 5개 모델 파일을 모두 필요로 합니다. "
                          "선택 모델 일부만 실행했으므로 평가를 건너뜁니다.")
                    continue

                cmd = evaluate_command(src_dir, dataset, augment, args)
                entry = run_command(cmd, root_dir, env, args.dry_run)
                entry.update({"dataset": dataset, "augment": augment, "stage": "evaluate"})
                run_log["commands"].append(entry)
                if entry["returncode"] != 0 and not args.continue_on_error:
                    raise RuntimeError(f"Failed: {entry['command']}")

    finally:
        run_log["finished_at"] = datetime.now().isoformat(timespec="seconds")
        run_log["success"] = all(c["returncode"] == 0 for c in run_log["commands"])
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(run_log, f, indent=4, ensure_ascii=False)
        print(f"\n[LOG] {log_path}")

    if not run_log["success"]:
        sys.exit(1)

    print("\n[DONE] 모든 실험이 완료되었습니다.")


if __name__ == "__main__":
    main()
