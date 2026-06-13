"""
bot_augmentation_experiment.py

Fast no-K-fold reproduction-style experiment inspired by AE-CGAN IDS.

Protocol:
  CICIDS2017 13-class data
    -> train/test split
    -> MinMax scaling fitted on train only
    -> optional Autoencoder fitted on train only
    -> compare scaled-original and/or AE latent feature spaces
    -> augment Bot class only
    -> RF 13-class classification
    -> report overall metrics and Bot-class metrics

This script intentionally avoids K-fold for the first screening experiment.
If Bot metrics improve, run K-fold later as a follow-up validation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from cicids2017_bot_config import (
    BOT_CLASS_ID,
    CLASS_NAMES,
    OUT_DIR,
    RANDOM_STATE,
)
from rf_evaluation import evaluate_rf
from cicids2017_preprocessing import (
    clean_features,
    infer_feature_columns,
    load_cicids2017,
    stratified_debug_sample,
)
from autoencoder_features import Autoencoder, encode_features, train_autoencoder
from augment_adasyn import augment_adasyn
from augment_borderline_smote import augment_borderline_smote
from augment_gan import augment_gan
from augment_ros import augment_ros
from augment_smote import augment_smote
from augment_wgan_gp import augment_wgan_gp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augments",
        nargs="+",
        default=["none", "smote", "borderline_smote", "adasyn", "gan", "wgan_gp"],
        choices=["none", "ros", "smote", "borderline_smote", "adasyn", "gan", "wgan_gp"],
    )
    parser.add_argument("--test_size", type=float, default=0.4)
    parser.add_argument("--latent_dim", type=int, default=40)
    parser.add_argument("--ae_epochs", type=int, default=20)
    parser.add_argument("--gan_epochs", type=int, default=20)
    parser.add_argument("--wgan_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--gan_batch_size", type=int, default=64)
    parser.add_argument("--wgan_batch_size", type=int, default=64)
    parser.add_argument("--noise_dim", type=int, default=256)
    parser.add_argument("--wgan_n_critic", type=int, default=5)
    parser.add_argument("--wgan_lambda_gp", type=float, default=10.0)
    parser.add_argument("--target_bot_count", type=int, default=10_000)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf"],
        choices=["rf"],
        help="Classifier to train for each feature-space/augmentation pair.",
    )
    parser.add_argument("--rf_estimators", type=int, default=100)
    parser.add_argument(
        "--preprocess",
        choices=["paper", "log1p"],
        default="paper",
        help="paper=MinMax only, log1p=log1p selected skewed flow features before MinMax.",
    )
    parser.add_argument(
        "--feature_spaces",
        nargs="+",
        default=["ae"],
        choices=["raw", "ae"],
        help="Feature spaces to evaluate: raw=scaled original, ae=AE latent.",
    )
    parser.add_argument(
        "--include_single_rf",
        action="store_true",
        help="Deprecated alias for --feature_spaces raw ae.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional stratified debug sample before train/test split.",
    )
    parser.add_argument(
        "--skip_ae",
        action="store_true",
        help="Use scaled original features instead of AE latent features.",
    )
    return parser.parse_args()


def target_bot_count(y_train: np.ndarray, requested: int) -> int:
    current = int((y_train == BOT_CLASS_ID).sum())
    if requested <= current:
        return current
    return requested


def _sample_rows(X: np.ndarray, max_rows: int, seed_offset: int = 0) -> np.ndarray:
    if len(X) <= max_rows:
        return X
    rng = np.random.RandomState(RANDOM_STATE + seed_offset)
    idx = rng.choice(len(X), size=max_rows, replace=False)
    return X[idx]


def compute_synthetic_diagnostics(
    X_real_bot: np.ndarray,
    X_fake_bot: np.ndarray,
    feature_space: str,
    augment: str,
    max_probe_rows: int = 5000,
) -> dict:
    real = _sample_rows(X_real_bot.astype(np.float32), max_probe_rows, seed_offset=11)
    fake = _sample_rows(X_fake_bot.astype(np.float32), max_probe_rows, seed_offset=23)
    eps = 1e-8

    real_std = real.std(axis=0)
    fake_std = fake.std(axis=0)
    active = real_std > eps

    nn_real = NearestNeighbors(n_neighbors=1).fit(real)
    fake_to_real = nn_real.kneighbors(fake, return_distance=True)[0].ravel()
    nn_fake = NearestNeighbors(n_neighbors=1).fit(fake)
    real_to_fake = nn_fake.kneighbors(real, return_distance=True)[0].ravel()

    fake_self_k = 2 if len(fake) > 1 else 1
    real_self_k = 2 if len(real) > 1 else 1
    fake_self = NearestNeighbors(n_neighbors=fake_self_k).fit(fake).kneighbors(
        fake, return_distance=True
    )[0][:, -1]
    real_self = NearestNeighbors(n_neighbors=real_self_k).fit(real).kneighbors(
        real, return_distance=True
    )[0][:, -1]

    X_probe = np.vstack([real, fake])
    y_probe = np.concatenate([
        np.zeros(len(real), dtype=np.int32),
        np.ones(len(fake), dtype=np.int32),
    ])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_probe,
        y_probe,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y_probe,
    )
    detector = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    detector.fit(X_tr, y_tr)
    fake_prob = detector.predict_proba(X_te)[:, 1]
    real_fake_auc = roc_auc_score(y_te, fake_prob)

    return {
        "feature_space": feature_space,
        "augment": augment,
        "real_bot_count": int(len(X_real_bot)),
        "fake_bot_count": int(len(X_fake_bot)),
        "probe_real_count": int(len(real)),
        "probe_fake_count": int(len(fake)),
        "real_fake_auc": float(real_fake_auc),
        "fake_to_real_nn_mean": float(fake_to_real.mean()),
        "fake_to_real_nn_p50": float(np.percentile(fake_to_real, 50)),
        "fake_to_real_nn_p95": float(np.percentile(fake_to_real, 95)),
        "real_to_fake_nn_mean": float(real_to_fake.mean()),
        "real_to_fake_nn_p95": float(np.percentile(real_to_fake, 95)),
        "fake_self_nn_mean": float(fake_self.mean()),
        "real_self_nn_mean": float(real_self.mean()),
        "fake_self_nn_ratio": float(fake_self.mean() / (real_self.mean() + eps)),
        "mean_abs_mean_diff": float(np.mean(np.abs(fake.mean(axis=0) - real.mean(axis=0)))),
        "mean_std_ratio": float(np.mean(fake_std[active] / (real_std[active] + eps))) if np.any(active) else 0.0,
        "low_variance_feature_ratio": float(np.mean(fake_std < (real_std * 0.1 + eps))),
        "range_violation_rate": float(np.mean((fake < -1e-6) | (fake > 1.0 + 1e-6))),
        "near_duplicate_real_rate": float(np.mean(fake_to_real < 1e-6)),
    }


def save_outputs(
    results: dict,
    summary_rows: list[dict],
    synthetic_diagnostic_rows: list[dict],
    scaler: MinMaxScaler,
    feature_cols: list[str],
    ae_model: Autoencoder | None,
    trained_models: dict[str, object],
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "results.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=4, ensure_ascii=False)
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary.csv", index=False)
    if synthetic_diagnostic_rows:
        pd.DataFrame(synthetic_diagnostic_rows).to_csv(
            run_dir / "synthetic_diagnostics.csv", index=False
        )
    with open(run_dir / "features.json", "w", encoding="utf-8") as fp:
        json.dump({"features": feature_cols, "classes": CLASS_NAMES}, fp, indent=4, ensure_ascii=False)
    joblib.dump(scaler, run_dir / "scaler.pkl")
    for key, model in trained_models.items():
        joblib.dump(model, run_dir / f"{key}.pkl")
    if ae_model is not None:
        torch.save(ae_model.state_dict(), run_dir / "autoencoder.pt")
    return run_dir


def run_training(args: argparse.Namespace) -> Path:
    if args.include_single_rf and "raw" not in args.feature_spaces:
        args.feature_spaces = ["raw", *args.feature_spaces]
    args.feature_spaces = list(dict.fromkeys(args.feature_spaces))

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_cicids2017()
    feature_cols = infer_feature_columns(df)
    print(f"[FEATURE] count={len(feature_cols)} preprocess={args.preprocess}")
    df = clean_features(df, feature_cols, args.preprocess)
    df = stratified_debug_sample(df, args.max_rows)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["class_id"].to_numpy(dtype=np.int32)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

    feature_space_data = {}
    if "raw" in args.feature_spaces or args.skip_ae:
        feature_space_data["raw"] = (X_train_scaled, X_test_scaled)

    ae_model = None
    if "ae" in args.feature_spaces and not args.skip_ae:
        ae_model = train_autoencoder(
            X_train_scaled,
            latent_dim=args.latent_dim,
            epochs=args.ae_epochs,
            batch_size=args.batch_size,
            device=device,
        )
        feature_space_data["ae"] = (
            encode_features(ae_model, X_train_scaled, device, args.batch_size),
            encode_features(ae_model, X_test_scaled, device, args.batch_size),
        )

    if args.skip_ae and "ae" in args.feature_spaces:
        print("[WARN] --skip_ae was set, so feature_space='ae' is skipped.")
        args.feature_spaces = [space for space in args.feature_spaces if space != "ae"]

    target_count = target_bot_count(y_train, args.target_bot_count)
    print(f"[TARGET] train Bot current={(y_train == BOT_CLASS_ID).sum():,} target={target_count:,}")

    results = {
        "protocol": "CICIDS2017 13-class RF; Bot class only augmented; no K-fold",
        "args": vars(args),
        "classes": CLASS_NAMES,
        "feature_count": len(feature_cols),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_distribution": {
            CLASS_NAMES[i]: int((y_train == i).sum()) for i in range(len(CLASS_NAMES))
        },
        "test_distribution": {
            CLASS_NAMES[i]: int((y_test == i).sum()) for i in range(len(CLASS_NAMES))
        },
        "results": {},
        "synthetic_diagnostics": [],
    }
    summary_rows = []
    synthetic_diagnostic_rows = []
    trained_models = {}

    for feature_space in args.feature_spaces:
        X_train_base, X_test_base = feature_space_data[feature_space]
        feature_label = "scaled_original" if feature_space == "raw" else "ae_latent"

        for augment in args.augments:
            print("=" * 72)
            print(f"[EXPERIMENT] feature_space={feature_space} augment={augment}")
            print("=" * 72)
            generator = None
            X_fake_bot = None
            if augment == "none":
                X_train_aug, y_train_aug = X_train_base, y_train
            elif augment == "ros":
                X_train_aug, y_train_aug = augment_ros(
                    X_train_base, y_train, target_count, BOT_CLASS_ID, RANDOM_STATE
                )
            elif augment == "smote":
                X_train_aug, y_train_aug = augment_smote(
                    X_train_base, y_train, target_count, BOT_CLASS_ID, RANDOM_STATE
                )
            elif augment == "borderline_smote":
                X_train_aug, y_train_aug = augment_borderline_smote(
                    X_train_base, y_train, target_count, BOT_CLASS_ID, RANDOM_STATE
                )
            elif augment == "adasyn":
                X_train_aug, y_train_aug = augment_adasyn(
                    X_train_base, y_train, target_count, BOT_CLASS_ID, RANDOM_STATE
                )
            elif augment == "gan":
                X_train_aug, y_train_aug, generator, X_fake_bot = augment_gan(
                    X_train_base,
                    y_train,
                    target_count,
                    noise_dim=args.noise_dim,
                    epochs=args.gan_epochs,
                    batch_size=args.gan_batch_size,
                    device=device,
                )
            elif augment == "wgan_gp":
                X_train_aug, y_train_aug, generator, X_fake_bot = augment_wgan_gp(
                    X_train_base,
                    y_train,
                    target_count,
                    noise_dim=args.noise_dim,
                    epochs=args.wgan_epochs,
                    batch_size=args.wgan_batch_size,
                    n_critic=args.wgan_n_critic,
                    lambda_gp=args.wgan_lambda_gp,
                    device=device,
                )
            else:
                raise ValueError(augment)

            if X_fake_bot is not None:
                diag = compute_synthetic_diagnostics(
                    X_train_base[y_train == BOT_CLASS_ID],
                    X_fake_bot,
                    feature_space=feature_label,
                    augment=augment,
                )
                synthetic_diagnostic_rows.append(diag)
                results["synthetic_diagnostics"].append(diag)
                print(
                    "[SYNTH_DIAG] "
                    f"real_fake_auc={diag['real_fake_auc']:.4f} "
                    f"fake_to_real_nn_mean={diag['fake_to_real_nn_mean']:.4f} "
                    f"fake_self_nn_ratio={diag['fake_self_nn_ratio']:.4f} "
                    f"mean_std_ratio={diag['mean_std_ratio']:.4f}"
                )

            print(
                f"[TRAIN] size={len(y_train_aug):,} "
                f"Bot={(y_train_aug == BOT_CLASS_ID).sum():,}"
            )
            if generator is not None:
                OUT_DIR.mkdir(parents=True, exist_ok=True)

            for model_name in args.models:
                if model_name == "rf":
                    trained_model, metrics = evaluate_rf(
                        X_train_aug,
                        y_train_aug,
                        X_test_base,
                        y_test,
                        n_estimators=args.rf_estimators,
                    )
                else:
                    raise ValueError(model_name)

                result_key = f"{model_name}_{feature_space}_{augment}"
                trained_models[result_key] = trained_model
                results["results"][result_key] = {
                    "model": model_name,
                    "feature_space": feature_label,
                    "augment": augment,
                    "train_size": int(len(y_train_aug)),
                    "train_bot_count": int((y_train_aug == BOT_CLASS_ID).sum()),
                    **metrics,
                }
                summary_rows.append(
                    {
                        "model": model_name,
                        "augment": augment,
                        "feature_space": feature_label,
                        "accuracy": metrics["accuracy"],
                        "macro_precision": metrics["macro_precision"],
                        "macro_recall": metrics["macro_recall"],
                        "macro_f1": metrics["macro_f1"],
                        "weighted_f1": metrics["weighted_f1"],
                        "bot_precision": metrics["bot"]["precision"],
                        "bot_recall": metrics["bot"]["recall"],
                        "bot_f1": metrics["bot"]["f1"],
                        "bot_fnr": metrics["bot"]["fnr"],
                        "bot_fpr": metrics["bot"]["fpr"],
                        "bot_support": metrics["bot"]["support"],
                        "train_size": int(len(y_train_aug)),
                        "train_bot_count": int((y_train_aug == BOT_CLASS_ID).sum()),
                    }
                )
                print(
                    f"[RESULT] model={model_name} feature_space={feature_space} "
                    f"augment={augment} macro_f1={metrics['macro_f1']:.4f} "
                    f"bot_recall={metrics['bot']['recall']:.4f} "
                    f"bot_f1={metrics['bot']['f1']:.4f}"
                )

    run_dir = save_outputs(
        results,
        summary_rows,
        synthetic_diagnostic_rows,
        scaler,
        feature_cols,
        ae_model,
        trained_models,
    )
    print(f"\n[SAVED] {run_dir}")
    return run_dir


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
