"""
debug_augmentation_quality.py

SMOTE / GAN / WCGAN-GP 생성 Botnet 샘플 품질 진단 스크립트

확인 항목:
1. 실제 Botnet vs 생성 Botnet feature 평균/표준편차 비교
2. PCA 2D 시각화
3. TSTR 평가
   - Train on Synthetic Botnet + Real Normal
   - Test on Real validation fold

사용법:
  python debug_augmentation_quality.py --dataset cicids2017 --augment smote
  python debug_augmentation_quality.py --dataset cicids2017 --augment gan
  python debug_augmentation_quality.py --dataset cicids2017 --augment wcgan_gp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# 인자 파싱
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="cicids2017",
    choices=["cicids2017", "cicids2018", "ctu13"],
)
parser.add_argument(
    "--augment",
    type=str,
    default="smote",
    choices=["smote", "gan", "wgan_gp", "wcgan_gp"],
)
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--target_ratio", type=float, default=0.005)
parser.add_argument("--n_vis", type=int, default=2000)
args = parser.parse_args()

DATASET = args.dataset
AUGMENT = args.augment
FOLD = args.fold
TARGET_RATIO = args.target_ratio
N_VIS = args.n_vis


# =========================================================
# 경로 설정
# =========================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT = SRC_DIR.parent
DATA_ROOT = PROJECT / "data" / "processed"
FLAT_DIR = DATA_ROOT / DATASET / "flat"

if AUGMENT == "smote":
    GEN_PATH = None
    CONDITIONAL = False
elif AUGMENT == "gan":
    GEN_PATH = DATA_ROOT / f"{DATASET}_gan" / "generator.pt"
    CONDITIONAL = False
else:
    GEN_PATH = DATA_ROOT / f"{DATASET}_wcgan_gp" / "generator_wcgan_gp.pt"
    CONDITIONAL = True


# =========================================================
# SMOTE synthetic Botnet 생성
# =========================================================
def generate_smote_botnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_ratio: float,
) -> np.ndarray:
    from imblearn.over_sampling import SMOTE

    n_majority = int((y_train == 0).sum())
    n_current = int(y_train.sum())
    n_target = int(n_majority * target_ratio / (1 - target_ratio))

    if n_target <= n_current:
        raise ValueError(
            f"SMOTE 생성 불필요: 현재 Bot={n_current:,}, 목표 Bot={n_target:,}"
        )

    smote = SMOTE(
        sampling_strategy={1: n_target},
        random_state=42,
        k_neighbors=min(5, n_current - 1),
    )

    X_aug, y_aug = smote.fit_resample(X_train, y_train)

    # imblearn SMOTE는 일반적으로 원본 뒤에 synthetic sample을 추가함
    X_added = X_aug[len(X_train):]
    y_added = y_aug[len(y_train):]

    fake_bot = X_added[y_added == 1].astype(np.float32)

    print("\n[SMOTE]")
    print(f"원본 Bot : {n_current:,}")
    print(f"목표 Bot : {n_target:,}")
    print(f"생성 Bot : {len(fake_bot):,}")
    print(f"증강 후 Bot 비율 예상: {n_target / (n_majority + n_target):.6f}")

    return fake_bot


# =========================================================
# Generator 정의
# augment_utils.py와 구조 동일해야 함
# =========================================================
def make_gan_generator(noise_dim: int, n_features: int):
    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(noise_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, n_features),
                nn.Sigmoid(),
            )

        def forward(self, z):
            return self.net(z)

    return Generator()


def make_wcgan_generator(noise_dim: int, label_dim: int, n_features: int):
    import torch.nn as nn

    class ConditionalGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.label_emb = nn.Embedding(2, label_dim)
            self.net = nn.Sequential(
                nn.Linear(noise_dim + label_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, n_features),
                nn.Sigmoid(),
            )

        def forward(self, z, labels):
            return self.net(torch.cat([z, self.label_emb(labels)], dim=1))

    return ConditionalGenerator()


# =========================================================
# GAN / WCGAN-GP synthetic Botnet 생성
# =========================================================
def generate_gan_botnet(
    gen_path: Path,
    n_generate: int,
    conditional: bool,
) -> np.ndarray:
    if n_generate <= 0:
        raise ValueError("생성할 synthetic Botnet 수가 0 이하입니다.")

    if gen_path is None or not gen_path.exists():
        raise FileNotFoundError(f"Generator 파일이 없습니다: {gen_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(gen_path, map_location=device)

    noise_dim = ckpt["noise_dim"]
    n_features = ckpt["n_features"]
    label_dim = ckpt.get("label_dim", 16)

    all_fake = []

    # 새 형식: 다중 세그먼트 Generator
    if "generators" in ckpt:
        state_dicts = ckpt["generators"]
        segment_sizes = ckpt["segment_sizes"]
        total_seg = sum(segment_sizes)

        generated_so_far = 0

        for i, (state_dict, seg_size) in enumerate(zip(state_dicts, segment_sizes)):
            if i == len(state_dicts) - 1:
                n_gen_i = n_generate - generated_so_far
            else:
                n_gen_i = int(n_generate * seg_size / total_seg)
                generated_so_far += n_gen_i

            if n_gen_i <= 0:
                continue

            if conditional:
                G = make_wcgan_generator(noise_dim, label_dim, n_features).to(device)
            else:
                G = make_gan_generator(noise_dim, n_features).to(device)

            G.load_state_dict(state_dict)
            G.eval()

            samples = []
            with torch.no_grad():
                for start in range(0, n_gen_i, 1024):
                    bs = min(1024, n_gen_i - start)
                    z = torch.randn(bs, noise_dim, device=device)

                    if conditional:
                        labels = torch.ones(bs, dtype=torch.long, device=device)
                        fake = G(z, labels)
                    else:
                        fake = G(z)

                    samples.append(fake.cpu().numpy())

            all_fake.append(np.vstack(samples))

    # 구 형식: 단일 Generator
    else:
        if conditional:
            G = make_wcgan_generator(noise_dim, label_dim, n_features).to(device)
        else:
            G = make_gan_generator(noise_dim, n_features).to(device)

        G.load_state_dict(ckpt["model_state_dict"])
        G.eval()

        samples = []
        with torch.no_grad():
            for start in range(0, n_generate, 1024):
                bs = min(1024, n_generate - start)
                z = torch.randn(bs, noise_dim, device=device)

                if conditional:
                    labels = torch.ones(bs, dtype=torch.long, device=device)
                    fake = G(z, labels)
                else:
                    fake = G(z)

                samples.append(fake.cpu().numpy())

        all_fake.append(np.vstack(samples))

    fake_bot = np.vstack(all_fake).astype(np.float32)

    print(f"\n[{AUGMENT.upper()}]")
    print(f"Generator : {gen_path}")
    print(f"생성 Bot  : {len(fake_bot):,}")

    return fake_bot


# =========================================================
# 1. 통계 비교
# =========================================================
def compare_statistics(real_bot: np.ndarray, fake_bot: np.ndarray):
    real_mean = real_bot.mean(axis=0)
    fake_mean = fake_bot.mean(axis=0)

    real_std = real_bot.std(axis=0)
    fake_std = fake_bot.std(axis=0)

    mean_abs_diff = np.abs(real_mean - fake_mean)
    std_abs_diff = np.abs(real_std - fake_std)

    print("\n" + "=" * 70)
    print("[1] Real Botnet vs Synthetic Botnet 통계 비교")
    print("=" * 70)
    print(f"Real Bot shape      : {real_bot.shape}")
    print(f"Synthetic Bot shape : {fake_bot.shape}")
    print(f"평균 차이 mean(abs)     : {mean_abs_diff.mean():.6f}")
    print(f"표준편차 차이 mean(abs) : {std_abs_diff.mean():.6f}")
    print(f"Real std 평균       : {real_std.mean():.6f}")
    print(f"Fake std 평균       : {fake_std.mean():.6f}")
    print(f"Fake std 최소       : {fake_std.min():.6f}")
    print(f"Fake std 최대       : {fake_std.max():.6f}")

    if fake_std.mean() < real_std.mean() * 0.3:
        print("진단: ⚠️ synthetic 분산이 실제보다 많이 작음 → mode collapse/coverage 부족 의심")
    else:
        print("진단: synthetic 분산이 극단적으로 작지는 않음")


# =========================================================
# 2. PCA 시각화
# =========================================================
def plot_pca(
    real_bot: np.ndarray,
    fake_bot: np.ndarray,
    save_path: Path,
    n_vis: int,
):
    n = min(len(real_bot), len(fake_bot), n_vis)

    rng = np.random.default_rng(42)
    real_idx = rng.choice(len(real_bot), n, replace=False)
    fake_idx = rng.choice(len(fake_bot), n, replace=False)

    X_vis = np.vstack([
        real_bot[real_idx],
        fake_bot[fake_idx],
    ])

    labels = np.array([0] * n + [1] * n)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_vis)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        X_2d[labels == 0, 0],
        X_2d[labels == 0, 1],
        s=6,
        alpha=0.5,
        label="real botnet",
    )
    plt.scatter(
        X_2d[labels == 1, 0],
        X_2d[labels == 1, 1],
        s=6,
        alpha=0.5,
        label=f"{AUGMENT} botnet",
    )
    plt.legend()
    plt.title(f"Real Botnet vs {AUGMENT.upper()} Botnet PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print("\n" + "=" * 70)
    print("[2] PCA 시각화 저장")
    print("=" * 70)
    print(f"저장 위치: {save_path}")


# =========================================================
# 3. TSTR 평가
# =========================================================
# =========================================================
# 3. TSTR 평가
# =========================================================
def run_tstr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fake_bot: np.ndarray,
):
    from sklearn.metrics import roc_curve

    real_normal = X_train[y_train == 0]

    n_bot = len(fake_bot)
    n_normal = min(len(real_normal), n_bot * 10)

    rng = np.random.default_rng(42)
    normal_idx = rng.choice(len(real_normal), n_normal, replace=False)

    X_tstr_train = np.vstack([
        real_normal[normal_idx],
        fake_bot,
    ]).astype(np.float32)

    y_tstr_train = np.concatenate([
        np.zeros(n_normal, dtype=int),
        np.ones(n_bot, dtype=int),
    ])

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )

    print("\n" + "=" * 70)
    print("[3] TSTR 평가")
    print("=" * 70)
    print(f"TSTR train Normal={n_normal:,}, Synthetic Bot={n_bot:,}")
    print(f"Real val shape={X_val.shape}, Bot ratio={y_val.mean():.6f}")

    clf.fit(X_tstr_train, y_tstr_train)

    # Bot 클래스 확률
    y_prob = clf.predict_proba(X_val)[:, 1]

    try:
        auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        auc = None

    # =====================================================
    # 1) 기존 방식: threshold 0.5
    # =====================================================
    y_pred_05 = (y_prob >= 0.5).astype(int)

    f1_05 = f1_score(y_val, y_pred_05, zero_division=0)
    recall_05 = recall_score(y_val, y_pred_05, zero_division=0)
    precision_05 = precision_score(y_val, y_pred_05, zero_division=0)

    print("\n" + "-" * 70)
    print("[3-1] Threshold 0.5 기준")
    print("-" * 70)
    print(classification_report(y_val, y_pred_05, digits=4, zero_division=0))
    print(f"F1       : {f1_05:.4f}")
    print(f"Recall   : {recall_05:.4f}")
    print(f"Precision: {precision_05:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if auc is not None else "ROC-AUC  : None")

    # =====================================================
    # 2) Youden's J threshold 기준
    # =====================================================
    if len(np.unique(y_val)) < 2:
        print("\n[Youden's J] y_val에 클래스가 하나뿐이라 threshold 계산 불가")
        return

    fpr, tpr, thresholds = roc_curve(y_val, y_prob)

    # threshold[0]이 inf일 수 있으므로 제외
    valid = np.isfinite(thresholds)
    fpr = fpr[valid]
    tpr = tpr[valid]
    thresholds = thresholds[valid]

    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_thr = thresholds[best_idx]

    y_pred_j = (y_prob >= best_thr).astype(int)

    f1_j = f1_score(y_val, y_pred_j, zero_division=0)
    recall_j = recall_score(y_val, y_pred_j, zero_division=0)
    precision_j = precision_score(y_val, y_pred_j, zero_division=0)

    print("\n" + "-" * 70)
    print("[3-2] Youden's J threshold 기준")
    print("-" * 70)
    print(f"Best threshold: {best_thr:.6f}")
    print(f"Youden's J    : {j_scores[best_idx]:.6f}")
    print(classification_report(y_val, y_pred_j, digits=4, zero_division=0))
    print(f"F1       : {f1_j:.4f}")
    print(f"Recall   : {recall_j:.4f}")
    print(f"Precision: {precision_j:.4f}")
    print(f"ROC-AUC  : {auc:.4f}" if auc is not None else "ROC-AUC  : None")

    # =====================================================
    # 3) 확률 분포 진단
    # =====================================================
    bot_prob = y_prob[y_val == 1]
    normal_prob = y_prob[y_val == 0]

    print("\n" + "-" * 70)
    print("[3-3] Probability 분포 진단")
    print("-" * 70)

    if len(bot_prob) > 0:
        print(f"Real Bot proba min : {bot_prob.min():.6f}")
        print(f"Real Bot proba mean: {bot_prob.mean():.6f}")
        print(f"Real Bot proba max : {bot_prob.max():.6f}")

    if len(normal_prob) > 0:
        print(f"Normal proba min   : {normal_prob.min():.6f}")
        print(f"Normal proba mean  : {normal_prob.mean():.6f}")
        print(f"Normal proba max   : {normal_prob.max():.6f}")

    # =====================================================
    # 4) 최종 진단
    # =====================================================
    print("\n[진단]")

    if recall_05 == 0 and auc is not None and auc >= 0.8:
        print("⚠️ 0.5 threshold에서는 Recall=0이지만 ROC-AUC는 높음")
        print("   → synthetic sample이 완전히 실패했다기보다 threshold mismatch 가능성이 큼")

    if recall_j > 0:
        print("✅ Youden's J threshold에서는 실제 Botnet 탐지가 가능함")
    else:
        print("⚠️ Youden's J threshold에서도 실제 Botnet 탐지 실패")

    if auc is not None and auc < 0.7:
        print("⚠️ ROC-AUC가 낮음 → synthetic sample의 실제 Botnet 대표성이 약함")
    elif auc is not None and auc >= 0.8:
        print("✅ ROC-AUC가 높음 → real Botnet과 Normal을 구분하는 ranking 능력은 있음")


# =========================================================
# main
# =========================================================
def main():
    print("=" * 70)
    print(f"증강 품질 진단 [{DATASET} / {AUGMENT}]")
    print("=" * 70)
    print(f"TARGET_RATIO: {TARGET_RATIO}")
    print(f"Fold        : {FOLD}")

    if GEN_PATH is not None:
        print(f"Generator   : {GEN_PATH}")

    X_all = np.load(FLAT_DIR / "X_trainval.npy").astype(np.float32)
    y_all = np.load(FLAT_DIR / "y_trainval.npy").astype(int)

    print(f"\n[LOAD] X_all={X_all.shape}, y ratio={y_all.mean():.6f}")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    selected = None
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all), 1):
        if fold == FOLD:
            selected = train_idx, val_idx
            break

    if selected is None:
        raise ValueError(f"fold는 1~5 사이여야 합니다. 입력값: {FOLD}")

    train_idx, val_idx = selected
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    real_bot = X_train[y_train == 1]

    n_majority = int((y_train == 0).sum())
    n_current = int(y_train.sum())
    n_target = int(n_majority * TARGET_RATIO / (1 - TARGET_RATIO))
    n_generate = max(0, n_target - n_current)

    print(f"\n[Fold {FOLD}]")
    print(f"Train Bot 원본 : {n_current:,}")
    print(f"Target Bot     : {n_target:,}")
    print(f"Generate Fake  : {n_generate:,}")

    if AUGMENT == "smote":
        fake_bot = generate_smote_botnet(
            X_train=X_train,
            y_train=y_train,
            target_ratio=TARGET_RATIO,
        )
    else:
        fake_bot = generate_gan_botnet(
            gen_path=GEN_PATH,
            n_generate=n_generate,
            conditional=CONDITIONAL,
        )

    compare_statistics(real_bot, fake_bot)

    save_dir = PROJECT / "artifacts" / "debug_augmentation_quality"
    save_dir.mkdir(parents=True, exist_ok=True)

    pca_path = save_dir / f"{DATASET}_{AUGMENT}_fold{FOLD}_pca.png"
    plot_pca(real_bot, fake_bot, pca_path, N_VIS)

    run_tstr(X_train, y_train, X_val, y_val, fake_bot)


if __name__ == "__main__":
    main()