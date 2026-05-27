# 2026-1_AI
수원대학교 4학년 2026-1학기 AI보안

딥러닝 기반 봇넷 탐지 시스템 — 데이터셋별 5-Fold 내부 교차검증  
Benign Subsampling 기반 학습 시간 단축 + 데이터 증강 기법 비교: SMOTE / GAN / WCGAN-GP

---

## 설치

### 1. PyTorch (CUDA 12.4) 먼저 설치
```bash
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

### 2. 나머지 패키지 설치
```bash
pip install -r requirements.txt
```

---

## 데이터셋 준비

```
project/data/raw/
  cic-ids2017/        ← CIC-IDS2017 CSV 파일들
  cic-ids2018/        ← CSE-CIC-IDS2018 CSV 파일 (Friday-02-03-2018.csv)
  ctu-13/
    scenario9_raw.csv ← CTU-13 시나리오 9 CICFlowMeter CSV
```

---

## 실행 순서

### Step 1. 전처리

> **반드시 CIC-IDS2017을 먼저 실행해야 합니다.**  
> CIC-IDS2017 전처리에서 chi-square 피처 선택(32개)을 수행하고 `selected_features.json`을 생성합니다.  
> CIC-IDS2018, CTU-13은 이 파일을 로드하여 동일한 32개 피처를 적용합니다.

```bash
python preprocess_cicids2017.py   # chi-square → 상위 32개 피처 선택 + selected_features.json 생성
python preprocess_cicids2018.py   # selected_features.json 로드하여 동일 32개 적용
python preprocess_ctu13.py        # selected_features.json 로드하여 동일 32개 적용
```

피처 선택 방식: **Chi-square (Zhao et al., 2024 동일)**  
선택 기준: CIC-IDS2017 trainval 기준으로 77개 → 상위 32개  
분할 방식: **Flow 단위 랜덤 Stratified split (Zhao et al., 2024 방식)**  
공유 방식: 세 데이터셋 모두 동일한 32개 피처 사용 (CICFlowMeter 피처 동일)

---

### Step 2. 학습 전 Benign Subsampling 정책

본 프로젝트에서는 학습 시간이 과도하게 증가하는 문제를 줄이기 위해 **train fold에만 Benign Subsampling**을 적용합니다.  
Botnet 샘플은 줄이지 않고 전부 유지하며, Benign 샘플만 제한합니다.

검증 데이터(`val fold`)는 항상 원본 분포를 유지합니다.  
따라서 학습 데이터 크기는 줄이되, 검증 성능은 실제 데이터 분포 기준으로 평가됩니다.

#### 핵심 인자

| 인자 | 의미 | 기본값 |
|------|------|------|
| `--max_normal` | fold당 사용할 최대 Benign 샘플 수 | `200000` |
| `--max_mismatch` | train Botnet 비율이 원본/val Botnet 비율보다 커질 수 있는 최대 배수 | `10` |

#### 동작 방식

```text
1. K-Fold로 train / val 분리
2. train fold에서 Botnet은 전부 유지
3. train fold에서 Benign만 subsampling
4. 단, train Botnet 비율이 val/original 비율보다 max_mismatch배 이상 커지지 않도록 제한
5. val fold는 원본 분포 그대로 유지
```

#### 예시: CTU-13

CTU-13은 원본 Botnet 비율이 약 2.35%로 CIC-IDS2017보다 높습니다.  
따라서 Benign을 너무 많이 줄이면 train Botnet 비율이 과도하게 높아져 Precision이 낮아질 수 있습니다.

예를 들어 `--max_mismatch 10`이면:

```text
원본 Botnet 비율 ≈ 2.35%
허용 train Botnet 비율 ≈ 23.5%
```

이 경우 모델이 Botnet을 과도하게 예측하여 Recall은 높지만 Precision/F1이 낮아질 수 있습니다.

따라서 CTU-13은 다음처럼 더 낮은 mismatch 값을 권장합니다.

```bash
python train_cnn_lstm.py --dataset ctu13 --max_normal 500000 --max_mismatch 2
```

권장 기준:

| 데이터셋 | 권장 설정 |
|---------|---------|
| CIC-IDS2017 | `--max_normal 200000 --max_mismatch 10` |
| CSE-CIC-IDS2018 | `--max_normal 500000 --max_mismatch 2` |
| CTU-13 | `--max_normal 500000 --max_mismatch 2` |

---

### Step 3. 학습 — Baseline (증강 없음)

데이터셋별로 독립 학습 + 5-Fold 내부 교차검증을 수행합니다.

```bash
# CIC-IDS2017
python train_rf.py       --dataset cicids2017 --max_normal 500000 --max_mismatch 7
python train_xgb.py      --dataset cicids2017 --max_normal 500000 --max_mismatch 7
python train_cnn_lstm.py --dataset cicids2017 --max_normal 500000 --max_mismatch 7
python train_gru.py      --dataset cicids2017 --max_normal 500000 --max_mismatch 7
python train_cnn_gru.py  --dataset cicids2017 --max_normal 500000 --max_mismatch 7

# CSE-CIC-IDS2018
python train_rf.py       --dataset cicids2018 --max_normal 500000 --max_mismatch 2
python train_xgb.py      --dataset cicids2018 --max_normal 500000 --max_mismatch 2
python train_cnn_lstm.py --dataset cicids2018 --max_normal 500000 --max_mismatch 2
python train_gru.py      --dataset cicids2018 --max_normal 500000 --max_mismatch 2
python train_cnn_gru.py  --dataset cicids2018 --max_normal 500000 --max_mismatch 2

# CTU-13
python train_rf.py       --dataset ctu13 --max_normal 500000 --max_mismatch 2
python train_xgb.py      --dataset ctu13 --max_normal 500000 --max_mismatch 2
python train_cnn_lstm.py --dataset ctu13 --max_normal 500000 --max_mismatch 2
python train_gru.py      --dataset ctu13 --max_normal 500000 --max_mismatch 2
python train_cnn_gru.py  --dataset ctu13 --max_normal 500000 --max_mismatch 2
```

---

### Step 4. 평가 — Baseline

```bash
python evaluate.py --dataset cicids2017
python evaluate.py --dataset cicids2018
python evaluate.py --dataset ctu13
```

---

### Step 5. 증강 후 재학습 및 평가

> **증강 방식에 따라 사전 준비 여부가 다릅니다.**
>
> - **SMOTE**: 별도 준비 불필요. K-fold 내부에서 train fold에만 자동 적용.
> - **GAN / WCGAN-GP**: Generator를 먼저 학습해야 합니다.

#### 증강 목표량

SMOTE / GAN / WCGAN-GP 모두 **현재 fold train Botnet 수의 2배**를 목표로 합성 샘플을 생성합니다.  
고정 비율(예: 10%)이 아니라 현재 Botnet 수 기준 상대 배수이므로, 데이터셋별 Botnet 비율 차이에 관계없이 일관된 증강량이 적용됩니다.

| 데이터셋 | subsample 후 Botnet (예시) | 증강 후 Botnet (2배) | 증가량 |
|---------|--------------------------|-------------------|-------|
| CIC-IDS2017 | ~1,259개 | ~2,518개 | +1,259 |
| CTU-13 | ~42,000개 | ~84,000개 | +42,000 |
| CIC-IDS2018 | ~12,000개 | ~24,000개 | +12,000 |

> 증강 사용 시 모델 내부 클래스 보정이 자동으로 비활성화됩니다 (이중 보정 방지).
>
> | 모델 | Baseline | 증강 시 |
> |------|---------|--------|
> | RF | `class_weight="balanced_subsample"` | `class_weight=None` |
> | XGBoost | `scale_pos_weight=√(neg/pos)` | `scale_pos_weight=1.0` |
> | CNN-LSTM / GRU / CNN-GRU | `FocalLoss(α=0.75)` | `CrossEntropyLoss` |

#### SMOTE — 사전 준비 불필요

```bash
# Benign subsampling 인자와 함께 동일하게 사용
# CIC-IDS2017
python train_rf.py       --dataset cicids2017 --augment smote --max_normal 500000 --max_mismatch 7
python train_xgb.py      --dataset cicids2017 --augment smote --max_normal 500000 --max_mismatch 7
python train_cnn_lstm.py --dataset cicids2017 --augment smote --max_normal 500000 --max_mismatch 7
python train_gru.py      --dataset cicids2017 --augment smote --max_normal 500000 --max_mismatch 7
python train_cnn_gru.py  --dataset cicids2017 --augment smote --max_normal 500000 --max_mismatch 7

# CIC-IDS2018
python train_rf.py       --dataset cicids2018 --augment smote --max_normal 500000 --max_mismatch 2
python train_xgb.py      --dataset cicids2018 --augment smote --max_normal 500000 --max_mismatch 2
python train_cnn_lstm.py --dataset cicids2018 --augment smote --max_normal 500000 --max_mismatch 2
python train_gru.py      --dataset cicids2018 --augment smote --max_normal 500000 --max_mismatch 2
python train_cnn_gru.py  --dataset cicids2018 --augment smote --max_normal 500000 --max_mismatch 2

# CTU-13
python train_rf.py       --dataset ctu13 --augment smote --max_normal 500000 --max_mismatch 2
python train_xgb.py      --dataset ctu13 --augment smote --max_normal 500000 --max_mismatch 2
python train_cnn_lstm.py --dataset ctu13 --augment smote --max_normal 500000 --max_mismatch 2
python train_gru.py      --dataset ctu13 --augment smote --max_normal 500000 --max_mismatch 2
python train_cnn_gru.py  --dataset ctu13 --augment smote --max_normal 500000 --max_mismatch 2


python evaluate.py --dataset cicids2017 --augment smote
python evaluate.py --dataset cicids2018 --augment smote
python evaluate.py --dataset ctu13      --augment smote
```

#### GAN — Generator 사전 학습 필요

```bash
python augment_gan.py --dataset [cicids2017/cicids2018/ctu13]

python train_rf.py       --dataset cicids2017 --augment gan --max_normal 500000 --max_mismatch 7
python train_rf.py       --dataset ctu13      --augment gan --max_normal 500000 --max_mismatch 2
# ... (동일 패턴)

python evaluate.py --dataset cicids2017 --augment gan
python evaluate.py --dataset cicids2018 --augment gan
python evaluate.py --dataset ctu13      --augment gan
```

#### WCGAN-GP — Generator 사전 학습 필요

```bash
python augment_wcgan_gp.py --dataset [cicids2017/cicids2018/ctu13]

python train_rf.py       --dataset cicids2017 --augment wcgan_gp --max_normal 500000 --max_mismatch 7
python train_rf.py       --dataset ctu13      --augment wcgan_gp --max_normal 500000 --max_mismatch 2
# ... (동일 패턴)

python evaluate.py --dataset cicids2017 --augment wcgan_gp
python evaluate.py --dataset cicids2018 --augment wcgan_gp
python evaluate.py --dataset ctu13      --augment wcgan_gp
```

---

### Step 6. 시각화

```bash
python visualize.py
python visualize.py --augment smote
python visualize.py --augment gan
python visualize.py --augment wcgan_gp
```

---

## 저장 구조

```
project/src/
  augment_utils.py          ← K-fold 내부 fold-level 증강 공통 모듈
  debug_utils.py            ← 원인 분석용 디버그 유틸리티

artifacts/
  models_cicids2017/        ← CIC-IDS2017 Baseline 모델
    rf/
    xgb/
    cnn_lstm/
    gru/
    cnn_gru/
  models_cicids2017_smote/  ← CIC-IDS2017 SMOTE 모델 (동일 구조)
  models_cicids2017_gan/
  models_cicids2017_wcgan_gp/
  models_cicids2018/
  models_ctu13/

  results_cicids2017/
  results_cicids2017_smote/
  results_cicids2017_gan/
  results_cicids2017_wcgan_gp/
  results_cicids2018/
  results_ctu13/

  figures/                  ← Baseline 시각화
  figures_smote/
  figures_gan/
  figures_wcgan_gp/

project/data/processed/
  cicids2017/
    flat/  X_trainval.npy  y_trainval.npy  X_test.npy  y_test.npy  ← (n, 32)
    seq/   X_trainval.npy  y_trainval.npy  X_test.npy  y_test.npy  ← (n, 32, 1)
           scaler_flow.pkl
    selected_features.json  ← chi2 선택 피처 32개 (CIC2018, CTU13에서 공유)
  cicids2017_gan/
    generator.pt            ← GAN Generator
  cicids2017_wcgan_gp/
    generator_wcgan_gp.pt   ← WCGAN-GP Generator
  cicids2018/
    flat/  X_trainval.npy  y_trainval.npy  X_test.npy  y_test.npy  ← (n, 32)
    seq/   ...
  ctu13/
    flat/  X_trainval.npy  y_trainval.npy  X_test.npy  y_test.npy  ← (n, 32)
    seq/   ...
```

---

## 증강 방식 비교

| 방식 | 사전 준비 | 증강 적용 위치 | 증강 목표량 | 특징 |
|------|---------|--------------|-----------|------|
| SMOTE | 없음 | K-fold train fold 내부 | 현재 Botnet × 2배 | 보간 기반 합성 |
| GAN | Generator 학습 필요 | K-fold train fold 내부 | 현재 Botnet × 2배 | 신경망 기반 생성 |
| WCGAN-GP | Generator 학습 필요 | K-fold train fold 내부 | 현재 Botnet × 2배 | 조건부 생성, 학습 안정성 ↑ |

**공통**: Benign Subsampling과 증강은 train fold에만 적용하며, val fold는 항상 원본 데이터 유지 → K-fold 결과가 실제 성능 반영

---

## 평가 방식

### 주 지표: F1-score, Recall

| 지표 | 의미 |
|------|------|
| **F1-score** | Precision과 Recall의 조화평균, 불균형 데이터 핵심 지표 |
| **Recall** | 봇넷 미탐지(FN) 최소화, 보안 도메인 핵심 지표 |

딥러닝 모델: softmax + argmax 기반 분류 (threshold 없음)  
머신러닝 모델: predict_proba >= 0.5 (argmax 동치)

### 보조 지표: ROC-AUC

| ROC-AUC | 의미 |
|---------|------|
| 1.0 | 완벽 |
| 0.9+ | 우수 |
| 0.7+ | 양호 |
| 0.5 | 랜덤 수준 |

### K-Fold 교차검증

- **방식**: StratifiedKFold(n_splits=5), fold 분할 시 봇넷 비율 유지
- **Benign Subsampling**: train fold에만 적용, Botnet은 전부 유지
- **증강 적용**: train fold에만 적용 (subsample 후), val fold는 항상 원본 유지
- **결과 보고**: fold별 평균 ± 표준편차
- **최종 평가**: holdout test set(20%)으로 수행 (`evaluate.py`)

---

## 전처리 방식

### 피처 선택

| 항목 | 내용 |
|------|------|
| 원본 피처 수 | 77개 (CICFlowMeter 기반, log 변환 적용) |
| **선택 방법** | **Chi-square (χ²)** |
| **선택 피처 수** | **32개** |
| 선택 기준 | CIC-IDS2017 trainval 기준 (test 정보 미사용) |
| 공유 방식 | `selected_features.json` → CIC-IDS2018, CTU-13에 동일 적용 |
| 근거 | Zhao et al. (2024, chi-square 32개) |

### 분할 방식

| 데이터셋 | trainval | test | 분할 기준 |
|---------|---------|------|---------|
| CIC-IDS2017 | 80% | 20% | **랜덤 Stratified split** |
| CIC-IDS2018 | 80% | 20% | **랜덤 Stratified split** |
| CTU-13 | 80% | 20% | **랜덤 Stratified split** |

근거: Zhao et al. (2024) — 8:2 랜덤 분할

### Scaler

```
CIC-IDS2017:
  MinMaxScaler.fit(trainval) → [0, 1]
  → chi-square 피처 선택 적용 (32개)
  저장: scaler_flow.pkl, selected_features.json

CIC-IDS2018 / CTU-13:
  ① CIC2017 MinMaxScaler.transform()
  ② Secondary MinMaxScaler.fit_transform(trainval) → [0, 1] 유지
  ③ selected_features.json 인덱스로 32개 피처 추출
```

근거: D'Hooge et al. (2020)

### 손실 함수

| 모델 | 손실 함수 |
|------|---------|
| RF | class_weight="balanced_subsample" |
| XGBoost | scale_pos_weight=√(neg/pos) |
| CNN-LSTM / GRU / CNN-GRU | Multiclass Focal Loss (α=0.75, γ=2.0) + softmax/argmax |

Focal Loss 적용 근거: Lin et al. (2017)

---

## 인자 정리

| 스크립트 | 인자 | 선택값 |
|---------|------|------|
| `train_*.py` | `--dataset` | `cicids2017` / `cicids2018` / `ctu13` |
| `train_*.py` | `--augment` | `none` / `smote` / `gan` / `wcgan_gp` |
| `train_*.py` | `--n_folds` | 정수 (기본값: 5) |
| `train_*.py` | `--debug` | 원인 분석 로그 출력 |
| `train_*.py` | `--max_normal` | fold당 최대 Benign 샘플 수, `0`이면 제한 없음 |
| `train_*.py` | `--max_mismatch` | train/val Botnet 비율 최대 배수 제한 |
| `augment_gan.py` | `--dataset` | `cicids2017` / `cicids2018` / `ctu13` |
| `augment_wcgan_gp.py` | `--dataset` | `cicids2017` / `cicids2018` / `ctu13` |
| `evaluate.py` | `--dataset` | `cicids2017` / `cicids2018` / `ctu13` |
| `evaluate.py` | `--augment` | `none` / `smote` / `gan` / `wcgan_gp` |

---

## 모델 목록

| 모델 | 파일 | 입력 형태 | 출력 방식 |
|------|------|---------|---------|
| Random Forest | `train_rf.py` | `flat/` (n, 32) | predict_proba (argmax 동치) |
| XGBoost | `train_xgb.py` | `flat/` (n, 32) | predict_proba (argmax 동치) |
| CNN-LSTM | `train_cnn_lstm.py` | `seq/` (n, 32, 1) | softmax + argmax |
| GRU | `train_gru.py` | `seq/` (n, 32, 1) | softmax + argmax |
| CNN-GRU | `train_cnn_gru.py` | `seq/` (n, 32, 1) | softmax + argmax |

---

## 데이터셋 정보

| 데이터셋 | 봇넷 유형 | 봇넷 비율 |
|---------|---------|---------|
| CIC-IDS2017 | Neris IRC (1 bot) | 약 0.05% |
| CIC-IDS2018 | Ares + Zeus HTTP | 약 2.2% |
| CTU-13 Scenario 9 | Neris IRC (10 bots) | 약 1.4% |

> 실제 전처리 결과의 Botnet 비율은 사용한 CSV, Background 처리 여부, split 대상에 따라 달라질 수 있습니다.  
> 예: 현재 CTU-13 trainval 로그 기준 Botnet 비율은 약 2.35%로 확인됨.

---

## 주요 참고 문헌

- D'Hooge et al. (2020). "Inter-dataset generalization strength of supervised machine learning methods for intrusion detection." *Journal of Information Security and Applications*, 54. DOI: 10.1016/j.jisa.2020.102564
- Zhao et al. (2024). "Enhancing Network Intrusion Detection Performance using Generative Adversarial Networks." *arXiv:2404.07464*. (chi-square 32개 피처, 8:2 랜덤 split)
- Sayegh et al. (2024). "Enhanced Intrusion Detection with LSTM-Based Model, Feature Selection, and SMOTE for Imbalanced Data." *Applied Sciences*, 14(2), 479. DOI: 10.3390/app14020479
- de Nascimento & Hou (2025). "Uncertainty-Aware Adaptive IDS Using Hybrid CNN-LSTM with cWGAN-GP." *MDPI Safety*, 11(4), 120. DOI: 10.3390/safety11040120
- Lin et al. (2017). "Focal Loss for Dense Object Detection." *ICCV 2017*.
- Garcia et al. (2014). "An empirical comparison of botnet detection methods." *Computers & Security*, 45. DOI: 10.1016/j.cose.2014.05.011