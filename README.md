# 2026-1_AI
수원대학교 4학년 2026-1학기 AI보안

딥러닝 기반 봇넷 탐지 시스템 — 데이터셋별 5-Fold 내부 교차검증  
데이터 증강 기법 비교: SMOTE / GAN / WCGAN-GP

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

### Step 2. 학습 — Baseline (증강 없음)

데이터셋별로 독립 학습 + 5-Fold 내부 교차검증을 수행합니다.

```bash
# CIC-IDS2017
python train_rf.py       --dataset cicids2017
python train_xgb.py      --dataset cicids2017
python train_cnn_lstm.py --dataset cicids2017
python train_gru.py      --dataset cicids2017
python train_cnn_gru.py  --dataset cicids2017

# CSE-CIC-IDS2018
python train_rf.py       --dataset cicids2018
# ... (동일 패턴)

# CTU-13
python train_rf.py       --dataset ctu13
# ... (동일 패턴)
```

---

### Step 3. 평가 — Baseline

```bash
python evaluate.py --dataset cicids2017
python evaluate.py --dataset cicids2018
python evaluate.py --dataset ctu13
```

---

### Step 4. 증강 후 재학습 및 평가

> **증강 방식에 따라 사전 준비 여부가 다릅니다.**
>
> - **SMOTE**: 별도 준비 불필요. K-fold 내부에서 train fold에만 자동 적용.
> - **GAN / WCGAN-GP**: Generator를 먼저 학습해야 합니다.

#### SMOTE — 사전 준비 불필요

```bash
python train_rf.py       --dataset [cicids2017/cicids2018/ctu13] --augment smote
python train_xgb.py      --dataset [cicids2017/cicids2018/ctu13] --augment smote
python train_cnn_lstm.py --dataset [cicids2017/cicids2018/ctu13] --augment smote
python train_gru.py      --dataset [cicids2017/cicids2018/ctu13] --augment smote
python train_cnn_gru.py  --dataset [cicids2017/cicids2018/ctu13] --augment smote

python evaluate.py --dataset [cicids2017/cicids2018/ctu13] --augment smote
```

#### GAN — Generator 사전 학습 필요

```bash
python augment_gan.py --dataset [cicids2017/cicids2018/ctu13]

python train_rf.py       --dataset [cicids2017/cicids2018/ctu13] --augment gan
# ... (동일 패턴)

python evaluate.py --dataset [cicids2017/cicids2018/ctu13] --augment gan
```

#### WCGAN-GP — Generator 사전 학습 필요

```bash
python augment_wcgan_gp.py --dataset [cicids2017/cicids2018/ctu13]

python train_rf.py       --dataset [cicids2017/cicids2018/ctu13] --augment wcgan_gp
# ... (동일 패턴)

python evaluate.py --dataset [cicids2017/cicids2018/ctu13] --augment wcgan_gp
```

---

### Step 5. 시각화

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

| 방식 | 사전 준비 | 증강 적용 위치 | 특징 |
|------|---------|--------------|------|
| SMOTE | 없음 | K-fold train fold 내부 | 보간 기반 합성 |
| GAN | Generator 학습 필요 | K-fold train fold 내부 | 신경망 기반 생성 |
| WCGAN-GP | Generator 학습 필요 | K-fold train fold 내부 | 조건부 생성, 학습 안정성 ↑ |

**공통**: val fold는 항상 원본 데이터 유지 → K-fold 결과가 실제 성능 반영

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

- **방식**: StratifiedKFold(n_splits=5), 봇넷 비율 유지
- **증강 적용**: train fold에만 적용, val fold는 항상 원본 유지
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

---

## 주요 참고 문헌

- D'Hooge et al. (2020). "Inter-dataset generalization strength of supervised machine learning methods for intrusion detection." *Journal of Information Security and Applications*, 54. DOI: 10.1016/j.jisa.2020.102564
- Zhao et al. (2024). "Enhancing Network Intrusion Detection Performance using Generative Adversarial Networks." *arXiv:2404.07464*. (chi-square 32개 피처, 8:2 랜덤 split)
- Sayegh et al. (2024). "Enhanced Intrusion Detection with LSTM-Based Model, Feature Selection, and SMOTE for Imbalanced Data." *Applied Sciences*, 14(2), 479. DOI: 10.3390/app14020479
- de Nascimento & Hou (2025). "Uncertainty-Aware Adaptive IDS Using Hybrid CNN-LSTM with cWGAN-GP." *MDPI Safety*, 11(4), 120. DOI: 10.3390/safety11040120
- Lin et al. (2017). "Focal Loss for Dense Object Detection." *ICCV 2017*.
- Garcia et al. (2014). "An empirical comparison of botnet detection methods." *Computers & Security*, 45. DOI: 10.1016/j.cose.2014.05.011