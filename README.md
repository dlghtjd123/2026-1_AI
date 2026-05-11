# 2026-1_AI
수원대학교 4학년 2026-1학기 AI보안

딥러닝 기반 봇넷 탐지 시스템 — CIC-IDS2017 학습 / CIC-IDS2018 + CTU-13 교차검증  
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
  cic-ids2018/        ← CSE-CIC-IDS2018 CSV 파일들 (선택)
  ctu-13/
    scenario9_raw.csv ← CTU-13 시나리오 9 CICFlowMeter CSV
```

---

## 실행 순서

### Step 1. 전처리

```bash
python preprocess_cicids2017.py
python preprocess_cicids2018.py
python preprocess_ctu13.py
```

### Step 2. 학습 — Baseline (증강 없음)

```bash
python train_rf.py
python train_xgb.py
python train_cnn_lstm.py
python train_gru.py
python train_cnn_gru.py
```

모델 저장 위치: `artifacts/models/`

### Step 3. 증강 후 재학습

증강 방식을 `--augment` 옵션으로 지정합니다.

#### SMOTE

```bash
python augment_smote.py

python train_rf.py       --augment smote
python train_xgb.py      --augment smote
python train_cnn_lstm.py --augment smote
python train_gru.py      --augment smote
python train_cnn_gru.py  --augment smote
```

모델 저장 위치: `artifacts/models_smote/`

#### GAN

```bash
python augment_gan.py

python train_rf.py       --augment gan
python train_xgb.py      --augment gan
python train_cnn_lstm.py --augment gan
python train_gru.py      --augment gan
python train_cnn_gru.py  --augment gan
```

모델 저장 위치: `artifacts/models_gan/`

#### WCGAN-GP

```bash
python augment_wcgan_gp.py

python train_rf.py       --augment wcgan_gp
python train_xgb.py      --augment wcgan_gp
python train_cnn_lstm.py --augment wcgan_gp
python train_gru.py      --augment wcgan_gp
python train_cnn_gru.py  --augment wcgan_gp
```

모델 저장 위치: `artifacts/models_wcgan_gp/`

### Step 4. 평가

```bash
python evaluate.py                    # Baseline
python evaluate.py --augment smote    # SMOTE
python evaluate.py --augment gan      # GAN
python evaluate.py --augment wcgan_gp # WCGAN-GP
```

결과 저장 위치:

| 옵션 | 저장 경로 |
|---|---|
| (기본값) | `artifacts/results/eval_results.json` |
| `--augment smote` | `artifacts/results_smote/eval_results.json` |
| `--augment gan` | `artifacts/results_gan/eval_results.json` |
| `--augment wcgan_gp` | `artifacts/results_wcgan_gp/eval_results.json` |

### Step 5. 시각화

```bash
python visualize.py                    # Baseline
python visualize.py --augment smote    # SMOTE
python visualize.py --augment gan      # GAN
python visualize.py --augment wcgan_gp # WCGAN-GP
```

---

## 저장 구조

```
artifacts/
  models/           ← Baseline 모델
    rf/             ← RF 모델
    xgb/            ← XGBoost 모델
    cnn_lstm/       ← CNN-LSTM 모델
    gru/            ← GRU 모델
    cnn_gru/        ← CNN-GRU 모델
  models_smote/     ← SMOTE 증강 모델 (동일 구조)
  models_gan/       ← GAN 증강 모델
  models_wcgan_gp/  ← WCGAN-GP 증강 모델

  results/          ← Baseline 평가 결과
  results_smote/
  results_gan/
  results_wcgan_gp/

project/data/processed/
  cicids2017/          ← CIC-IDS2017 전처리 데이터 (train/val/test)
  cicids2017_smote/    ← SMOTE 증강 데이터
  cicids2017_gan/      ← GAN 증강 데이터
  cicids2017_wcgan_gp/ ← WCGAN-GP 증강 데이터
  cicids2018/          ← CIC-IDS2018 교차검증 데이터 (test only)
  ctu13/               ← CTU-13 교차검증 데이터 (test only)
```

---

## 평가 방식

### 주 지표: ROC-AUC (threshold-independent)

클래스 불균형 및 교차 데이터셋 환경에서 threshold 선택과 무관하게  
모델의 구별 능력을 공정하게 측정합니다.

> Accuracy can be misleading under class imbalance, so we emphasize  
> F1 (which balances precision/recall) and AUC (threshold-independent).  
> — Transformer-IDS, Journal of Computer Security 2025

| ROC-AUC | 의미 |
|---|---|
| 1.0 | 완벽 |
| 0.9+ | 우수 |
| 0.7+ | 양호 |
| 0.5 | 랜덤 수준 |
| < 0.5 | 랜덤보다 나쁨 |

### 보조 지표: F1 @ Youden's J threshold

Youden's J = TPR - FPR 최대화로 최적 threshold 선택.  
target dataset 정보 사용 (Safety 2025 방식) — 논문 명시 필요.

---

## 전처리 방식

### Scaler: MinMaxScaler + Secondary MinMaxScaler

```
CIC-IDS2017:
  MinMaxScaler.fit(train) → [0, 1] 범위
  저장: scaler_flow.pkl

CIC-IDS2018 / CTU-13 (교차검증):
  ① MinMaxScaler.transform()   ← CIC2017 기준 변환
  ② MinMaxScaler.fit_transform() ← target 분포 재정렬
  최종 범위: [0, 1] 유지
```

- **근거**: D'Hooge et al. (2020) — MinMaxScaler 없이 교차검증 F1=0%  
- **Secondary scaler**: Safety 2025 (de Nascimento & Hou, 2025) 방식  
- ※ target dataset 정보 사용 — Strict zero-shot 아님

### 교차검증 데이터셋

| 데이터셋 | 봇넷 유형 | 비고 |
|---|---|---|
| CIC-IDS2017 (학습) | Neris IRC (1 bot) | 학습 소스 |
| CIC-IDS2018 (교차검증) | Ares + Zeus HTTP | D'Hooge (2020) 동일 설정 |
| CTU-13 시나리오 9 (교차검증) | Neris IRC (10 bots) | 같은 패밀리, 다른 환경 |

---

## 모델 목록

| 모델 | 파일 | 입력 형태 |
|---|---|---|
| Random Forest | `train_rf.py` | `flat/` (n, 77) |
| XGBoost | `train_xgb.py` | `flat/` (n, 77) |
| CNN-LSTM | `train_cnn_lstm.py` | `seq/` (n, 77, 1) |
| GRU | `train_gru.py` | `seq/` (n, 77, 1) |
| CNN-GRU | `train_cnn_gru.py` | `seq/` (n, 77, 1) |

---

## 주요 참고 문헌

- D'Hooge et al. (2020). "Inter-dataset generalization strength of supervised machine learning methods for intrusion detection." *Journal of Information Security and Applications*, 54. DOI: 10.1016/j.jisa.2020.102564
- de Nascimento & Hou (2025). "Uncertainty-Aware Adaptive IDS Using Hybrid CNN-LSTM with cWGAN-GP." *MDPI Safety*, 11(4), 120. DOI: 10.3390/safety11040120
- Cantone et al. (2024). "Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study." *IEEE Access*, 12. DOI: 10.1109/ACCESS.2024.3472907
- Garcia et al. (2014). "An empirical comparison of botnet detection methods." *Computers & Security*, 45. DOI: 10.1016/j.cose.2014.05.011