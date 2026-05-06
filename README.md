# 2026-1_AI
수원대학교 4학년 2026-1학기 AI보안

딥러닝 기반 봇넷 탐지 시스템 — CIC-IDS2017 학습 / CTU-13 교차검증  
데이터 증강 기법 비교: SMOTE / GAN / WGAN-GP / WCGAN-GP

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
  ctu-13/
    scenario9_raw.csv ← CTU-13 시나리오 9 CICFlowMeter CSV
```

---

## 실행 순서

### Step 1. 전처리

```bash
python preprocess_cicids2017.py   # scaler 생성 (반드시 먼저)
python preprocess_ctu13.py        # CIC2017 scaler 적용
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

#### WGAN-GP

```bash
python augment_wgan_gp.py

python train_rf.py       --augment wgan_gp
python train_xgb.py      --augment wgan_gp
python train_cnn_lstm.py --augment wgan_gp
python train_gru.py      --augment wgan_gp
python train_cnn_gru.py  --augment wgan_gp
```

모델 저장 위치: `artifacts/models_wgan_gp/`

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
python evaluate.py --augment wgan_gp  # WGAN-GP
python evaluate.py --augment wcgan_gp # WCGAN-GP
```

결과 저장 위치:

| 옵션 | 저장 경로 |
|---|---|
| `--augment none` (기본값) | `artifacts/results/eval_results.json` |
| `--augment smote` | `artifacts/results_smote/eval_results.json` |
| `--augment gan` | `artifacts/results_gan/eval_results.json` |
| `--augment wgan_gp` | `artifacts/results_wgan_gp/eval_results.json` |
| `--augment wcgan_gp` | `artifacts/results_wcgan_gp/eval_results.json` |

### Step 5. 시각화

```bash
python visualize.py                    # Baseline
python visualize.py --augment smote    # SMOTE
python visualize.py --augment gan      # GAN
python visualize.py --augment wgan_gp  # WGAN-GP
python visualize.py --augment wcgan_gp # WCGAN-GP
```

---

## 저장 구조

```
artifacts/
  models/           ← Baseline 모델
  models_smote/     ← SMOTE 증강 모델
  models_gan/       ← GAN 증강 모델
  models_wgan_gp/   ← WGAN-GP 증강 모델
  models_wcgan_gp/  ← WCGAN-GP 증강 모델

  results/          ← Baseline 평가 결과
  results_smote/
  results_gan/
  results_wgan_gp/
  results_wcgan_gp/

project/data/processed/
  cicids2017/       ← 원본 전처리 데이터
  cicids2017_smote/ ← SMOTE 증강 데이터
  cicids2017_gan/   ← GAN 증강 데이터
  cicids2017_wgan_gp/
  cicids2017_wcgan_gp/
  ctu13/            ← CTU-13 교차검증 데이터
```

---

## 교차검증 방식

CIC-IDS2017로 학습한 모델을 CTU-13 시나리오 9로 교차검증합니다.

- **Scaler**: CIC2017 scaler → Secondary StandardScaler (분포 정렬)
- **Threshold**: target dataset(CTU-13) 기반 best-F1 탐색
- **방식**: Safety 2025 (de Nascimento & Hou, 2025) 방식 적용
- ※ target dataset 정보 사용 — Strict zero-shot 아님

| 데이터셋 | 봇넷 유형 |
|---|---|
| CIC-IDS2017 (학습) | Neris IRC 봇넷 (1 bot) |
| CTU-13 시나리오 9 (검증) | Neris IRC 봇넷 (10 bots) |

---

## 모델 목록

| 모델 | 파일 | 입력 |
|---|---|---|
| Random Forest | `train_rf.py` | `flat/` (n, 77) |
| XGBoost | `train_xgb.py` | `flat/` (n, 77) |
| CNN-LSTM | `train_cnn_lstm.py` | `seq/` (n, 77, 1) |
| GRU | `train_gru.py` | `seq/` (n, 77, 1) |
| CNN-GRU | `train_cnn_gru.py` | `seq/` (n, 77, 1) |