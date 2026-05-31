# 2026-1_AI

수원대학교 4학년 2026-1학기 AI보안

딥러닝 기반 봇넷 탐지 시스템  
내부 데이터셋 검증 기반 데이터 증강 기법 성능 비교: **None / SMOTE / GAN / WCGAN-GP**

---

## 연구 흐름

본 프로젝트는 네트워크 플로우 기반 봇넷 탐지에서 데이터 증강 기법의 효과를 비교한다.

실험은 다음 원칙을 따른다.

```text
1. 전체 데이터에서 trainval / holdout test 분리
2. trainval 내부에서 Stratified K-Fold 검증
3. 각 fold의 train split에만 Benign subsampling 및 증강 적용
4. validation fold는 항상 원본 분포 유지
5. K-Fold로 설정을 선택한 뒤 trainval 전체로 최종 모델 재학습
6. holdout test set으로 최종 평가
```

중요: `test` 데이터는 전처리 이후 어떤 학습, 증강, Generator 학습에도 사용하지 않는다.

---

## 설치

### 1. PyTorch CUDA 12.4

```bash
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

### 2. 나머지 패키지

```bash
pip install -r requirements.txt
```

---

## 데이터셋 준비

```text
project/data/raw/
  cic-ids2017/        CIC-IDS2017 CSV 파일들
  cic-ids2018/        CSE-CIC-IDS2018 CSV 파일
  ctu-13/
    scenario9_raw.csv CTU-13 Scenario 9 CICFlowMeter CSV
```

---

## 전처리

CIC-IDS2017을 먼저 전처리해야 한다.  
CIC-IDS2017에서 chi-square 피처 선택 결과인 `selected_features.json`을 생성하고, CIC-IDS2018과 CTU-13은 동일한 32개 피처를 사용한다.

```bash
cd project/src

python preprocess_cicids2017.py
python preprocess_cicids2018.py
python preprocess_ctu13.py
```

전처리 요약:

| 항목 | 내용 |
|------|------|
| 분할 | trainval 80% / test 20% |
| split 방식 | 랜덤 Stratified split |
| 피처 선택 | Chi-square |
| 선택 피처 수 | 32개 |
| 선택 기준 | CIC-IDS2017 trainval |
| 스케일링 | MinMaxScaler 기반 [0, 1] |

---

## 실험 자동 실행

전체 실험은 [run_experiments.py](project/src/run_experiments.py)로 자동 실행할 수 있다.

```bash
python project/src/run_experiments.py --datasets cicids2017 --augments none smote gan wcgan_gp
python project/src/run_experiments.py --datasets cicids2018 --augments none smote gan wcgan_gp
python project/src/run_experiments.py --datasets ctu13 --augments none smote gan wcgan_gp
```

validation F1 기준으로 threshold를 최적화해서 실행:

```bash
python project/src/run_experiments.py \
  --datasets cicids2017 \
  --augments none smote gan wcgan_gp \
  --threshold_mode f1_opt
```

증강 배수를 바꿔서 실행:

```bash
python project/src/run_experiments.py \
  --datasets cicids2017 \
  --augments smote gan wcgan_gp \
  --augment_multiplier 5
```

세 데이터셋 전체 실행:

```bash
python project/src/run_experiments.py \
  --datasets cicids2017 cicids2018 ctu13 \
  --augments none smote gan wcgan_gp
```

명령만 확인하고 실제 실행하지 않기:

```bash
python project/src/run_experiments.py --datasets cicids2017 --augments none --dry_run
```

특정 모델만 실행:

```bash
python project/src/run_experiments.py \
  --datasets cicids2017 \
  --augments smote \
  --models rf xgb
```

단, `evaluate.py`는 5개 모델 파일을 모두 필요로 하므로 일부 모델만 실행하면 평가는 자동으로 건너뛴다.

실행 로그는 다음 위치에 저장된다.

```text
artifacts/experiment_runs/run_YYYYMMDD_HHMMSS.json
```

---

## 수동 실행

자동화 스크립트 대신 개별 실행도 가능하다.

### Baseline

```bash
cd project/src

python train_rf.py       --dataset cicids2017 --augment none
python train_xgb.py      --dataset cicids2017 --augment none
python train_cnn_lstm.py --dataset cicids2017 --augment none
python train_gru.py      --dataset cicids2017 --augment none
python train_cnn_gru.py  --dataset cicids2017 --augment none

python evaluate.py --dataset cicids2017 --augment none
```

threshold 최적화 옵션을 개별 학습에 적용:

```bash
python train_cnn_lstm.py --dataset cicids2017 --augment smote --threshold_mode f1_opt
python evaluate.py --dataset cicids2017 --augment smote
```

증강 배수 옵션을 개별 학습에 적용:

```bash
python train_cnn_lstm.py --dataset cicids2017 --augment smote --augment_multiplier 5
python evaluate.py --dataset cicids2017 --augment smote --augment_multiplier 5
```

### SMOTE

```bash
python train_rf.py       --dataset cicids2017 --augment smote
python train_xgb.py      --dataset cicids2017 --augment smote
python train_cnn_lstm.py --dataset cicids2017 --augment smote
python train_gru.py      --dataset cicids2017 --augment smote
python train_cnn_gru.py  --dataset cicids2017 --augment smote

python evaluate.py --dataset cicids2017 --augment smote
```

### GAN / WCGAN-GP

현재 논문용 실험 설계에서는 GAN/WCGAN-GP Generator를 미리 학습하지 않는다.  
각 K-Fold의 train split 안에서만 Generator를 학습하고 synthetic Bot 샘플을 생성한다.

```bash
python train_rf.py       --dataset cicids2017 --augment gan
python train_xgb.py      --dataset cicids2017 --augment gan
python train_cnn_lstm.py --dataset cicids2017 --augment gan
python train_gru.py      --dataset cicids2017 --augment gan
python train_cnn_gru.py  --dataset cicids2017 --augment gan

python evaluate.py --dataset cicids2017 --augment gan
```

```bash
python train_rf.py       --dataset cicids2017 --augment wcgan_gp
python train_xgb.py      --dataset cicids2017 --augment wcgan_gp
python train_cnn_lstm.py --dataset cicids2017 --augment wcgan_gp
python train_gru.py      --dataset cicids2017 --augment wcgan_gp
python train_cnn_gru.py  --dataset cicids2017 --augment wcgan_gp

python evaluate.py --dataset cicids2017 --augment wcgan_gp
```

---

## 증강 방식

| 방식 | Generator 사전학습 | 적용 위치 | validation 사용 여부 | 목표량 |
|------|------------------|----------|--------------------|--------|
| None | 없음 | 없음 | 원본 유지 | 없음 |
| SMOTE | 없음 | fold train split 내부 | 미사용 | Bot 수 `--augment_multiplier`배 |
| GAN | fold마다 train split으로 학습 | fold train split 내부 | 미사용 | Bot 수 `--augment_multiplier`배 |
| WCGAN-GP | fold마다 train split으로 학습 | fold train split 내부 | 미사용 | Bot 수 `--augment_multiplier`배 |

GAN/WCGAN-GP는 논문 실험의 엄밀성을 위해 **fold-local Generator**를 사용한다.

```text
Fold k:
  train split Bot only -> Generator 학습
  Generator -> synthetic Bot 생성
  train split + synthetic Bot -> classifier 학습
  val split -> 원본 그대로 평가
```

최종 holdout test 평가 전에는 `trainval` 전체만으로 final Generator를 다시 학습하고, final classifier를 재학습한다.

기본 증강 배수는 `2`이다. `--augment_multiplier 5`처럼 변경하면 기존 2배 결과를 덮어쓰지 않도록 별도 폴더에 저장된다.

```text
artifacts/results_cicids2017_smote/       # 기본 2배
artifacts/results_cicids2017_smote_mul5/  # 5배
```

fold-local synthetic sample cache:

```text
project/data/processed/{dataset}_{augment}_fold_cache/
```

---

## GAN/WCGAN-GP Epoch 설정

CNN-LSTM / GRU / CNN-GRU 분류 모델은 최대 30 epoch로 학습한다.

GAN/WCGAN-GP Generator epoch는 별도 설정이다.

| 항목 | 기본값 |
|------|--------|
| `FOLD_GAN_EPOCHS` | 500 |
| `FOLD_WCGAN_EPOCHS` | 500 |

자동화 스크립트에서 변경:

```bash
python project/src/run_experiments.py \
  --datasets cicids2017 \
  --augments gan wcgan_gp \
  --fold_gan_epochs 100 \
  --fold_wcgan_epochs 200
```

PowerShell 환경변수로 직접 지정:

```powershell
$env:FOLD_GAN_EPOCHS="100"
$env:FOLD_WCGAN_EPOCHS="200"
```

논문 결과에는 사용한 Generator epoch를 반드시 명시한다.

---

## Benign Subsampling

학습 시간과 클래스 불균형을 조절하기 위해 train split에만 Benign subsampling을 적용한다.  
Botnet 샘플은 제거하지 않는다.

| 인자 | 의미 | 기본값 |
|------|------|--------|
| `--max_normal` | fold당 최대 Benign 샘플 수 | `500000` |
| `--max_mismatch` | train Botnet 비율이 원본보다 커질 수 있는 최대 배수 | `cicids2017=5.0`, `cicids2018=2.0`, `ctu13=2.0` |

동작:

```text
1. K-Fold로 train / val 분리
2. train split에서 Botnet은 모두 유지
3. train split에서 Benign만 subsampling
4. train Botnet 비율이 원본 비율의 max_mismatch배를 넘지 않도록 제한
5. val split은 원본 분포 그대로 유지
```

---

## 클래스 불균형 처리

증강을 사용하지 않을 때는 모델 내부의 불균형 보정을 사용한다.  
증강을 사용할 때는 이중 보정을 피하기 위해 내부 보정을 비활성화한다.

| 모델 | Baseline | 증강 사용 시 |
|------|----------|-------------|
| RF | `class_weight="balanced_subsample"` | `class_weight=None` |
| XGBoost | `scale_pos_weight=sqrt(neg/pos)` | `scale_pos_weight=1.0` |
| CNN-LSTM / GRU / CNN-GRU | Focal Loss | CrossEntropyLoss |

---

## 학습 및 평가 방식

### K-Fold 검증

| 항목 | 내용 |
|------|------|
| 방식 | `StratifiedKFold` |
| 기본 fold 수 | 5 |
| split 대상 | trainval |
| subsampling | train split only |
| augmentation | train split only |
| validation fold | 원본 유지 |
| 보고 지표 | fold 평균, 표준편차, 최소, 최대 |

### 최종 모델

K-Fold 모델 자체를 test에 사용하지 않는다.

```text
K-Fold:
  설정 선택용
  예: best epoch, XGBoost best n_estimators

Final training:
  trainval 전체로 최종 모델 재학습

Final evaluation:
  holdout test set으로 평가
```

이 방식은 best fold 모델을 그대로 test에 사용하는 낙관적 평가를 피하기 위한 것이다.

### 평가 지표

| 지표 | 용도 |
|------|------|
| F1-score | 주 지표 |
| Recall | 주 지표, 봇넷 미탐지 감소 |
| Precision | 보조 지표 |
| ROC-AUC | 보조 지표 |
| Accuracy | 참고 지표 |

### Threshold 설정

기본값은 기존 논문들과 비교하기 쉬운 `fixed` 모드이다.

| 모드 | 의미 | 사용 목적 |
|------|------|----------|
| `fixed` | threshold 0.5 고정 | 기본 비교 실험 |
| `f1_opt` | 각 fold validation set에서 F1이 최대가 되는 threshold 선택 | threshold 민감도 / 추가 분석 |

`f1_opt`는 test set을 보지 않고 validation fold에서만 threshold를 선택한다.  
최종 holdout test 평가에는 fold별 최적 threshold의 평균값을 사용한다.

논문 본문에서는 `fixed` 결과를 주 결과로 두고, `f1_opt` 결과는 threshold 보정 후 성능 변화 또는 추가 실험으로 분리해 보고하는 것을 권장한다.

---

## 모델 목록

| 모델 | 파일 | 입력 |
|------|------|------|
| Random Forest | `train_rf.py` | `flat/` `(n, 32)` |
| XGBoost | `train_xgb.py` | `flat/` `(n, 32)` |
| CNN-LSTM | `train_cnn_lstm.py` | `seq/` `(n, 32, 1)` |
| GRU | `train_gru.py` | `seq/` `(n, 32, 1)` |
| CNN-GRU | `train_cnn_gru.py` | `seq/` `(n, 32, 1)` |

---

## 주요 스크립트

| 파일 | 역할 |
|------|------|
| `preprocess_cicids2017.py` | CIC-IDS2017 전처리 및 chi-square 피처 선택 |
| `preprocess_cicids2018.py` | CIC-IDS2018 전처리 |
| `preprocess_ctu13.py` | CTU-13 전처리 |
| `augment_utils.py` | fold-local SMOTE/GAN/WCGAN-GP 증강 |
| `train_rf.py` | RF K-Fold 검증 및 final refit |
| `train_xgb.py` | XGBoost K-Fold 검증 및 final refit |
| `train_cnn_lstm.py` | CNN-LSTM K-Fold 검증 및 final refit |
| `train_gru.py` | GRU K-Fold 검증 및 final refit |
| `train_cnn_gru.py` | CNN-GRU K-Fold 검증 및 final refit |
| `evaluate.py` | holdout test 평가 |
| `run_experiments.py` | 전체 실험 자동화 |
| `visualize.py` | 결과 시각화 |

`augment_gan.py`, `augment_wcgan_gp.py`는 사전학습 Generator 방식의 보조 스크립트로 남아 있으나, 현재 논문용 실험 경로는 `augment_utils.py`의 fold-local Generator 학습을 사용한다.

---

## 저장 구조

```text
artifacts/
  models_{dataset}/
  models_{dataset}_smote/
  models_{dataset}_gan/
  models_{dataset}_wcgan_gp/

  results_{dataset}/
  results_{dataset}_smote/
  results_{dataset}_gan/
  results_{dataset}_wcgan_gp/

  experiment_runs/
    run_YYYYMMDD_HHMMSS.json

project/data/processed/
  cicids2017/
    flat/
      X_trainval.npy
      y_trainval.npy
      X_test.npy
      y_test.npy
    seq/
      X_trainval.npy
      y_trainval.npy
      X_test.npy
      y_test.npy
      scaler_flow.pkl
    selected_features.json

  cicids2017_gan_fold_cache/
  cicids2017_wcgan_gp_fold_cache/
```

모델 저장 파일은 final trainval refit 모델이다.  
각 모델의 threshold JSON에는 `saved_model: final_trainval_refit`가 기록된다.

---

## 결과 파일

K-Fold 결과:

```text
artifacts/results_{dataset}_{augment}/{model}_flow_kfold_results.json
```

최종 holdout test 결과:

```text
artifacts/results_{dataset}_{augment}/eval_results.json
```

---

## 인자 정리

| 스크립트 | 인자 | 선택값 / 의미 |
|---------|------|---------------|
| `train_*.py` | `--dataset` | `cicids2017`, `cicids2018`, `ctu13` |
| `train_*.py` | `--augment` | `none`, `smote`, `gan`, `wcgan_gp` |
| `train_*.py` | `--n_folds` | K-Fold 수, 기본값 5 |
| `train_*.py` | `--debug` | 디버그 로그 출력 |
| `train_*.py` | `--max_normal` | 최대 Benign 수 |
| `train_*.py` | `--max_mismatch` | Botnet 비율 증가 제한, 기본값은 `cicids2017=5`, `cicids2018=2`, `ctu13=2` |
| `train_*.py` | `--threshold_mode` | `fixed`, `f1_opt` |
| `train_*.py` | `--augment_multiplier` | 증강 후 Bot 수 목표 배수, 기본값 `2` |
| `evaluate.py` | `--dataset` | 평가 데이터셋 |
| `evaluate.py` | `--augment` | 평가할 증강 설정 |
| `evaluate.py` | `--augment_multiplier` | 평가할 모델의 증강 배수 |
| `run_experiments.py` | `--datasets` | 여러 데이터셋 자동 실행 |
| `run_experiments.py` | `--augments` | 여러 증강 방식 자동 실행 |
| `run_experiments.py` | `--models` | 실행할 모델 선택 |
| `run_experiments.py` | `--threshold_mode` | 전체 학습에 threshold 모드 적용 |
| `run_experiments.py` | `--augment_multiplier` | 전체 학습에 증강 배수 적용 |
| `run_experiments.py` | `--dry_run` | 명령만 출력 |
| `run_experiments.py` | `--fold_gan_epochs` | fold-local GAN epoch |
| `run_experiments.py` | `--fold_wcgan_epochs` | fold-local WCGAN-GP epoch |

---

## 시각화

```bash
cd project/src

python visualize.py
python visualize.py --augment smote
python visualize.py --augment gan
python visualize.py --augment wcgan_gp
python visualize.py --augment smote --augment_multiplier 5
```

증강 방식별 F1 비교 그림은 어떤 `--augment`로 실행해도 함께 생성된다.

```text
artifacts/figures_augmentation_compare/01_f1_by_augmentation_dataset.png
artifacts/figures_augmentation_compare/02_delta_f1_by_augmentation.png
artifacts/figures_augmentation_compare_mul5/01_f1_by_augmentation_dataset.png
```

---

## 데이터셋 정보

| 데이터셋 | 봇넷 유형 | 비고 |
|---------|----------|------|
| CIC-IDS2017 | Neris IRC | 내부 기준 피처 선택 데이터셋 |
| CSE-CIC-IDS2018 | Ares, Zeus 계열 | CIC2017 선택 피처와 정렬 |
| CTU-13 Scenario 9 | Neris IRC | CICFlowMeter 기반 변환 |

실제 Botnet 비율은 사용한 원본 파일, 전처리 방식, split 결과에 따라 달라질 수 있다.

---

## 주요 참고 문헌

- D'Hooge et al. (2020). "Inter-dataset generalization strength of supervised machine learning methods for intrusion detection." *Journal of Information Security and Applications*, 54.
- Zhao et al. (2024). "Enhancing Network Intrusion Detection Performance using Generative Adversarial Networks." *arXiv:2404.07464*.
- Sayegh et al. (2024). "Enhanced Intrusion Detection with LSTM-Based Model, Feature Selection, and SMOTE for Imbalanced Data." *Applied Sciences*, 14(2), 479.
- de Nascimento & Hou (2025). "Uncertainty-Aware Adaptive IDS Using Hybrid CNN-LSTM with cWGAN-GP." *MDPI Safety*, 11(4), 120.
- Lin et al. (2017). "Focal Loss for Dense Object Detection." *ICCV 2017*.
- Garcia et al. (2014). "An empirical comparison of botnet detection methods." *Computers & Security*, 45.
