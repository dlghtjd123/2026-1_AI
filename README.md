# CICIDS2017 Botnet Augmentation Experiment

수원대학교 2026-1학기 AI보안 프로젝트.

이 저장소의 현재 실험 목적은 **CICIDS2017 전체 13개 클래스를 다중 분류하면서, 희귀 클래스인 Bot 클래스만 증강했을 때 Bot 탐지 성능이 향상되는지** 확인하는 것이다.

## 연구 방향

```text
CICIDS2017 13-class multiclass classification
증강 대상: Bot class only
평가 대상: 전체 Macro F1 + Bot Precision/Recall/F1/FNR
주요 비교: 증강 없음 vs ROS/SMOTE/Borderline-SMOTE/ADASYN/GAN/WGAN-GP
기본 분류기: Random Forest
```

현재 결론은 CICIDS2017 Bot 탐지에서는 GAN/WGAN-GP보다 **SMOTE, Borderline-SMOTE, ADASYN 계열의 전통적 오버샘플링이 Bot Recall과 Bot F1 개선에 더 안정적**이라는 쪽이다.

## 사용하는 데이터

입력 데이터는 CICIDS2017의 `MachineLearningCSV` 형식 CSV이다. 원본 PCAP 파일은 사용하지 않는다.

기본 위치:

```text
project/data/raw/cic-ids2017/
```

사용 파일 예시:

```text
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

전처리에서는 중복 feature인 `Fwd Header Length.1`을 제거하여 논문에서 흔히 언급되는 CICIDS2017 기준인 **77개 numeric feature**를 사용한다.

## 실행 흐름

```text
1. CICIDS2017 CSV 로드
2. Label 정리
   - Web Attack 계열은 Web Attack 하나로 통합
   - 전체 13개 클래스로 매핑
3. feature 선택
   - ID/Timestamp/Label 제거
   - Fwd Header Length.1 제거
   - numeric feature만 사용
4. train/test split
   - 기본 test_size = 0.4
   - stratify 적용
5. train set 기준 MinMaxScaler 학습
6. feature space 구성
   - raw: scaled original feature
   - ae: Autoencoder latent feature, 선택 사항
7. train set의 Bot 클래스만 목표 개수까지 증강
8. 증강 train set으로 RF 학습
9. 원본 test set으로 13-class 평가
10. summary.csv, none_class_report.csv, results.json, synthetic_diagnostics.csv 저장
```

중요한 점은 **test set은 증강하지 않는다**는 것이다. 증강은 train set의 Bot 클래스에만 적용된다.

## 주요 코드

| 파일 | 역할 |
|---|---|
| `project/src/run_bot_augmentation_experiment.py` | 전체 실험 CLI 실행 파일 |
| `project/src/bot_augmentation_experiment.py` | 전체 실험 루프, 인자 처리, 결과 저장 |
| `project/src/cicids2017_preprocessing.py` | CSV 로드, 라벨 정리, feature 선택, 전처리 |
| `project/src/rf_evaluation.py` | Random Forest 학습 및 성능 평가 함수 |
| `project/src/cicids2017_bot_config.py` | 경로, 클래스명, 공통 상수 |
| `project/src/autoencoder_features.py` | Autoencoder 학습 및 latent feature 추출 |
| `project/src/augment_ros.py` | Random Oversampling 증강 |
| `project/src/augment_smote.py` | SMOTE 증강 |
| `project/src/augment_borderline_smote.py` | Borderline-SMOTE 증강 |
| `project/src/augment_adasyn.py` | ADASYN 증강 |
| `project/src/augment_gan.py` | GAN 기반 Bot 증강 |
| `project/src/augment_wgan_gp.py` | WGAN-GP 기반 Bot 증강 |
| `project/src/run_single_augmentation_method.py` | 증강 방식별 단독 실행 공통 launcher |
| `project/src/plot_bot_augmentation_results.py` | `summary.csv`를 막대그래프로 시각화 |

## 기본 실행

PowerShell에서는 줄바꿈 기호로 `\`가 아니라 백틱 `` ` `` 을 사용한다.

```powershell
python project/src/run_bot_augmentation_experiment.py `
  --augments none ros smote borderline_smote adasyn gan wgan_gp `
  --feature_spaces raw `
  --test_size 0.4 `
  --preprocess paper `
  --models rf `
  --gan_epochs 100 `
  --wgan_epochs 100 `
  --target_bot_count 10000 `
  --rf_estimators 100
```

## 증강량별 실행

Bot train 개수를 3,000개까지 늘리는 실험:

```powershell
python project/src/run_bot_augmentation_experiment.py `
  --augments none ros smote borderline_smote adasyn gan wgan_gp `
  --feature_spaces raw `
  --test_size 0.4 `
  --preprocess paper `
  --models rf `
  --gan_epochs 100 `
  --wgan_epochs 100 `
  --target_bot_count 3000 `
  --rf_estimators 100
```

5,000 / 20,000 / 50,000 실험은 `--target_bot_count`만 바꾸면 된다.

```powershell
--target_bot_count 5000
--target_bot_count 20000
--target_bot_count 50000
```

3,000 / 5,000 / 10,000 / 20,000 / 50,000을 한 번에 순서대로 실행하려면:

```powershell
foreach ($target in 3000, 5000, 10000, 20000, 50000) {
  python project/src/run_bot_augmentation_experiment.py `
    --augments none ros smote borderline_smote adasyn gan wgan_gp `
    --feature_spaces raw `
    --test_size 0.4 `
    --preprocess paper `
    --models rf `
    --gan_epochs 100 `
    --wgan_epochs 100 `
    --target_bot_count $target `
    --rf_estimators 100
}
```

## 빠른 확인용 실행

전체 데이터로 돌리기 전에 코드가 정상 동작하는지만 빠르게 확인할 때 사용한다.

```powershell
python project/src/run_bot_augmentation_experiment.py `
  --augments none smote `
  --feature_spaces raw `
  --test_size 0.4 `
  --preprocess paper `
  --models rf `
  --max_rows 200000 `
  --target_bot_count 3000 `
  --rf_estimators 30
```

## Autoencoder 포함 실행

최종 분석은 raw feature 중심이지만, 논문식 AE 압축을 비교하려면 `ae`를 추가한다.

```powershell
python project/src/run_bot_augmentation_experiment.py `
  --augments none ros smote borderline_smote adasyn gan wgan_gp `
  --feature_spaces raw ae `
  --test_size 0.4 `
  --preprocess paper `
  --models rf `
  --latent_dim 40 `
  --ae_epochs 20 `
  --gan_epochs 100 `
  --wgan_epochs 100 `
  --target_bot_count 10000 `
  --rf_estimators 100
```

## 옵션 설명

| 옵션 | 의미 |
|---|---|
| `--augments` | 비교할 증강 방식 목록 |
| `--feature_spaces` | `raw`는 scaled original feature, `ae`는 Autoencoder latent feature |
| `--test_size` | test set 비율 |
| `--preprocess paper` | 논문식에 가깝게 log1p 없이 MinMax scaling 사용 |
| `--models rf` | Random Forest 사용 |
| `--gan_epochs` | GAN 학습 epoch 수 |
| `--wgan_epochs` | WGAN-GP 학습 epoch 수 |
| `--target_bot_count` | train set Bot 클래스의 목표 개수 |
| `--rf_estimators` | Random Forest tree 개수 |
| `--max_rows` | 디버깅용 row 수 제한 |

## 결과 파일

실행 결과는 아래 경로에 저장된다.

```text
artifacts/ae_cgan_bot_multiclass/run_YYYYMMDD_HHMMSS_TARGET/
```

예를 들어 `--target_bot_count 3000`으로 실행하면 폴더 마지막에 `_3000`이 붙는다.

주요 파일:

| 파일 | 내용 |
|---|---|
| `summary.csv` | 실험별 핵심 metric 표 |
| `none_class_report.csv` | 증강 없음 baseline의 전체 클래스별 Precision/Recall/F1/Support |
| `results.json` | classification report, confusion matrix 포함 상세 결과 |
| `synthetic_diagnostics.csv` | GAN/WGAN-GP 생성 데이터 품질 진단 |
| `features.json` | 사용 feature 목록 |
| `*.pkl` | 학습된 scaler/model |
| `autoencoder.pt` | AE 사용 시 저장되는 autoencoder |

## 결과 시각화

가장 최근 실행 결과를 자동으로 찾아 RF/raw 기준 막대그래프를 생성한다.

```powershell
python project/src/plot_bot_augmentation_results.py
```

특정 실행 폴더를 지정하려면:

```powershell
python project/src/plot_bot_augmentation_results.py `
  --run_dir artifacts/ae_cgan_bot_multiclass/run_20260612_183021
```

특정 `summary.csv`를 직접 지정하려면:

```powershell
python project/src/plot_bot_augmentation_results.py `
  --summary_csv artifacts/ae_cgan_bot_multiclass/run_20260612_183021/summary.csv
```

AE 결과를 그리려면 feature space를 바꾼다.

```powershell
python project/src/plot_bot_augmentation_results.py `
  --feature_space ae_latent
```

그래프는 실행 폴더의 `figures/` 아래에 저장된다.

```text
figures/
  rf_scaled_original_macro_f1.png
  rf_scaled_original_bot_f1.png
  rf_scaled_original_bot_recall.png
  rf_scaled_original_bot_fnr.png
  rf_scaled_original_bot_precision_recall_f1.png
  rf_scaled_original_macro_f1_bot_f1_fnr.png
```

## 대표 결과 해석

10,000개 목표 실험 기준 대표 결과:

| Augment | Macro F1 | Bot Precision | Bot Recall | Bot F1 | Bot FNR |
|---|---:|---:|---:|---:|---:|
| none | 0.9618 | 0.8470 | 0.6196 | 0.7157 | 0.3804 |
| ROS | 0.9683 | 0.7921 | 0.8142 | 0.8030 | 0.1858 |
| SMOTE | 0.9690 | 0.7492 | 0.8817 | 0.8101 | 0.1183 |
| Borderline-SMOTE | 0.9687 | 0.7282 | 0.9033 | 0.8064 | 0.0967 |
| ADASYN | 0.9694 | 0.7369 | 0.9122 | 0.8152 | 0.0878 |
| GAN | 0.9571 | 0.8247 | 0.5445 | 0.6559 | 0.4555 |
| WGAN-GP | 0.9546 | 0.8218 | 0.4987 | 0.6207 | 0.5013 |

해석:

```text
증강 없음 대비 전통적 오버샘플링은 Bot Recall과 Bot F1을 크게 개선했다.
특히 ADASYN은 Bot FNR을 0.3804에서 0.0878까지 낮췄다.
반면 GAN/WGAN-GP는 생성 데이터 품질 문제가 있어 Bot Recall과 Bot F1이 낮았다.
```

20,000개와 50,000개까지 늘리면 Bot Recall은 더 올라갈 수 있지만, Precision이 떨어져 Bot F1 개선이 제한된다. 따라서 현재 결과에서는 **10,000개 또는 5,000개 수준이 성능 균형이 가장 좋다**고 볼 수 있다.

## 생성 데이터 품질 진단

`synthetic_diagnostics.csv`에서 주로 보는 항목:

| 항목 | 의미 |
|---|---|
| `real_fake_auc` | 진짜 Bot과 생성 Bot을 별도 분류기가 얼마나 쉽게 구분하는지 |
| `fake_to_real_nn_mean` | 생성 Bot이 실제 Bot과 평균적으로 얼마나 가까운지 |
| `fake_self_nn_ratio` | 생성 Bot끼리 뭉쳐 있는 정도 |
| `mean_abs_mean_diff` | 실제 Bot과 생성 Bot의 feature 평균 차이 |
| `mean_std_ratio` | 실제 Bot 대비 생성 Bot의 분산 비율 |

GAN의 `real_fake_auc`가 1.0에 가까우면 생성 Bot이 실제 Bot과 쉽게 구분된다는 뜻이다. 이 경우 증강 데이터가 분류기에 좋은 일반화 정보를 주기보다 오히려 Bot 결정 경계를 흐릴 수 있다.
