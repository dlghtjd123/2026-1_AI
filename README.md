# 2026-1_AI
수원대학교 4학년 2026-1학기 AI보안 

# ==============================================
# 설치 방법
# ==============================================
# 1. PyTorch (CUDA 12.4) 먼저 설치:
#    pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
#
# 2. 나머지 패키지 설치:
#    pip install -r requirements.txt
# ==============================================

# ==============================================
# 실행 순서
# ==============================================

# 1. 전처리
python preprocess_cicids.py
python preprocess_ctu13.py

# 2. CIC 단독 평가용 학습 (77 features)
python train_rf.py --mode full
python train_xgb.py --mode full
python train_cnn_lstm.py --mode full

# 3. 평가 (1단계 CIC + 2단계 CTU 비교표 출력)
python evaluate.py



# (생략) CTU 데이터셋 통일 (CICFlowMeter, src 폴더 기준)
python pcap_to_csv.py

# (생략) CTU 교차검증용 학습 (8 features)
python train_rf.py --mode common
python train_xgb.py --mode common
python train_cnn_lstm.py --mode common