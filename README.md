# 2026-1_AI
수원대학교 4학년 2026-1학기 AI보안

# 설치 방법
# 1. PyTorch (CUDA 12.4) 먼저 설치:
#    pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
# 2. 나머지 패키지 설치:
#    pip install -r requirements.txt

# 실행 순서
# 1. 전처리
python preprocess_cicids2017.py
python preprocess_cicids2018.py 
# 2. CIC 단독 평가용 학습
python train_rf.py
python train_xgb.py
python train_cnn_lstm.py
python train_gru.py
python train_cnn_gru.py
# 3. Baseline cross-dataset 성능 측정
python evaluate.py
# 4. 결과 시각화
python visualize.py               # 결과 시각화