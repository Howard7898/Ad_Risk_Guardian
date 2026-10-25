# Ad Risk Guardian — 딥페이크 영상 · 과장광고 멀티모달 탐지

> **논문:** "Implementation of a Multimodal Integrated Detection Framework for Deepfake Videos and Exaggerated Advertisements"
> Journal of Korean Institute of Information Technology (JKIIT), Vol. 24, No. 5, pp. 201–214 (2026) · [PDF](https://ki-it.com/xml/49709/49709.pdf)
>
> 고려대학교 지능정보 SW 아카데미 7기 1조 프로젝트(2025.08–12)의 코드를 공개용으로 정리한 저장소입니다.
> API 키·데이터셋·모델 가중치는 포함하지 않으며, 원 팀 저장소의 커밋 기록은 가져오지 않았습니다.

---

## 개요

YouTube 광고 영상을 시청하는 동안 **딥페이크 영상**과 **과장광고 문장**을 탐지해 경고하는 크롬 확장 프로그램입니다.

| 구성 | 내용 |
|---|---|
| 영상 분기 | 프레임 추출 → MTCNN 얼굴 검출·정렬·크롭 → **MobileNetV3-Large + LSTM** → 딥페이크 확률 |
| 텍스트 분기 | 음성 추출 → Whisper STT → **KcELECTRA** 분류 → 과장 확률 |
| 통합 | 두 확률을 late fusion으로 결합해 광고 위험도를 판정 |
| 서비스 | FastAPI + Celery + Redis 비동기 추론 서버, Chrome Extension(Manifest V3) |

### 결과 (논문 기준, 검증셋)

| 모듈 | Accuracy | F1 |
|---|---|---|
| 영상 — MobileNetV3-Large + LSTM | 0.8833 | 0.8829 |
| 텍스트 — KcELECTRA | 0.9748 | 0.9753 |

---

## 역할 분담

| 구성원 | 담당 |
|---|---|
| **등덕호** (저장소 관리자) | **팀 리드 · 영상 분기 전 과정** — 데이터 수집·EDA, MTCNN 전처리, 시퀀스 단위 증강·클래스 밸런싱, 백본·시계열 모듈 비교 실험, Optuna 튜닝 (`pipeline/video/`) |
| 박현호 · 안수빈 · 이다은 · 임채림 · 한주은 | 텍스트 분기(`pipeline/text/`), 통합 추론(`pipeline/integration/`), 백엔드(`backend/`), 크롬 확장(`extensions/`) |

> 팀원 동의를 받아 공개합니다.

---

## 영상 분기 상세 (`pipeline/video/`)

| 단계 | 파일 | 내용 |
|---|---|---|
| 수집 | `collect.py` | KoDF · FaceForensics++ 1,500편(KoDF : FF++ = 7 : 3, 남녀 1 : 1), 학습 : 검증 = 8 : 2 분할 |
| EDA | `eda.py` | FPS · 해상도 · 밝기 분포 확인 |
| 전처리 | `preprocessing.py` | MTCNN 얼굴 검출·정렬·크롭(256px) — 검출 실패 시 직전 성공 프레임 재사용, 첫 구간은 중앙 크롭으로 대체 |
| 증강 | `preprocessing.py` | 한 영상의 모든 프레임에 같은 변환(반전·회전·이동·색상·블러·cutout)을 적용해 **시간적 일관성 유지**, 소수 클래스 증강으로 real : fake = 1 : 1, PCA · t-SNE로 원본·증강 분포 확인 |
| 모델 | `modeling.py` | MobileNetV3-Large(프레임별 공간 특징) + LSTM(프레임 간 시간 특징), Optuna 27 trials(학습률 · 배치 · 시퀀스 길이), 검증 Accuracy 기준 Early Stopping |
| 백본 비교 | `backbone_comparison/` | MobileNetV3 + LSTM · Bi-LSTM · ConvLSTM, EfficientNet + LSTM, ResNeXt + LSTM |


---

## 저장소 구조

```
Ad_Risk_Guardian/
├── pipeline/
│   ├── video/                         # 영상 분기 (등덕호)
│   │   ├── collect.py                 # 데이터 수집 · 분할
│   │   ├── eda.py                     # EDA
│   │   ├── preprocessing.py           # MTCNN 전처리 · 시퀀스 단위 증강 · 클래스 밸런싱
│   │   ├── modeling.py                # MobileNetV3-Large + LSTM 학습 · Optuna 튜닝
│   │   ├── backbone_comparison/       # 백본 · 시계열 모듈 비교 실험 5종
│   │   ├── requirements.txt           # collect · eda · modeling
│   │   └── requirements_preprocess.txt
│   ├── text/                          # 텍스트 분기
│   │   ├── collect.py · eda.py · preprocessing.py · modeling.py
│   │   ├── hparam_tuning/             # KcELECTRA 하이퍼파라미터 실험 5종 (learning rate · weight decay · dropout)
│   │   └── requirements.txt
│   └── integration/
│       └── integrated_inference.py    # 영상 + 텍스트 통합 추론
├── backend/                           # FastAPI + Celery 추론 서버
├── extensions/                        # Chrome Extension (Manifest V3)
├── .gitignore
└── README.md
```

- 텍스트 분기·통합 추론은 원래 Google Colab 노트북이었으며, 저장소에서는 `.py`로 변환해 두었습니다. 셀 경계는 `# %%`로 표시되어 VS Code · Jupytext에서 셀 단위로 실행할 수 있고, Colab 셸 명령(`!pip` 등)은 주석 처리했고, `drive.mount` 등 Colab 코드는 Colab 환경을 전제로 합니다.
- 텍스트 분기의 경로(`/content/drive/MyDrive/...`)와 영상 분기의 경로(`C:\Video_Dataset\...`)는 실행 환경에 맞게 바꿔 사용합니다.

---

## 포함하지 않은 것

- **데이터셋** — KoDF(AI Hub) · FaceForensics++는 이용 약관상 재배포가 금지되어 있어 각 출처에서 직접 신청해 받아야 합니다. 텍스트 데이터(광고 문장)도 포함하지 않습니다.
- **모델 가중치** — `backend/weights/`에 KcELECTRA 체크포인트와 `mobilenetv3_lstm_best_freeze_optuna_acc.pth`가 있어야 백엔드가 동작합니다.
- **API 키** — OpenAI Whisper API 키는 코드에 넣지 않고 환경변수로 주입합니다.

```bash
export OPENAI_API_KEY="..."   # Windows PowerShell: $env:OPENAI_API_KEY="..."
```

---

## 실행 방법

### 1. 백엔드 (Python 3.11)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py                # Redis(Docker 컨테이너) · Celery worker · FastAPI(127.0.0.1:8000) 실행
```

`run_all.py`가 Redis를 Docker로 띄우므로 Docker Desktop이 실행 중이어야 합니다.

### 2. 크롬 확장

1. `chrome://extensions/` 접속 → 개발자 모드 켜기
2. "압축 해제된 확장 프로그램 로드" → `extensions/` 폴더 선택
3. YouTube 광고 영상을 재생하면 백그라운드에서 분석 결과를 표시합니다.

### 개발 환경

- 영상 분기: Spyder(로컬 GPU) · 텍스트 분기: Google Colab · 백엔드/확장: VS Code

---

## 참고문헌

- Yang, K. (2021). Transformer-based Korean Pretrained Language Models: A Survey on Three Years of Progress. arXiv:2112.03014.
- Han, Y.-H., Huang, T.-M., Hua, K.-L., & Chen, J.-C. (2024). Towards more general video-based deepfake detection through facial component guided adaptation for foundation model.
- Nguyen, D., Astrid, M., Kacem, A., Ghorbel, E., & Aouada, D. (2025). Vulnerability-aware spatio-temporal learning for generalizable deepfake video detection.
- Xu, Y., Liang, J., Jia, G., Yang, Z., Zhang, Y., & He, R. (2023). TALL: Thumbnail layout for deepfake video detection.
- Pathan, M. et al. (2024). Deepfake Detection Using Deep Learning: ResNext and LSTM.
- Anandgasivam et al. (2024). Enhancing Deepfake Detection Through Hybrid MobileNet-LSTM Model with Real-Time Image and Video Analysis.
- Suratkar, S. et al. (2022). Deep Fake Video Detection Using Transfer Learning Approach.
