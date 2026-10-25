# -*- coding: utf-8 -*-
"""
영상 + 텍스트 통합 추론 (late fusion)

Jupyter(Google Colab) 노트북을 .py로 변환한 파일입니다.
- 셀 경계는 "# %%" 로 표시합니다 (VS Code · Jupytext에서 셀 단위 실행 가능).
- Colab 전용 셸 명령(!pip 등)과 매직 명령(%)은 주석 처리했습니다.
"""

# %%
# !pip install yt-dlp moviepy imageio-ffmpeg openai opencv-python facenet-pytorch safetensors

# %%
import os
import cv2
import re
from pathlib import Path
from typing import List
from openai import OpenAI
import uuid

from moviepy.editor import VideoFileClip

# ============================================
# 영상 → 프레임 추출 함수
# ============================================
def extract_frames(video_path: str, output_dir: str, num_frames: int = 10) -> List[str]:
    """
    영상의 앞에서 num_frames만큼 프레임을 추출하는 함수 (OpenCV 사용)

    Args:
        video_path (str): 입력 영상 경로
        output_dir (str): 추출한 프레임 저장 경로
        num_frames (int): 뽑을 프레임 수 (기본 10)

    Returns:
        List[str]: 저장된 프레임 이미지 경로 리스트
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    extracted_paths = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # total_frames보다 num_frames이 큰 경우 조절
    actual_frames = min(num_frames, total_frames)

    # 프레임 순차 추출
    for i in range(actual_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = output_dir / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        extracted_paths.append(str(frame_path))

    cap.release()
    return extracted_paths

# ============================================
# 오디오 추출 + Whisper STT
# ============================================

# OpenAI Whisper API 클라이언트 생성
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # 환경변수로 키 주입

def extract_audio(video_path, output_audio_path=None, audio_format="mp3", whisper_model="whisper-1"):
    """
    1) 비디오에서 FFmpeg로 오디오 추출
    2) Whisper STT 적용
    3) 텍스트 반환

    :param video_path: 입력 영상 경로
    :param output_audio_path: 출력 오디오 파일명 (없으면 자동 생성)
    :param audio_format: mp3 또는 wav
    :param whisper_model: Whisper 모델 이름
    """
    video_path = Path(video_path)

    # 출력 오디오 파일명 지정
    if output_audio_path is None:
        output_audio_path = video_path.with_suffix(f".{audio_format}")
    output_audio_path = Path(output_audio_path)

    # ----------------------------
    # 1. moviepy 오디오 추출 (imageio-ffmpeg 백엔드 사용)
    # ----------------------------
    try:
        # 1. VideoFileClip 객체 생성
        video_clip = VideoFileClip(str(video_path))

        # 2. 오디오 클립 추출
        audio_clip = video_clip.audio

        # 3. 오디오 파일로 저장
        # audio_clip.write_audiofile()이 내부적으로 imageio-ffmpeg을 통해 FFmpeg 명령을 실행합니다.
        audio_clip.write_audiofile(
            str(output_audio_path),
            codec=audio_format, # moviepy는 파일 확장자에 따라 코덱을 자동 선택
            logger=None           # 터미널에 FFmpeg 로그 출력을 숨겨서 깔끔하게 만듭니다.
        )

        # 4. 클립 자원 해제
        audio_clip.close()
        video_clip.close()

    except Exception as e:
        print(f"❌ 오디오 추출 실패 (moviepy): {e}")
        return # 추출 실패 시 함수 종료
    # ----------------------------
    # 2. Whisper API로 STT 수행
    # ----------------------------
    with open(output_audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=whisper_model,  # Whisper 모델 예: "whisper-1"
            file=audio_file,
            language="ko",
        )

    text = result.text
    return text

# ============================================
# 텍스트 → 문장 단위 분리
# ============================================
def split_sentences(paragraph: str):
    """
    문단을 '.!? ' 기준으로 문장 단위로 분리하여 리스트로 반환
    """
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]

# %%
# models.py
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file

# ============================================
# MobileNetV3 + LSTM 구조 (영상 딥페이크 탐지 모델)
# ============================================

IMG_SIZE = 256
BASE_DIR = Path("/content")
MODEL_PATH = BASE_DIR / "weights" / "mobilenetv3_lstm_best_freeze_optuna_acc.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MobileNetV3LSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        latent_dim: int = 960,
        lstm_layers: int = 1,
        hidden_dim: int = 2048,
        bidirectional: bool = False,
        dropout: float = 0.4
    ):

        super().__init__()

        # MobileNetV3 Large 사전학습 모델 로드
        # features 부분만 사용 (고정된 feature extractor)
        base = models.mobilenet_v3_large(pretrained=True)
        self.feature_extractor = base.features

        # -----------------------------------
        # CNN backbone freeze (추론 시에도 그대로 사용)
        # -----------------------------------
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        # 마지막 block만 unfreeze 할 수 있으나, inference에는 영향 없음

        # 영상 프레임 feature → 1×1 평균 풀링
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 시간적 연속성을 위한 LSTM
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Fully-connected classifier
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        """
        x: 영상 시퀀스 (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape

        # (B*T, C, H, W) 형태로 CNN 통과
        x = x.view(B * T, C, H, W)
        fmap = self.feature_extractor(x)

        # (B*T, latent_dim)
        x = self.avgpool(fmap)
        x = x.view(B, T, -1)

        # LSTM으로 temporal feature 처리
        x_lstm, _ = self.lstm(x)

        # 모든 time-step 평균
        x = torch.mean(x_lstm, dim=1)

        # 최종 binary logit 출력
        x = self.fc(self.dropout(self.relu(x)))

        return x

# ============================================
# 영상 모델 로드 함수
# ============================================
def load_trained_model(model_path: Path, device: torch.device):
    """저장된 가중치 파일(.pth)을 로드하여 모델 복원"""
    assert model_path.is_file(), f"모델 파일이 없습니다: {model_path}"

    model = MobileNetV3LSTM()
    ckpt = torch.load(model_path, map_location=device)

    # 다양한 저장 형태 지원
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] 영상 모델 로드 완료")
    return model

# ============================================
# 텍스트 과장광고 분류 모델 (KcELECTRA)
# ============================================

BASE_MODEL_NAME = "beomi/KcELECTRA-base-v2022"
CHECKPOINT_ROOT = "/content/drive/MyDrive/Colab Notebooks/텍스트/BCE/checkpoints/[최종]KcELECTRA(BCE_마지막 레이어 4층만 unfreeze_X축 step_파리미터 기본값)/checkpoint-150"

class TextExaggerationClassifier:
    """
    KcELECTRA 기반 단문 과장광고 분류기
    - 입력: 문장(str)
    - 출력: 정상(1)일 확률
    """

    def __init__(self,
                 checkpoint_root: str = CHECKPOINT_ROOT,
                 base_model_name: str = BASE_MODEL_NAME,
                 device: torch.device = device):

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

        # 사전 fine-tuned 된 모델 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_ROOT)

        self.model.to(device)
        self.model.eval()
        self.device = device

        print(f"[INFO] 텍스트 모델 로드 완료")

    @torch.no_grad()
    def predict_proba(self, texts):
        """
        입력 문장 리스트에 대해 정상 확률 반환
        """
        if isinstance(texts, str):
            texts = [texts]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**enc)

        # binary classification → sigmoid 적용
        logits = outputs.logits.squeeze(-1)
        probs = torch.sigmoid(logits)

        return probs.cpu().numpy().tolist()

    def predict_label(self, texts, threshold: float = 0.5):
        """
        threshold 기준으로 정상/과장 라벨(1/0) 반환
        """
        probs = self.predict_proba(texts)
        labels = [1 if p >= threshold else 0 for p in probs]
        return labels

# ============================================
# Models 클래스: 영상 + 텍스트 모델 통합
# ============================================
class Models:
    def __init__(self):
        # 1) 영상 딥페이크 모델 로드
        self.model = load_trained_model(MODEL_PATH, device)

        # 2) 얼굴 검출용 MTCNN
        self.mtcnn = MTCNN(
            image_size=IMG_SIZE,
            margin=20,
            keep_all=False,
            device=device,
            post_process=False
        )

        # 3) 기본 이미지 전처리
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 4) 디바이스
        self.device = device

        # 5) 텍스트 과장광고 분류 모델
        self.text_classifier = TextExaggerationClassifier()

    # ============================================
    # 영상 예측: 가짜 확률(p_fake) 반환
    # ============================================
    def predict_video(self, frame_dir: str, seq_len: int = 10):
        frame_dir = Path(frame_dir)

        # jpg/png 프레임 로드
        frame_paths = sorted(frame_dir.glob("*.jpg")) + \
                      sorted(frame_dir.glob("*.png"))

        if len(frame_paths) == 0:
            raise RuntimeError(f"No frames found in {frame_dir}")

        faces = []

        # 최대 seq_len 프레임만 사용
        for i, frame_path in enumerate(frame_paths[:seq_len]):
            img = Image.open(frame_path).convert("RGB")

            # 얼굴 검출 (없으면 resize fallback)
            with torch.no_grad():
                face = self.mtcnn(img)

            if face is None:
                # 얼굴이 없으면 전체 프레임을 resize하여 사용
                img_resized = img.resize((256, 256))
                face = self.to_tensor(img_resized)

                # 디버깅용 저장
                output_path = frame_dir / f"resized_{i:03d}.jpg"
                img_resized.save(output_path)
            else:
                # mtcnn 출력이 다양한 형태일 수 있음 → Tensor로 정규화
                if isinstance(face, np.ndarray):
                    face = torch.from_numpy(face)
                if isinstance(face, Image.Image):
                    face = self.to_tensor(face)
                if face.max() > 1:
                    face = face / 255.0
                if face.ndim == 4:
                    face = face[0]

                # crop된 얼굴 저장(디버깅)
                if isinstance(face, torch.Tensor):
                    face_img = transforms.ToPILImage()(face.cpu().clamp(0.0, 1.0))
                    output_path = frame_dir / f"face_cropped_{i:03d}.jpg"
                    face_img.save(output_path)

            # Normalize 적용
            face = self.normalize(face)
            faces.append(face)

        # 프레임이 부족하면 마지막 프레임 반복
        while len(faces) < seq_len:
            faces.append(faces[-1].clone())

        faces_tensor = torch.stack(faces, dim=0)          # (T, C, H, W)
        faces_tensor = faces_tensor.unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        # 추론
        with torch.no_grad():
            logits = self.model(faces_tensor)
            p_real = torch.sigmoid(logits).item()
            p_fake = 1 - p_real

        return round(p_fake, 4)

    # ============================================
    # 텍스트 예측: 문장별 과장 확률 등을 포함한 결과 반환
    # ============================================
    def predict_text(self, text: str):
        sentences = split_sentences(text)

        if len(sentences) == 0:
            return {
                "most_exaggerated_sentence": None,
                "exaggeration_prob": 0.0,
                "sentence_probs": []
            }

        # 문장별 정상 확률 계산
        p_normals = self.text_classifier.predict_proba(sentences)

        results = []
        for sent, p_normal in zip(sentences, p_normals):
            p_exag = 1 - p_normal
            results.append((sent, float(p_normal), float(p_exag)))

        # 과장 확률이 가장 높은 문장 선택
        most_exaggerated = max(results, key=lambda x: x[2])

        return {
            "most_exaggerated_sentence": most_exaggerated[0],
            "exaggeration_prob": most_exaggerated[2],
            "sentence_probs": results
        }

# 전역 인스턴스 (tasks 및 main에서 import하여 사용)
models = Models()

# %%
import uuid
import json
import os

# ==============================================
# 1) 로컬 영상 분석 - 프레임 추출 + 딥페이크 점수 계산
# ==============================================
def task_video_local(video_path, unique_id, num_frames=10):
    # 고유 ID 기반 temp 폴더 생성
    base_dir = f"./temp/{unique_id}"
    frame_dir = f"{base_dir}/frames"
    os.makedirs(frame_dir, exist_ok=True)

    # 영상 → 프레임 추출
    extract_frames(video_path, frame_dir, num_frames)

    # 모델로 프레임 분석 → 딥페이크 점수 반환
    score = models.predict_video(frame_dir)
    return score

# ==============================================
# 2) 로컬 오디오 분석 - 음성 추출 + 텍스트 분석
# ==============================================
def task_audio_local(video_path, unique_id):
    # 고유 ID 기반 temp 폴더 생성
    base_dir = f"./temp/{unique_id}"
    os.makedirs(base_dir, exist_ok=True)

    # 영상 → 오디오 파일(mp3) 추출
    audio_path = f"{base_dir}/audio.mp3"
    transcript = extract_audio(video_path, audio_path)

    # 텍스트 기반 과장광고 확률 계산
    result = models.predict_text(transcript)

    # exaggeration_prob 값을 float 형태로 반환
    score_value = result.get("exaggeration_prob")
    return float(score_value)

# ==============================================
# 3) 딥페이크 + 과장광고 점수 → 위험 레이블 결정 + Late Fusion
# ==============================================
def combine_results_local(video_score, text_score):
    # 위험도 등급 분류
    if text_score >= 0.5 and video_score >= 0.5:
        danger_class = "매우위험"
    elif text_score >= 0.5:
        danger_class = "위험"
    elif video_score >= 0.5:
        danger_class = "주의"
    else:
        danger_class = "안전"

    # Late Fusion (가중 평균 후 inverse: 안전할수록 높게 나오지 않도록 뒤집음)
    joint_score = 0.7 * video_score + 0.3 * text_score
    final_score = 1 - joint_score

    return {
        "video_score": video_score,
        "text_score": text_score,
        "label": danger_class,
        "final_score": final_score
    }

# ==============================================
# 4) 전체 로컬 인퍼런스 파이프라인 실행
#    (프레임 분석 + 오디오 분석 → Late Fusion)
# ==============================================
def run_inference_local(video_path):
    # temp 디렉토리 분리를 위한 UUID 생성
    unique_id = str(uuid.uuid4())

    # ① 영상 기반 딥페이크 점수
    video_score = task_video_local(video_path, unique_id)

    # ② 오디오 기반 과장광고 점수
    text_score = task_audio_local(video_path, unique_id)

    # ③ 두 결과 결합 + 레이블 결정
    result = combine_results_local(video_score, text_score)
    return result

# %%
# sample data 폴더 영상 사용
video_path = "/content/test.mp4"

# inference 실행
result = run_inference_local(video_path)
result

