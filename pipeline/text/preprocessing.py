# -*- coding: utf-8 -*-
"""
텍스트 전처리

Jupyter(Google Colab) 노트북을 .py로 변환한 파일입니다.
- 셀 경계는 "# %%" 로 표시합니다 (VS Code · Jupytext에서 셀 단위 실행 가능).
- Colab 전용 셸 명령(!pip 등)과 매직 명령(%)은 주석 처리했습니다.
"""

# %% [markdown]
#
# ### **1. Google Drive 마운트**
# - 데이터셋 및 모델 체크포인트를 저장/불러오기 위함

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ### **2. 라이브러리 임포트**

# %%
# 데이터 처리 및 전처리 관련 라이브러리
import numpy as np
import pandas as pd

# 딥러닝 모델링(PyTorch) 및 학습 관련 라이브러리
import torch
import torch.nn as nn

# 시각화 관련 라이브러리
import matplotlib.pyplot as plt

# 유틸리티 함수 및 환경 설정 관련 라이브러리
import random
import os
import json

# Hugging Face 라이브러리
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# Transformers 관련 라이브러리
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from torch.nn import BCEWithLogitsLoss

# %% [markdown]
# ### **3. 랜덤 시드 설정 함수**

# %%
def set_random_seed(seed=42):
    """랜덤 시드 고정 (재현성 확보)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"랜덤 시드 {seed}로 고정 완료")

# %% [markdown]
# ### **4. 모델 및 토크나이저 로드 함수**

# %%
def load_model_and_tokenizer(model_name="beomi/KcELECTRA-base-v2022", num_labels=1):
    """모델 및 토크나이저 로드"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # WordPiece 토크나이저 (OOV 대응 위해 use_fast=False 유지)

    # 이진 분류 모델
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,  # BCE Loss 사용을 위해 출력 차원을 1로 설정
    )

    print(f"모델 및 토크나이저 로드 완료: {model_name}")
    return model, tokenizer

# %% [markdown]
# ### **5. 데이터 로드 및 전처리 함수**

# %%
def load_and_filter_data(file_path):
    """
    학습 데이터 로드 및 필터링

    Args:
        file_path: 데이터 파일 경로

    Returns:
        필터링된 DataFrame
    """
    # 학습 데이터 로드
    df = pd.read_excel(file_path)

    # 데이터 필터링 : 과장광고 (label=0) 데이터 중 '정상문장 (type1="Normal")은 제거함, label=1(정상광고)인 데이터는 모두 포함
    df_filtered = df[
        ((df["label"] == 0) & (df["type1"] != "Normal")) |
        (df["label"] == 1)
    ]

    print("필터 전 개수:", len(df))
    print("필터 후 개수:", len(df_filtered))
    print("줄어든 개수:", len(df) - len(df_filtered))
    print(df_filtered["label"].value_counts())

    return df_filtered

# %%

def load_test_data(file_path):
    """최종 모델 평가를 위한 데이터 로드"""
    test_df = pd.read_excel(file_path)

    print("\n=== Test 데이터 정보 ===")
    print(f"Test 크기: {len(test_df)}")
    print("Test 라벨 분포:")
    print(test_df["label"].value_counts())

    return test_df

# %%

def balance_data_by_undersampling(df_filtered, random_state=42):
    """
    언더샘플링을 통한 데이터 균형 맞추기

    Args:
        df_filtered: 필터링된 DataFrame
        random_state: 랜덤 시드

    Returns:
        균형잡힌 DataFrame
    """
    # label=1(정상광고)와 label=0(과장광고) 분리
    df_major = df_filtered[df_filtered["label"] == 1]
    df_minor = df_filtered[df_filtered["label"] == 0]

    # 랜덤 샘플링
    df_major_down = resample(
        df_major,
        replace=False,
        n_samples=len(df_minor),
        random_state=random_state
    )

    # 최종 balanced 데이터
    df_balanced = pd.concat([df_major_down, df_minor]).sample(frac=1, random_state=random_state)

    print("=== 최종 Balanced 라벨 분포 ===")
    print(df_balanced["label"].value_counts())

    return df_balanced

# %%
def split_train_val(df_balanced, test_size=0.2, random_state=42):
    """
    학습/검증 데이터 분할

    Args:
        df_balanced: 균형잡힌 DataFrame
        test_size: 검증 데이터 비율
        random_state: 랜덤 시드

    Returns:
        train_df, val_df
    """
    train_df, val_df = train_test_split(
        df_balanced,
        test_size=test_size,
        random_state=random_state,
        stratify=df_balanced['label']  # 라벨 비율 유지
    )

    print("\n[Train 라벨 분포]")
    print(train_df['label'].value_counts())
    print("\n[Validation 라벨 분포]")
    print(val_df['label'].value_counts())

    return train_df, val_df

# %% [markdown]
# ### **6. 데이터셋 생성 및 토크나이징 함수**

# %%
def create_dataset(train_df, val_df):
    """DatasetDict 생성"""
    dataset = DatasetDict({
        'train': Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True)),
        'validation': Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))
    })

    return dataset

# %%
def tokenize_function(examples, tokenizer, max_length=256):
    """
    텍스트를 모델 입력 형식으로 토크나이징 하는 함수

    Returns:
    - truncation: 최대 길이 초과 시 잘라냄
    - padding: 배치 내 모든 시퀀스를 동일한 길이로 맞춤
    - max_length: 최대 토큰 개수
    """
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

# %%
def tokenize_datasets(dataset, tokenizer, max_length=256):
    """
    Train/Validation 데이터셋 토크나이징

    Args:
        dataset: DatasetDict
        tokenizer: 토크나이저
        max_length: 최대 토큰 길이

    Returns:
        토크나이징된 데이터셋
    """
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print(f"\n=== 토크나이징 완료 (Train/Val) ===")
    print(f"Train 크기: {len(tokenized_datasets['train'])}")
    print(f"Validation 크기: {len(tokenized_datasets['validation'])}")

    return tokenized_datasets

# %%
def tokenize_test_dataset(test_df, tokenizer, max_length=256):
    """
    Test 데이터셋 토크나이징

    Args:
        test_df: Test DataFrame
        tokenizer: 토크나이저
        max_length: 최대 토큰 길이

    Returns:
        토크나이징된 테스트 데이터셋
    """
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']].reset_index(drop=True))
    tokenized_test = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print(f"\n=== 토크나이징 완료 (Test) ===")
    print(f"Test 크기: {len(tokenized_test)}")

    return tokenized_test

# %% [markdown]
# ### **7. 전처리 실행 및 데이터 저장**

# %%
# 실행 예시
SEED = 42
set_random_seed(SEED)

# 토크나이저 로드 (데이터 토크나이징에 필요)
_, tokenizer = load_model_and_tokenizer()

# 데이터 로드 및 전처리
df_filtered = load_and_filter_data("/content/광고데이터.xlsx")
test_df = load_test_data("/content/테스트데이터.xlsx")
df_balanced = balance_data_by_undersampling(df_filtered)
train_df, val_df = split_train_val(df_balanced)

# 데이터셋 생성 및 토크나이징
dataset = create_dataset(train_df, val_df)
tokenized_datasets = tokenize_datasets(dataset, tokenizer)
tokenized_test = tokenize_test_dataset(test_df, tokenizer)

print("\n전처리 완료!")

# %%
# 전처리된 데이터 저장
import pickle

# Google Drive에 저장
save_path = '/content/drive/MyDrive/preprocessed_data.pkl'

data_to_save = {
    'tokenized_datasets': tokenized_datasets,
    'tokenized_test': tokenized_test,
}

with open(save_path, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"전처리된 데이터가 {save_path}에 저장되었습니다.")

