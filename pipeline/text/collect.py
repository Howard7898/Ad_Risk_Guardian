# -*- coding: utf-8 -*-
"""
광고 음성 수집 · STT 텍스트화

Jupyter(Google Colab) 노트북을 .py로 변환한 파일입니다.
- 셀 경계는 "# %%" 로 표시합니다 (VS Code · Jupytext에서 셀 단위 실행 가능).
- Colab 전용 셸 명령(!pip 등)과 매직 명령(%)은 주석 처리했습니다.
"""

# %%
# !pip install pytube boto3 requests

# %%
# !pip install --upgrade pytube

# %% [markdown]
#  ### **학습 데이터 수집 : Clova Speech API 사용**

# %%
import os
from yt_dlp import YoutubeDL

#YouTube 오디오 다운로드
def download_multiple_audios(video_urls, download_path="downloads", use_index_prefix=True, limit=None):
    """
     yt-dlp를 사용해 YouTube 오디오를 로컬에 저장하는 함수

    - video_urls: 유튜브 영상 URL 리스트
    - download_path: 저장할 폴더 경로
    - use_index_prefix: True면 01_, 02_ 처럼 번호 prefix 붙여서 저장
    - limit: 처리할 최대 개수 (None이면 전체)
    """
    if limit is not None:
        video_urls = video_urls[:limit]

    os.makedirs(download_path, exist_ok=True)

    print(f"[INFO] 이번에 다운로드할 영상 개수: {len(video_urls)}")

    for idx, url in enumerate(video_urls, start=1):
        print("\n" + "=" * 60)
        print(f"[INFO] ({idx}/{len(video_urls)}) 다운로드 중: {url}")
        print("=" * 60)

        try:
            prefix = f"{idx:02d}_" if use_index_prefix else ""
            download_youtube_audio(url, download_path=download_path, prefix=prefix)
        except Exception as e:
            print(f"[ERROR] ({idx}/{len(video_urls)}) 영상 다운로드 실패")
            print("URL:", url)
            print("에러:", e)

# 실행 테스트용 (스크립트로 실행할 때만 동작)
if __name__ == "__main__":
    video_urls = [
                "https://www.youtube.com/watch?v=JKu_P5xtlsw"
    ]
    download_multiple_audios(video_urls, download_path="downloads", use_index_prefix=True, limit=None)

# %%
import requests
import sys
# Clova Speech API 정보
CLOVA_SPEECH_SECRET_KEY = '' # Secret key 입력

CLOVA_SPEECH_INVOKE_URL = '' # Invoke URL 입력
def convert_speech_to_text(file_url):
   # Clova Speech API를 호출하여 음성 파일 URL을 텍스트로 변환하는 함수
    headers = {'X-CLOVASPEECH-API-KEY': CLOVA_SPEECH_SECRET_KEY}
    data = {
        'url': file_url,
        'language': 'ko-KR',
        'completion': 'sync'    }
    response = requests.post(f'{CLOVA_SPEECH_INVOKE_URL}/recognizer/url', headers=headers, json=data)
    return response.json()
if __name__ == '__main__':
   #object storage 음성 파일 경로
    file_urls = [
    # Object Storage 음성 파일 URL 입력
]
    # 여러 파일을 순서대로 처리
    for file_url in file_urls:
        print(f"=== 처리 중: {file_url} ===")
        try:
            response = convert_speech_to_text(file_url)
            # 전체 response 찍고 싶으면 아래 한 줄만 두고
            # print(response)

            # text만 출력
            text = response.get("text", "")
            print("인식 결과:", text)
        except Exception as e:
            print("에러 발생:", e)
        print()

#코드 참조 블로그 링크 : https://velog.io/@brickmate_-/NCP-Clova-Speech-Studio-API-%ED%99%9C%EC%9A%A9-%ED%95%B4%EB%B3%B4%EA%B8%B0

