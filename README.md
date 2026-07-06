---
title: CarCheck
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# CarCheck — AI 차량 손상 보험 상담

YOLOv8 세그멘테이션으로 차량 손상을 감지하고, 보험 처리 vs 자비 처리 비용을 비교해주는 AI 상담 앱입니다.

## 사용 방법
1. 차량 모델·연식 입력
2. 연간 자동차보험료 입력
3. 손상 부위 사진 업로드
4. AI 분석 결과 및 처리 방법 추천 확인

## 환경 변수
- `GROQ_API_KEY` : Groq API 키 (HF Spaces Secrets에 등록)
