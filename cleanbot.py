import streamlit as st
import os
import base64
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
login(token=HF_TOKEN)

@st.cache_resource
def load_models():
    # 감성/감정 분석용 제로샷 모델
    zero_shot_classifier = pipeline("zero-shot-classification", model='facebook/bart-large-mnli')

    # 악플 탐지 모델
    profanity_pipe = pipeline('text-classification', model='smilegate-ai/kor_unsmile')

    return zero_shot_classifier, profanity_pipe

def set_custom_ui(main_bg):
    main_bg_ext = "jpg"
        
    st.markdown(
         f"""
         <style>
         /* 전체 배경화면 설정 */
         .stApp {{
             background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover;
             background-repeat: no-repeat;
             background-attachment: fixed;
             background-position: center;
         }}

         /* 전체 기본 글자색을 흰색으로 설정 */
         body, .stApp, .stButton>button, .stTextInput>div>div>input {{
             color: white;
         }}

         /* 제목 및 캡션 색상 설정 */
         h1, h2, h3, h4, h5, h6 {{
             color: white;
         }}
         .stCaption {{
             color: #A9A9A9; /* 밝은 회색으로 설정 */
         }}

         /* 채팅 메시지 UI (다크 모드) */
         [data-testid="stChatMessage"] {{
             background-color: rgba(255, 255, 255, 0.85); /* 85% 투명도의 흰색 배경 */
             border-radius: 10px;
             padding: 15px;
             margin-bottom: 10px;
             color: black !important; /* <<< 검은색 글씨를 최우선 적용 */
         }}

         /* AI가 보낸 메시지(봇) 아바타 UI */
         [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {{
             background-color: #1E3A8A;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_custom_ui('giants_black.jpg') 

st.set_page_config(
    page_title="부산갈매기 실시간 응원톡",
    page_icon="🗣️",
    layout="centered"
)
st.title("🗣️ 부산갈매기 실시간 응원톡")
st.caption("클린봇이 불쾌한 응원글을 자동으로 관리합니다.")
# ---

with st.spinner('클릿봇이 잠에서 깨는 중입니다...🤖'):
    zero_shot_model, profanity_model = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message.get("is_censored", False):
        with st.chat_message("assistant", avatar="🤖"):
            with st.expander("🛡️ 클린봇에 의해 가려진 메시지입니다. (원인 보기)"):
                st.markdown(message["analysis_details"])
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(message["content"])

if prompt := st.chat_input("응원 메시지를 입력하세요"):
    with st.spinner("메시지를 분석하는 중..."):
        # 제로샷 모델로 감성/감정 분석 수행
        sentiment_labels = ["긍정적인", "부정적인"]
        sentiment_result = zero_shot_model(prompt, candidate_labels=sentiment_labels)
        emotion_labels = ["기쁨", "감동", "분노", "실망", "슬픔", "중립"]
        emotion_result = zero_shot_model(prompt, candidate_labels=emotion_labels)

        # 악플 탐지 모델 수행
        profanity_result = profanity_model(prompt)[0]

        # 분석 결과를 바탕으로 변수(긍정 신뢰도, 악플 신뢰도) 설정
        # 긍정 신뢰도
        # '긍정'으로 판단하면 확률 그대로 사용
        is_positive = sentiment_result['labels'][0] == "긍정적인"
        # '부정'으로 판단하면 (1 - 부정 확률)을 계산하여 긍정 신뢰도로 변환 (ex. 부정확률 80% -> 긍정 신뢰도 0.2)
        positive_score = sentiment_result['scores'][0] if is_positive else (1 - sentiment_result['scores'][0])

        # 악플 신뢰도
        # 모델이 'clean'이 아닌 다른 라벨(ex. 악플/욕설)로 판단하면 확률 그대로 사용
        is_profane = profanity_result['label'] != 'clean'
        # 'clean'으로 판단하면 (1 - (clean 확률))을 계산하여 악플 신뢰도로 변환 (ex. clean 확률 90% -> 악플 신뢰도 0.1)
        profane_score = profanity_result['score'] if is_profane else (1 - profanity_result['score'])
        
        # 최종 부정 점수 계산
        # 악플이 감지되었을 때 부정 점수가 훨씬 더 크게 증가하도록 설계
        negative_score = (1 - positive_score) + (profane_score * 1.5)
        negative_score_percent = min(round(negative_score / 2.5 * 100), 100)

    if negative_score_percent > 40:
        # 가려야 할 메시지
        analysis_details_text = (
            f"- **부정 점수**: {negative_score_percent}점\n"
            f"- **감성**: {sentiment_result['labels'][0]} (확률: {sentiment_result['scores'][0]:.2f})\n"
            f"- **주요 감정**: {emotion_result['labels'][0]} (확률: {emotion_result['scores'][0]:.2f})\n"
            f"- **악플 유형**: {profanity_result['label']} (확률: {profanity_result['score']:.2f})"
        )
        new_message = {
            "is_censored": True,
            "analysis_details": analysis_details_text
        }
    else:
        # 정상 메시지
        new_message = {
            "is_censored": False,
            "content": prompt
        }

    st.session_state.messages.append(new_message)
    st.rerun()