import streamlit as st
from PIL import Image

from services.yolo_service import detect_damage
from services.cost_service import estimate_repair_cost
from services.insurance_service import calculate_insurance_impact
from services.llm_service import generate_report, answer_question

st.set_page_config(
    page_title="차량 손상 보험 상담",
    page_icon="🚗",
    layout="centered",
)

# ── 세션 초기화 ──────────────────────────────────────────────────────
def _init():
    if "step" not in st.session_state:
        st.session_state.step = "car_info"
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요! 🚗 **차량 손상 보험 상담 서비스**입니다.\n\n"
                    "사진 한 장으로 손상을 분석하고, **보험 처리 vs 자비 처리** 중 "
                    "어떤 게 유리한지 알려드려요.\n\n"
                    "먼저 차량 정보를 입력해주세요."
                ),
            }
        ]

_init()

# ── 메시지 렌더링 ─────────────────────────────────────────────────────
def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg:
                st.image(msg["image"], use_container_width=True)
            if "html" in msg:
                st.markdown(msg["html"], unsafe_allow_html=True)

# ── 비교 카드 HTML ────────────────────────────────────────────────────
def _comparison_html(ins, total, rec):
    def card(title, burden, extra_label, extra_val, real_burden, is_best):
        bg     = "#e8f5e9" if is_best else "#f8f9fa"
        border = "#4caf50" if is_best else "#dee2e6"
        badge  = "<br><span style='color:#4caf50;font-weight:700;'>✅ 추천</span>" if is_best else ""
        return (
            f'<div style="flex:1;background:{bg};border:2px solid {border};'
            f'border-radius:14px;padding:18px;min-width:0;">'
            f'<div style="font-size:1.1em;font-weight:700;margin-bottom:10px;">{title}</div>'
            f'<div>수리비 부담 &nbsp;<b>{burden}</b></div>'
            f'<div>{extra_label} &nbsp;<b>{extra_val}</b></div>'
            f'<hr style="margin:10px 0">'
            f'<div>실질 부담 &nbsp;<b>{real_burden}</b></div>'
            f'{badge}</div>'
        )

    ins_card = card(
        "🏢 보험처리", "0원",
        f"{ins['surcharge_years']}년 보험료 인상",
        f"+{ins['total_increase']:,}원",
        f"{ins['total_increase']:,}원",
        rec == "insurance",
    )
    self_card = card(
        "💵 자비처리", f"{total:,}원",
        "보험료 인상", "0원",
        f"{total:,}원",
        rec == "self",
    )
    return f'<div style="display:flex;gap:14px;margin-top:8px;">{ins_card}{self_card}</div>'

# ── 대화 이력 출력 ────────────────────────────────────────────────────
render_messages()

# ── 단계별 입력 UI ────────────────────────────────────────────────────
step = st.session_state.step

if step == "car_info":
    with st.form("car_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        car_model = col1.text_input(
            "차량 모델", placeholder="예) 현대 아반떼, 기아 K5, 제네시스 G80",
            label_visibility="collapsed",
        )
        car_year = col2.selectbox(
            "연식", list(range(2025, 2009, -1)), index=5,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("다음 →", use_container_width=True, type="primary")

    if submitted:
        if not car_model.strip():
            st.warning("차량 모델명을 입력해주세요.")
        else:
            st.session_state.car_model = car_model.strip()
            st.session_state.car_year  = car_year
            st.session_state.messages += [
                {"role": "user",      "content": f"🚗 **{car_model}** {car_year}년형"},
                {"role": "assistant", "content": (
                    f"**{car_model} {car_year}년형** 접수했습니다! 😊\n\n"
                    "다음으로 **현재 연간 자동차보험료**를 입력해주세요.\n"
                    "보험 가입증서 또는 보험사 앱에서 확인하실 수 있어요."
                )},
            ]
            st.session_state.step = "premium"
            st.rerun()

elif step == "premium":
    with st.form("premium_form", clear_on_submit=True):
        premium = st.number_input(
            "연간 자동차보험료 (원)",
            min_value=100_000, max_value=5_000_000,
            value=600_000, step=10_000, format="%d",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("다음 →", use_container_width=True, type="primary")

    if submitted:
        st.session_state.annual_premium = premium
        st.session_state.messages += [
            {"role": "user",      "content": f"💰 연간 보험료: **{premium:,}원**"},
            {"role": "assistant", "content": (
                "확인했어요!\n\n"
                "마지막으로 **수리가 필요한 부위의 사진**을 업로드해주세요. 📸\n"
                "선명하게 촬영할수록 정확도가 높아져요."
            )},
        ]
        st.session_state.step = "photo"
        st.rerun()

elif step == "photo":
    uploaded = st.file_uploader(
        "사진 업로드", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.session_state.upload_bytes = uploaded.getvalue()
        st.session_state.messages.append(
            {"role": "user", "content": "📸 사진 업로드 완료", "image": image}
        )
        st.session_state.step = "analyzing"
        st.rerun()

elif step == "analyzing":
    import io
    image = Image.open(io.BytesIO(st.session_state.upload_bytes)).convert("RGB")

    with st.spinner("AI가 손상 부위를 분석하는 중... 잠시만 기다려주세요 🔍"):
        damage_result    = detect_damage(image)
        repair_cost      = estimate_repair_cost(damage_result)
        insurance_result = calculate_insurance_impact(
            repair_cost=repair_cost["total"],
            annual_premium=st.session_state.annual_premium,
        )
        report = generate_report(damage_result, repair_cost, insurance_result)

    BADGE = {"Scratched": "🟡", "Breakage": "🔴", "Crushed": "🔴", "Separated": "🟠"}
    damages_lines = "\n".join(
        f"{BADGE.get(d['type'], '⚪')} **{d['type_ko']}** — {d['part']} (신뢰도 {d['confidence']:.0%})"
        for d in damage_result["damages"]
    ) or "감지된 손상 없음"

    ins   = insurance_result
    total = repair_cost["total"]
    rec   = ins["recommendation"]
    saving = abs(ins["total_increase"] - total)
    winner = "자비처리" if rec == "self" else "보험처리"

    st.session_state.messages += [
        {
            "role":    "assistant",
            "content": f"분석 완료! 아래 결과를 확인해주세요.\n\n**감지된 손상:**\n{damages_lines}",
            "image":   damage_result["annotated_image"],
            "html":    _comparison_html(ins, total, rec),
        },
        {
            "role":    "assistant",
            "content": (
                f"💡 **{winner}**가 약 **{saving:,}원** 더 유리합니다.\n\n"
                "---\n\n"
                f"{report}"
            ),
        },
        {
            "role":    "assistant",
            "content": (
                "보험 관련 궁금한 점이 있으면 아래 채팅창에 질문해주세요. 😊\n"
                "> 📋 **삼성화재 약관 (2025.08.16)** · **손해보험협회 할증 기준 (2024)** 기반으로 답변드립니다."
            ),
        },
    ]
    # 추가 질문용 컨텍스트 저장
    st.session_state.analysis_context = (
        f"차량: {st.session_state.get('car_model','')} {st.session_state.get('car_year','')}년형\n"
        f"감지 손상: {damages_lines}\n"
        f"예상 수리비: {total:,}원\n"
        f"추천: {winner} ({saving:,}원 유리)"
    )
    st.session_state.chat_history = []
    st.session_state.step = "done"
    st.rerun()

elif step == "done":
    # ── 추가 질문 채팅 입력 ───────────────────────────────────────────
    if user_input := st.chat_input("보험 관련 질문을 입력하세요..."):
        # 사용자 메시지 표시
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("답변 생성 중..."):
            answer = answer_question(
                user_query=user_input,
                chat_history=st.session_state.chat_history[:-1],  # 방금 추가한 것 제외
                analysis_context=st.session_state.get("analysis_context", ""),
            )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    st.divider()
    if st.button("🔄 새로운 상담 시작", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
