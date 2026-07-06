import io
import streamlit as st
from PIL import Image

from services.yolo_service import detect_damage
from services.cost_service import estimate_repair_cost
from services.insurance_service import calculate_insurance_impact
from services.llm_service import generate_report, answer_question

st.set_page_config(
    page_title="CarCheck · 차량 손상 보험 상담",
    page_icon="🚗",
    layout="centered",
)

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: #F0F4FF; }

[data-testid="stSidebar"] { background: #1B3A6B !important; }
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebarNav"] a { color: #93C5FD !important; border-radius: 8px !important; }
[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: rgba(255,255,255,0.15) !important; color: white !important;
}

.main .block-container { max-width: 780px; padding-top: 1.5rem; }

.stButton > button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.28) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.38) !important;
}

[data-testid="stChatMessage"] {
    background: white !important; border-radius: 16px !important;
    border: 1px solid #E0E8FF !important;
    box-shadow: 0 2px 10px rgba(27,58,107,0.07) !important;
    margin-bottom: 10px !important;
}

[data-testid="stChatInput"] > div {
    border: 2px solid #2563EB !important; border-radius: 16px !important;
    background: white !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius: 12px !important; border: 2px solid #E0E8FF !important;
    font-size: 1rem !important; padding: 0.6rem 0.9rem !important;
    background: white !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

.stSelectbox > div > div {
    border-radius: 12px !important; border: 2px solid #E0E8FF !important;
    background: white !important;
}

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #2563EB !important; border-radius: 16px !important;
    background: white !important; padding: 32px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { background: #F0F4FF !important; }

[data-testid="stForm"] {
    background: white !important; border-radius: 16px !important;
    padding: 20px !important; border: 1px solid #E0E8FF !important;
    box-shadow: 0 2px 10px rgba(27,58,107,0.06) !important;
}

hr { border-color: #E0E8FF !important; }

[data-testid="stMetric"] {
    background: white !important; border-radius: 14px !important;
    padding: 16px !important; border: 1px solid #E0E8FF !important;
}
</style>
""", unsafe_allow_html=True)

# ── 세션 초기화 ──────────────────────────────────────────────────────
def _init():
    if "step" not in st.session_state:
        st.session_state.step = "car_info"
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "안녕하세요! 👋\n\n"
                "**차량 정보 → 보험료 → 사진 업로드** 순서로 진행하면 "
                "AI가 손상을 분석하고 **보험 처리 vs 자비 처리** 비용을 비교해드려요.\n\n"
                "먼저 차량 정보를 입력해주세요."
            ),
        }]
_init()

# ── 헤더 ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #1B3A6B 0%, #2563EB 100%);
            padding: 26px 32px; border-radius: 20px; margin-bottom: 6px;
            box-shadow: 0 8px 28px rgba(27,58,107,0.22);">
  <div style="display:flex; align-items:center; gap:16px;">
    <div style="font-size:2.8rem; line-height:1;">🚗</div>
    <div>
      <h1 style="color:white;margin:0;font-size:2rem;font-weight:800;letter-spacing:-0.5px;">
        CarCheck
      </h1>
      <p style="color:#93C5FD;margin:5px 0 0;font-size:0.88rem;">
        AI 차량 손상 분석 &nbsp;·&nbsp; 보험 처리 비용 비교 &nbsp;·&nbsp; 전문 상담
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 단계 인디케이터 ───────────────────────────────────────────────────
def _step_indicator(step: str):
    order  = ["car_info", "premium", "photo", "analyzing", "done"]
    labels = ["차량 정보", "보험료", "사진 업로드", "분석 결과"]
    cur    = order.index(step) if step in order else 0

    items = []
    for i, label in enumerate(labels):
        step_idx = i  # labels[i] corresponds to order[i]
        if step_idx < cur:
            bg, fg, ring, num = "#1B3A6B", "white", "#1B3A6B", "✓"
        elif step_idx == cur:
            bg, fg, ring, num = "#2563EB", "white", "#2563EB", str(i + 1)
        else:
            bg, fg, ring, num = "white", "#94A3B8", "#CBD5E1", str(i + 1)

        lbl_color = "#1B3A6B" if bg == "#1B3A6B" else ("#2563EB" if bg == "#2563EB" else "#94A3B8")
        circle = (
            f'<div style="width:36px;height:36px;border-radius:50%;background:{bg};'
            f'border:2px solid {ring};display:flex;align-items:center;justify-content:center;'
            f'color:{fg};font-weight:700;font-size:0.85rem;flex-shrink:0;">{num}</div>'
        )
        items.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:5px;">'
            f'{circle}'
            f'<span style="font-size:0.72rem;color:{lbl_color};white-space:nowrap;'
            f'font-weight:{"700" if bg != "white" else "400"};">{label}</span></div>'
        )
        if i < len(labels) - 1:
            line_bg = "#1B3A6B" if i < cur else "#E2E8F0"
            items.append(
                f'<div style="flex:1;height:2px;background:{line_bg};margin-top:18px;min-width:20px;"></div>'
            )

    st.markdown(
        '<div style="display:flex;align-items:flex-start;gap:6px;margin:16px 0 20px;'
        'padding:16px 20px;background:white;border-radius:14px;border:1px solid #E0E8FF;">'
        + "".join(items) + "</div>",
        unsafe_allow_html=True,
    )

# ── 비교 카드 HTML ────────────────────────────────────────────────────
def _comparison_html(ins: dict, total: int, rec: str) -> str:
    def card(icon, title, b_val, e_label, e_val, real_val, is_best):
        left  = "border-left:4px solid #2563EB;" if is_best else "border-left:4px solid #E0E8FF;"
        shade = "box-shadow:0 4px 16px rgba(37,99,235,0.12);" if is_best else "box-shadow:0 2px 8px rgba(0,0,0,0.04);"
        badge = (
            '<span style="display:inline-block;background:#DBEAFE;color:#1D4ED8;'
            'font-size:0.75rem;font-weight:700;padding:2px 10px;border-radius:20px;'
            'margin-left:8px;">✅ 추천</span>'
        ) if is_best else ""
        green = "color:#16A34A;"
        return (
            f'<div style="flex:1;background:white;border-radius:16px;padding:22px 20px;'
            f'min-width:0;{left}{shade}">'
            f'<div style="font-size:1.1rem;font-weight:800;color:#1B3A6B;margin-bottom:14px;">'
            f'{icon} {title}{badge}</div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:9px;">'
            f'<span style="color:#64748B;font-size:0.87rem;">수리비 부담</span>'
            f'<span style="font-weight:700;font-size:1rem;{green if b_val == "0원" else ""}">{b_val}</span></div>'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:9px;">'
            f'<span style="color:#64748B;font-size:0.87rem;">{e_label}</span>'
            f'<span style="font-weight:600;color:#EF4444;">{e_val}</span></div>'
            f'<div style="height:1px;background:#F1F5F9;margin:12px 0;"></div>'
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span style="color:#1B3A6B;font-weight:700;font-size:0.88rem;">실질 부담</span>'
            f'<span style="font-size:1.35rem;font-weight:800;color:#1B3A6B;">{real_val}</span></div>'
            f'</div>'
        )

    ins_card = card(
        "🏢", "보험처리",
        "0원",
        f"{ins['surcharge_years']}년 보험료 인상", f"+{ins['total_increase']:,}원",
        f"{ins['total_increase']:,}원",
        rec == "insurance",
    )
    self_card = card(
        "💵", "자비처리",
        f"{total:,}원",
        "보험료 인상", "<span style='color:#16A34A;font-weight:600;'>0원</span>",
        f"{total:,}원",
        rec == "self",
    )
    return f'<div style="display:flex;gap:14px;margin:10px 0;">{ins_card}{self_card}</div>'

# ── 이미지 → bytes ────────────────────────────────────────────────────
def _to_bytes(img) -> bytes:
    if isinstance(img, bytes):
        return img
    buf = io.BytesIO()
    pil = img if isinstance(img, Image.Image) else Image.fromarray(img)
    pil.save(buf, format="JPEG", quality=90)
    return buf.getvalue()

# ── 메시지 렌더링 ─────────────────────────────────────────────────────
def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg:
                st.image(msg["image"], use_container_width=True)
            if "html" in msg:
                st.markdown(msg["html"], unsafe_allow_html=True)

# ── 렌더링 ───────────────────────────────────────────────────────────
_step_indicator(st.session_state.step)
render_messages()

# ── 단계별 UI ─────────────────────────────────────────────────────────
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
        submitted = st.form_submit_button("다음 단계 →", use_container_width=True, type="primary")

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
        submitted = st.form_submit_button("다음 단계 →", use_container_width=True, type="primary")

    if submitted:
        st.session_state.annual_premium = premium
        st.session_state.messages += [
            {"role": "user",      "content": f"💰 연간 보험료: **{premium:,}원**"},
            {"role": "assistant", "content": (
                "확인했어요!\n\n"
                "마지막으로 **수리가 필요한 부위의 사진**을 업로드해주세요. 📸\n"
                "선명하게 촬영할수록 분석 정확도가 올라가요."
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
        img_bytes = uploaded.getvalue()
        image     = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.session_state.upload_bytes = img_bytes
        st.session_state.messages.append(
            {"role": "user", "content": "📸 사진 업로드 완료", "image": _to_bytes(image)}
        )
        st.session_state.step = "analyzing"
        st.rerun()

elif step == "analyzing":
    image = Image.open(io.BytesIO(st.session_state.upload_bytes)).convert("RGB")

    with st.spinner("🔍 AI가 손상 부위를 분석하는 중... 잠시만 기다려주세요"):
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

    ins    = insurance_result
    total  = repair_cost["total"]
    rec    = ins["recommendation"]
    saving = abs(ins["total_increase"] - total)
    winner = "자비처리" if rec == "self" else "보험처리"

    st.session_state.messages += [
        {
            "role":    "assistant",
            "content": f"✅ 분석 완료!\n\n**감지된 손상:**\n{damages_lines}",
            "image":   _to_bytes(damage_result["annotated_image"]),
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
                "추가로 보험 관련 궁금한 점이 있으시면 아래에 질문해주세요. 😊\n\n"
                "> 📋 **삼성화재 약관 (2025.08.16)** · **손해보험협회 할증 기준 (2024)** 기반 답변"
            ),
        },
    ]
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
    if user_input := st.chat_input("보험 관련 질문을 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("답변 생성 중..."):
            answer = answer_question(
                user_query=user_input,
                chat_history=st.session_state.chat_history[:-1],
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
