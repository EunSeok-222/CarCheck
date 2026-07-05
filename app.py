import streamlit as st
from PIL import Image

from services.yolo_service import detect_damage
from services.cost_service import estimate_repair_cost
from services.insurance_service import calculate_insurance_impact
from services.llm_service import generate_report

st.set_page_config(
    page_title="사고차 수리비 분석기",
    page_icon="🚗",
    layout="wide",
)

# ── 헤더 ──────────────────────────────────────────────────────────────
st.title("🚗 사고차 수리비 분석기")
st.caption("사진 한 장으로 수리비 예측 + 보험처리 vs 자비처리 손익 비교")
st.divider()

# ── 입력 영역 ──────────────────────────────────────────────────────────
col_img, col_info = st.columns([2, 1])

with col_img:
    uploaded_file = st.file_uploader(
        "사고 차량 사진을 올려주세요",
        type=["jpg", "jpeg", "png"],
        help="JPG / PNG, 최대 10MB",
    )
    if uploaded_file:
        st.image(uploaded_file, caption="업로드된 사진", use_container_width=True)

with col_info:
    st.subheader("보험 정보")
    annual_premium = st.number_input(
        "현재 연간 자동차보험료 (원)",
        min_value=100_000,
        max_value=5_000_000,
        value=600_000,
        step=10_000,
        format="%d",
        help="보험 가입증서에서 확인하세요",
    )
    car_year = st.selectbox(
        "차량 연식",
        options=list(range(2025, 2009, -1)),
        index=5,
    )
    st.caption(f"선택: {car_year}년형")

# ── 분석 버튼 ──────────────────────────────────────────────────────────
st.divider()
analyze_clicked = st.button(
    "🔍 분석하기",
    type="primary",
    disabled=(uploaded_file is None),
    use_container_width=True,
)

if not uploaded_file:
    st.info("사고 사진을 업로드하면 분석을 시작할 수 있어요.")
    st.stop()

# ── 분석 실행 ──────────────────────────────────────────────────────────
if analyze_clicked:
    for key in ["damage_result", "repair_cost", "insurance_result", "report"]:
        st.session_state.pop(key, None)

if "damage_result" not in st.session_state and analyze_clicked:
    with st.spinner("AI가 손상 부위를 분석하는 중..."):
        image = Image.open(uploaded_file)

        damage_result    = detect_damage(image)
        repair_cost      = estimate_repair_cost(damage_result)
        insurance_result = calculate_insurance_impact(
            repair_cost=repair_cost["total"],
            annual_premium=annual_premium,
        )
        report = generate_report(damage_result, repair_cost, insurance_result)

        st.session_state["damage_result"]    = damage_result
        st.session_state["repair_cost"]      = repair_cost
        st.session_state["insurance_result"] = insurance_result
        st.session_state["report"]           = report

# ── 결과 표시 ──────────────────────────────────────────────────────────
if "damage_result" not in st.session_state:
    st.stop()

damage_result    = st.session_state["damage_result"]
repair_cost      = st.session_state["repair_cost"]
insurance_result = st.session_state["insurance_result"]
report           = st.session_state["report"]

st.divider()
st.subheader("📊 분석 결과")

# 이미지 비교
col_orig, col_anno = st.columns(2)
with col_orig:
    st.markdown("**원본 사진**")
    st.image(damage_result["original_image"], use_container_width=True)
with col_anno:
    st.markdown("**손상 감지 결과**")
    st.image(damage_result["annotated_image"], use_container_width=True)

# 감지된 손상 목록
st.markdown("**감지된 손상**")
BADGE = {"Scratched": "🟡", "Breakage": "🔴", "Crushed": "🔴", "Separated": "🟠"}
for d in damage_result["damages"]:
    icon = BADGE.get(d["type"], "⚪")
    st.markdown(f"{icon} `{d['type_ko']}` — {d['part']} &nbsp; 신뢰도: **{d['confidence']:.0%}**")

st.divider()

# ── 비용 비교 카드 ──────────────────────────────────────────────────────
st.subheader("💰 보험처리 vs 자비처리")

ins   = insurance_result
total = repair_cost["total"]
rec   = ins["recommendation"]

col_ins, col_self = st.columns(2)

with col_ins:
    is_best = rec == "insurance"
    border  = "#4caf50" if is_best else "#e0e0e0"
    bg      = "#e8f5e9" if is_best else "#fafafa"
    st.markdown(
        f"""<div style="background:{bg};border:2px solid {border};
                        border-radius:12px;padding:20px;">
            <h3>🏢 보험처리</h3>
            <p>수리비 부담: <b>0원</b></p>
            <p>{ins['surcharge_years']}년 보험료 인상: <b>+{ins['total_increase']:,}원</b>
               &nbsp;(연 {ins['increase_per_year']:,}원)</p>
            <hr>
            <p>실질 부담: <b>{ins['total_increase']:,}원</b></p>
            {"<p style='color:#4caf50;font-weight:700;'>✅ 추천</p>" if is_best else ""}
            </div>""",
        unsafe_allow_html=True,
    )

with col_self:
    is_best = rec == "self"
    border  = "#4caf50" if is_best else "#e0e0e0"
    bg      = "#e8f5e9" if is_best else "#fafafa"
    st.markdown(
        f"""<div style="background:{bg};border:2px solid {border};
                        border-radius:12px;padding:20px;">
            <h3>💵 자비처리</h3>
            <p>수리비 부담: <b>{total:,}원</b></p>
            <p>보험료 인상: <b>0원</b></p>
            <hr>
            <p>실질 부담: <b>{total:,}원</b></p>
            {"<p style='color:#4caf50;font-weight:700;'>✅ 추천</p>" if is_best else ""}
            </div>""",
        unsafe_allow_html=True,
    )

saving = abs(ins["total_increase"] - total)
winner = "자비처리" if rec == "self" else "보험처리"
st.success(f"💡 **{winner}**가 약 **{saving:,}원** 더 유리합니다.")

# 수리 내역 상세
with st.expander("🔧 수리비 상세 내역 보기"):
    ACTION_KO = {"coating": "도색", "exchange": "교체", "repair": "수리", "sheet_metal": "판금"}
    for item in repair_cost["breakdown"]:
        c1, c2 = st.columns([3, 1])
        c1.text(f"{item['part']}  —  {ACTION_KO.get(item['action'], item['action'])}")
        c2.text(f"{item['cost']:,}원")
    st.markdown(f"**합계: {total:,}원**")

# AI 리포트
with st.expander("📄 AI 상세 리포트 보기"):
    st.markdown(report)

# 다시 분석
st.divider()
if st.button("🔄 새 사진으로 다시 분석", use_container_width=True):
    for key in ["damage_result", "repair_cost", "insurance_result", "report"]:
        st.session_state.pop(key, None)
    st.rerun()
