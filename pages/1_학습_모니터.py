from pathlib import Path
import pandas as pd
import streamlit as st

RESULTS_CSV = Path(__file__).parent.parent / "models" / "car_damage_seg" / "results.csv"

st.set_page_config(page_title="CarCheck · 학습 결과", page_icon="📊", layout="wide")

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
.main .block-container { padding-top: 1.5rem; }
[data-testid="stMetric"] {
    background: white !important; border-radius: 14px !important;
    padding: 16px !important; border: 1px solid #E0E8FF !important;
    box-shadow: 0 2px 8px rgba(27,58,107,0.07) !important;
}
hr { border-color: #E0E8FF !important; }
</style>
""", unsafe_allow_html=True)

# ── 헤더 ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #1B3A6B 0%, #2563EB 100%);
            padding: 26px 32px; border-radius: 20px; margin-bottom: 24px;
            box-shadow: 0 8px 28px rgba(27,58,107,0.22);">
  <div style="display:flex; align-items:center; gap:16px;">
    <div style="font-size:2.8rem; line-height:1;">📊</div>
    <div>
      <h1 style="color:white;margin:0;font-size:2rem;font-weight:800;letter-spacing:-0.5px;">
        학습 결과 리포트
      </h1>
      <p style="color:#93C5FD;margin:5px 0 0;font-size:0.88rem;">
        YOLOv8n-seg · Colab T4 GPU · 100 Epochs
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 섹션 1: Colab 100 epoch 핵심 지표 ────────────────────────────────
st.markdown("### 🏆 Colab 학습 결과 요약 (100 Epochs)")
st.caption("Tesla T4 GPU · YOLOv8n-seg · 960장 학습 / 240장 검증")

c1, c2, c3, c4 = st.columns(4)
c1.metric("최고 mAP50 (Box)",  "0.2398", help="Bounding Box 기준 최고 mAP50")
c2.metric("최고 mAP50 (Mask)", "0.2119", help="Segmentation Mask 기준 최고 mAP50")
c3.metric("학습 데이터",        "960장",  help="train 960장 / val 240장")
c4.metric("학습 시간",          "~1.5시간", help="T4 GPU 기준")

st.divider()

# ── 섹션 2: 학습 설정 + 클래스 분포 ──────────────────────────────────
st.markdown("### ⚙️ 학습 설정")

col_cfg, col_cls = st.columns(2)

with col_cfg:
    cfg_df = pd.DataFrame({
        "항목": ["모델", "Epochs", "Batch Size", "Image Size",
                 "Optimizer", "Learning Rate", "Device", "Patience"],
        "값":   ["YOLOv8n-seg", "100", "16", "640 × 640",
                 "AdamW", "0.00125", "Tesla T4 GPU", "20 (Early Stop)"],
    })
    st.dataframe(cfg_df.set_index("항목"), use_container_width=True, height=318)

with col_cls:
    st.markdown("**클래스별 학습 데이터 분포**")
    cls_df = pd.DataFrame({
        "클래스":        ["Scratched (긁힘)", "Separated (분리)", "Breakage (파손)", "Crushed (찌그러짐)"],
        "Annotation 수": [1329, 302, 205, 174],
        "비율 (%)":      [65.3, 14.8, 10.1, 8.5],
    })
    st.dataframe(cls_df.set_index("클래스"), use_container_width=True)
    st.bar_chart(
        cls_df.set_index("클래스")["Annotation 수"],
        color="#2563EB",
        height=180,
    )

st.divider()

# ── 섹션 3: 학습 곡선 ─────────────────────────────────────────────────
st.markdown("### 📈 학습 곡선 (로컬 30 Epoch CSV)")

if RESULTS_CSV.exists():
    df = pd.read_csv(RESULTS_CSV)
    df.columns = df.columns.str.strip()

    col_map, col_loss = st.columns(2)

    with col_map:
        st.markdown("**mAP50 추이**")
        map_df = df[["epoch", "metrics/mAP50(B)", "metrics/mAP50(M)"]].rename(columns={
            "metrics/mAP50(B)": "mAP50 Box",
            "metrics/mAP50(M)": "mAP50 Mask",
        }).set_index("epoch")
        st.line_chart(map_df, color=["#2563EB", "#F97316"], height=260)

    with col_loss:
        st.markdown("**Train Loss 추이**")
        loss_df = df[["epoch", "train/box_loss", "train/seg_loss", "train/cls_loss"]].rename(columns={
            "train/box_loss": "Box Loss",
            "train/seg_loss": "Seg Loss",
            "train/cls_loss": "Cls Loss",
        }).set_index("epoch")
        st.line_chart(loss_df, color=["#1B3A6B", "#EF4444", "#16A34A"], height=260)

    best_epoch = int(df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"])
    best_map   = df["metrics/mAP50(B)"].max()
    st.success(f"✅ 로컬 Best — Epoch **{best_epoch}** · mAP50(Box): **{best_map:.4f}**")

    with st.expander("📋 전체 Epoch 데이터 보기"):
        show = ["epoch", "train/box_loss", "train/seg_loss", "train/cls_loss",
                "metrics/mAP50(B)", "metrics/mAP50(M)", "val/box_loss", "val/seg_loss"]
        st.dataframe(
            df[show].set_index("epoch").style
                .format("{:.4f}")
                .highlight_max(subset=["metrics/mAP50(B)", "metrics/mAP50(M)"], color="#DBEAFE"),
            use_container_width=True,
        )
else:
    st.info("로컬 results.csv 없음 — 로컬 학습을 실행하면 차트가 표시됩니다.")

st.divider()

# ── 섹션 4: 다음 학습 개선 계획 ──────────────────────────────────────
st.markdown("### 🚀 다음 학습 개선 계획")

plan_df = pd.DataFrame({
    "항목":           ["총 학습 데이터", "Epoch", "Scratched", "Breakage", "Separated", "Crushed", "예상 mAP50"],
    "현재 (완료)":     ["960장",  "100", "1,329개", "205개",    "302개",    "174개",    "0.239"],
    "다음 학습 예정":   ["9,744장 ×10.1", "10",
                       "19,461개 ×14.6", "2,977개 ×14.5",
                       "4,461개 ×14.8",  "3,053개 ×17.5", "0.40+ (예상)"],
})
st.dataframe(plan_df.set_index("항목"), use_container_width=True)

st.markdown("""
<div style="background:white;border-radius:14px;padding:18px 22px;
            border:1px solid #E0E8FF;margin-top:8px;
            box-shadow:0 2px 8px rgba(27,58,107,0.06);">
  <div style="color:#1B3A6B;font-weight:700;font-size:1rem;margin-bottom:10px;">💡 개선 포인트</div>
  <ul style="color:#475569;margin:0;padding-left:18px;line-height:2.0;">
    <li>기존 960장 → <b>9,744장</b>으로 10배 증가 (AI Hub 160. 차량파손 이미지 데이터 추가)</li>
    <li>소수 클래스(Breakage·Crushed) 데이터 <b>15~18배 증가</b> → 클래스 불균형 해소</li>
    <li>10x 데이터 × 10 epoch = 기존 100 epoch 대비 <b>총 학습량 3배 이상</b></li>
    <li>Colab 무료 플랜 주간 재개 후 파인튜닝 예정</li>
  </ul>
</div>
""", unsafe_allow_html=True)
