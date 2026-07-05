import time
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_CSV  = Path(__file__).parent.parent / "models" / "car_damage_seg" / "results.csv"
TOTAL_EPOCHS = 30
REFRESH_SEC  = 30

st.set_page_config(page_title="학습 모니터", page_icon="📈", layout="wide")
st.title("📈 YOLO 학습 모니터")

if not RESULTS_CSV.exists():
    st.warning("아직 학습이 시작되지 않았거나 results.csv가 없습니다.")
    st.stop()

df = pd.read_csv(RESULTS_CSV)
df.columns = df.columns.str.strip()

latest    = df.iloc[-1]
cur_epoch = int(latest["epoch"])
progress  = cur_epoch / TOTAL_EPOCHS

# ── 진행률 & 핵심 지표 ───────────────────────────────────
st.progress(progress, text=f"Epoch {cur_epoch} / {TOTAL_EPOCHS}  ({progress*100:.0f}%)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("현재 Epoch",      f"{cur_epoch} / {TOTAL_EPOCHS}")
m2.metric("mAP50 (Box)",    f"{latest['metrics/mAP50(B)']:.4f}")
m3.metric("mAP50 (Mask)",   f"{latest['metrics/mAP50(M)']:.4f}")
m4.metric("Train cls_loss", f"{latest['train/cls_loss']:.4f}")

st.divider()

# ── Loss 곡선 & mAP 곡선 ─────────────────────────────────
col_loss, col_map = st.columns(2)

with col_loss:
    st.subheader("Loss 곡선")
    loss_df = df[["epoch",
                  "train/box_loss", "train/seg_loss", "train/cls_loss",
                  "val/box_loss",   "val/seg_loss",   "val/cls_loss"]].rename(columns={
        "train/box_loss": "train_box",
        "train/seg_loss": "train_seg",
        "train/cls_loss": "train_cls",
        "val/box_loss":   "val_box",
        "val/seg_loss":   "val_seg",
        "val/cls_loss":   "val_cls",
    }).set_index("epoch")
    st.line_chart(loss_df)

with col_map:
    st.subheader("mAP50 추이")
    map_df = df[["epoch",
                 "metrics/mAP50(B)",
                 "metrics/mAP50(M)"]].rename(columns={
        "metrics/mAP50(B)": "mAP50_Box",
        "metrics/mAP50(M)": "mAP50_Mask",
    }).set_index("epoch")
    st.line_chart(map_df)

# ── 전체 테이블 ───────────────────────────────────────────
with st.expander("전체 epoch 데이터 보기"):
    show_cols = ["epoch",
                 "train/box_loss", "train/seg_loss", "train/cls_loss",
                 "metrics/mAP50(B)", "metrics/mAP50(M)",
                 "val/box_loss",   "val/seg_loss"]
    st.dataframe(
        df[show_cols].set_index("epoch").style.format("{:.4f}"),
        use_container_width=True,
    )

# ── 자동 새로고침 ─────────────────────────────────────────
if cur_epoch < TOTAL_EPOCHS:
    placeholder = st.empty()
    for remaining in range(REFRESH_SEC, 0, -1):
        placeholder.caption(f"🔄 {remaining}초 후 자동 새로고침")
        time.sleep(1)
    st.rerun()
else:
    st.success("✅ 학습 완료!")
    best_epoch = int(df.loc[df["metrics/mAP50(B)"].idxmax(), "epoch"])
    st.info(f"Best epoch: {best_epoch}  |  최고 mAP50(B): {df['metrics/mAP50(B)'].max():.4f}")
