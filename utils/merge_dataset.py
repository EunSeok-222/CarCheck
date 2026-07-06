"""
AI Hub 차량파손 데이터 → YOLO 형식 변환 + 기존 데이터셋 병합
python3 utils/merge_dataset.py
"""
import json
import shutil
import random
from pathlib import Path

# ── 경로 설정 ────────────────────────────────────────────────────────
TRAIN_IMG_DIR = Path("/tmp/car_extract/train_images/damage")
TRAIN_LBL_DIR = Path("/tmp/car_extract/train_labels/damage")
VAL_LBL_DIR   = Path("/tmp/car_extract/val_labels/damage")

DATASET_DIR   = Path(__file__).parent.parent / "data" / "yolo_dataset"
OUT_IMG_TRAIN = DATASET_DIR / "images" / "train"
OUT_IMG_VAL   = DATASET_DIR / "images" / "val"
OUT_LBL_TRAIN = DATASET_DIR / "labels" / "train"
OUT_LBL_VAL   = DATASET_DIR / "labels" / "val"

DAMAGE_TO_CLS = {"Scratched": 0, "Breakage": 1, "Separated": 2, "Crushed": 3}

VAL_RATIO = 0.1   # 신규 데이터 중 10%를 val로 분리


def json_to_yolo(json_path: Path) -> str | None:
    """AI Hub JSON → YOLO segmentation txt 반환."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    img_w = data["images"]["width"]
    img_h = data["images"]["height"]
    lines = []

    for ann in data.get("annotations", []):
        dmg = ann.get("damage", "")
        cls = DAMAGE_TO_CLS.get(dmg)
        if cls is None:
            continue
        seg = ann.get("segmentation", [])
        if not seg or not seg[0] or not seg[0][0]:
            continue
        pts = seg[0][0]
        if len(pts) < 3:
            continue
        coords = " ".join(f"{p[0]/img_w:.6f} {p[1]/img_h:.6f}" for p in pts)
        lines.append(f"{cls} {coords}")

    return "\n".join(lines) if lines else None


def process(json_path: Path, img_src_dir: Path,
            out_img_dir: Path, out_lbl_dir: Path) -> bool:
    stem    = json_path.stem
    img_src = img_src_dir / f"{stem}.jpg"
    img_dst = out_img_dir / f"{stem}.jpg"
    lbl_dst = out_lbl_dir / f"{stem}.txt"

    if img_dst.exists():
        return False
    if not img_src.exists():
        return False

    yolo_txt = json_to_yolo(json_path)
    if not yolo_txt:
        return False

    shutil.copy(img_src, img_dst)
    lbl_dst.write_text(yolo_txt, encoding="utf-8")
    return True


def main():
    for d in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LBL_TRAIN, OUT_LBL_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 학습 라벨 처리 ───────────────────────────────────────────────
    train_jsons = sorted(TRAIN_LBL_DIR.glob("*.json"))
    print(f"학습 라벨 JSON: {len(train_jsons)}개")

    added_train = added_val = skipped = 0
    random.seed(42)

    for jp in train_jsons:
        is_val  = random.random() < VAL_RATIO
        lbl_out = OUT_LBL_VAL   if is_val else OUT_LBL_TRAIN
        img_out = OUT_IMG_VAL   if is_val else OUT_IMG_TRAIN
        ok = process(jp, TRAIN_IMG_DIR, img_out, lbl_out)
        if ok:
            if is_val: added_val += 1
            else:      added_train += 1
        else:
            skipped += 1

    print(f"  → train +{added_train} | val +{added_val} | 스킵 {skipped}")

    # ── 검증 라벨 처리 (val_labels: 매칭 이미지가 train_images에 있으면 추가) ──
    val_jsons = sorted(VAL_LBL_DIR.glob("*.json"))
    print(f"\n검증 라벨 JSON: {len(val_jsons)}개")

    matched = skipped_v = 0
    for jp in val_jsons:
        ok = process(jp, TRAIN_IMG_DIR, OUT_IMG_VAL, OUT_LBL_VAL)
        if ok:      matched   += 1
        else:       skipped_v += 1

    print(f"  → val +{matched} | 스킵(이미지 없음) {skipped_v}")

    # ── 최종 통계 ────────────────────────────────────────────────────
    total_train = len(list(OUT_IMG_TRAIN.glob("*.jpg")))
    total_val   = len(list(OUT_IMG_VAL.glob("*.jpg")))
    print(f"\n✅ 병합 완료 — train: {total_train}장 | val: {total_val}장 | 합계: {total_train+total_val}장")

    from collections import Counter
    counter = Counter()
    for lbl in list(OUT_LBL_TRAIN.glob("*.txt")) + list(OUT_LBL_VAL.glob("*.txt")):
        for line in lbl.read_text().strip().splitlines():
            if line:
                counter[int(line.split()[0])] += 1

    cls_names = {0: "Scratched", 1: "Breakage", 2: "Separated", 3: "Crushed"}
    print("\n클래스별 annotation 수:")
    for c, n in sorted(counter.items()):
        print(f"  {cls_names[c]:12s}: {n:,}개")


if __name__ == "__main__":
    main()
