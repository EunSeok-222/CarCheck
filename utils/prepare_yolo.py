"""
AI Hub JSON 라벨 → YOLO segmentation 형식 변환 + 데이터셋 분할
실행: python3 utils/prepare_yolo.py
"""
import json
import shutil
from pathlib import Path
import random

IMAGE_DIR = Path("/Users/eunseoklee/Downloads/New_Sample/원천데이터/TS_damage/damage")
LABEL_DIR = Path("/Users/eunseoklee/Downloads/New_Sample/라벨링데이터/TL_damage/damage")
OUT_DIR   = Path(__file__).parent.parent / "data" / "yolo_dataset"

CLASS_MAP = {
    "Scratched": 0,
    "Breakage":  1,
    "Separated": 2,
    "Crushed":   3,
}

random.seed(42)
VAL_RATIO = 0.2


def convert_label(json_path: Path, out_txt: Path, img_w: int, img_h: int):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    lines = []

    for ann in data.get("annotations", []):
        cls_id = CLASS_MAP.get(ann.get("damage"), None)
        if cls_id is None:
            continue

        seg = ann.get("segmentation", [])
        if not seg or not seg[0] or not seg[0][0]:
            continue

        points = seg[0][0]  # [[[x,y],...]] → [[x,y],...]
        if len(points) < 3:
            continue

        coords = []
        for pt in points:
            xn = max(0.0, min(1.0, pt[0] / img_w))
            yn = max(0.0, min(1.0, pt[1] / img_h))
            coords.extend([f"{xn:.6f}", f"{yn:.6f}"])

        lines.append(f"{cls_id} " + " ".join(coords))

    out_txt.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def main():
    label_files = sorted(LABEL_DIR.glob("*.json"))
    print(f"라벨 파일: {len(label_files):,}개")

    pairs = []
    for lf in label_files:
        stem = lf.stem
        img  = IMAGE_DIR / f"{stem}.jpg"
        if img.exists():
            pairs.append((img, lf))

    print(f"유효 쌍: {len(pairs):,}개")

    random.shuffle(pairs)
    val_n = int(len(pairs) * VAL_RATIO)
    val   = pairs[:val_n]
    train = pairs[val_n:]
    print(f"train: {len(train):,}  val: {len(val):,}")

    for split, subset in [("train", train), ("val", val)]:
        img_out = OUT_DIR / "images" / split
        lbl_out = OUT_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        ok = 0
        for img_path, lbl_path in subset:
            data     = json.loads(lbl_path.read_text(encoding="utf-8"))
            img_info = data.get("images", {})
            w = img_info.get("width",  640)
            h = img_info.get("height", 640)

            out_txt = lbl_out / f"{img_path.stem}.txt"
            n = convert_label(lbl_path, out_txt, w, h)
            if n > 0:
                shutil.copy2(img_path, img_out / img_path.name)
                ok += 1

        print(f"  [{split}] 완료: {ok:,}개")

    print(f"\n데이터셋 경로: {OUT_DIR}")


if __name__ == "__main__":
    main()
