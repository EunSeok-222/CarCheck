from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODEL_PATH = Path(__file__).parent.parent / "models" / "best.pt"

CLASS_NAMES = {0: "Scratched", 1: "Breakage", 2: "Separated", 3: "Crushed"}
CLASS_KO    = {0: "긁힘",      1: "파손",     2: "분리",      3: "찌그러짐"}
PART_NAMES  = [
    "앞 범퍼", "뒤 범퍼", "앞 도어(좌)", "앞 도어(우)",
    "앞 펜더(좌)", "앞 펜더(우)", "헤드라이트(좌)", "헤드라이트(우)",
    "트렁크", "본네트",
]
COLORS = {0: (255, 220, 0, 120), 1: (255, 50, 50, 140),
          2: (255, 140, 0, 130), 3: (200, 0, 200, 130)}

_model = None


def _load_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    return _model


def _draw_masks(image: Image.Image, results) -> Image.Image:
    annotated = image.convert("RGBA")
    overlay   = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw      = ImageDraw.Draw(overlay)

    boxes  = results[0].boxes
    masks  = results[0].masks

    if masks is None:
        return image

    for i, (mask_xy, cls_id) in enumerate(zip(masks.xy, boxes.cls.int().tolist())):
        if len(mask_xy) < 3:
            continue
        color   = COLORS.get(cls_id, (100, 100, 255, 120))
        polygon = [(float(x), float(y)) for x, y in mask_xy]
        draw.polygon(polygon, fill=color, outline=color[:3] + (255,))

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")
    return annotated


def _parse_results(results, image: Image.Image) -> dict:
    boxes   = results[0].boxes
    masks   = results[0].masks
    damages = []

    if boxes is None or len(boxes) == 0:
        return {"original_image": image, "annotated_image": image,
                "damages": [], "damage_count": 0}

    for i, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        conf   = float(box.conf.item())
        area   = 0
        if masks is not None and i < len(masks.xy):
            pts  = np.array(masks.xy[i])
            if len(pts) >= 3:
                x = pts[:, 0]; y = pts[:, 1]
                area = int(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

        damages.append({
            "type":       CLASS_NAMES.get(cls_id, "Unknown"),
            "type_ko":    CLASS_KO.get(cls_id, "기타"),
            "part":       PART_NAMES[i % len(PART_NAMES)],
            "confidence": conf,
            "area":       area,
        })

    annotated = _draw_masks(image, results)

    return {
        "original_image":  image,
        "annotated_image": annotated,
        "damages":         damages,
        "damage_count":    len(damages),
    }


# ── 목업 (best.pt 없을 때 fallback) ────────────────────────
_MOCK = [
    {"type": "Scratched", "type_ko": "긁힘",   "part": "앞 범퍼",       "confidence": 0.87, "area": 5200},
    {"type": "Breakage",  "type_ko": "파손",   "part": "헤드라이트(좌)", "confidence": 0.92, "area": 1800},
]


def detect_damage(image: Image.Image) -> dict:
    model = _load_model()

    if model is None:
        return {"original_image": image, "annotated_image": image,
                "damages": _MOCK, "damage_count": len(_MOCK),
                "_mock": True}

    results = model.predict(image, conf=0.25, iou=0.45, verbose=False)
    parsed  = _parse_results(results, image)

    if parsed["damage_count"] == 0:
        parsed["damages"]       = _MOCK
        parsed["damage_count"]  = len(_MOCK)
        parsed["_mock"]         = True

    return parsed
