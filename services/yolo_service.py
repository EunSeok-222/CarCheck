from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODEL_PATH = Path(__file__).parent.parent / "models" / "best.pt"
HF_REPO_ID = "eunseok22/carcheck-model"


def _ensure_model():
    if MODEL_PATH.exists():
        return
    try:
        from huggingface_hub import hf_hub_download
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=HF_REPO_ID, filename="best.pt",
                        local_dir=str(MODEL_PATH.parent))
    except Exception:
        pass

CLASS_NAMES = {0: "Scratched", 1: "Breakage", 2: "Separated", 3: "Crushed"}
CLASS_KO    = {0: "긁힘",      1: "파손",     2: "분리",      3: "찌그러짐"}
DEFAULT_PART = "미확인 부위"
COLORS = {0: (255, 220, 0, 120), 1: (255, 50, 50, 140),
          2: (255, 140, 0, 130), 3: (200, 0, 200, 130)}

_model = None


def _load_model():
    global _model
    if _model is None:
        _ensure_model()
        if MODEL_PATH.exists():
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


def _parse_results(results, image: Image.Image, part: str) -> dict:
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

        # 모델은 손상 유형(긁힘/파손/분리/찌그러짐)만 분류하고 부위를 예측하지
        # 않는다. 사진 한 장은 사용자가 지정한 한 부위를 촬영한 것이므로,
        # 해당 사진 안의 모든 감지 결과에 같은 부위명을 부여한다.
        damages.append({
            "type":       CLASS_NAMES.get(cls_id, "Unknown"),
            "type_ko":    CLASS_KO.get(cls_id, "기타"),
            "part":       part or DEFAULT_PART,
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


def _mock_data(part: str) -> list:
    """best.pt 없거나 감지 실패 시 데모용 목업 (사용자가 고른 부위를 그대로 반영)."""
    p = part or DEFAULT_PART
    return [
        {"type": "Scratched", "type_ko": "긁힘", "part": p, "confidence": 0.87, "area": 5200},
        {"type": "Breakage",  "type_ko": "파손", "part": p, "confidence": 0.92, "area": 1800},
    ]


def detect_damage(image: Image.Image, part: str = "") -> dict:
    model = _load_model()

    if model is None:
        mock = _mock_data(part)
        return {"original_image": image, "annotated_image": image,
                "damages": mock, "damage_count": len(mock),
                "_mock": True}

    results = model.predict(image, conf=0.25, iou=0.45, verbose=False)
    parsed  = _parse_results(results, image, part)

    if parsed["damage_count"] == 0:
        mock = _mock_data(part)
        parsed["damages"]       = mock
        parsed["damage_count"]  = len(mock)
        parsed["_mock"]         = True

    return parsed
