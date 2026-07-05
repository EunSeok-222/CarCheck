"""
YOLOv8n-seg 학습 스크립트 (M1 Pro MPS 최적화)
실행: python3 utils/train_yolo.py
"""
from pathlib import Path
import shutil
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).parent.parent
YAML_PATH   = PROJECT_DIR / "data" / "car_damage.yaml"
MODEL_DIR   = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

model = YOLO("yolov8n-seg.pt")

results = model.train(
    data=str(YAML_PATH),
    epochs=100,
    imgsz=640,
    batch=8,
    device="mps",
    workers=4,
    project=str(MODEL_DIR),
    name="car_damage_seg",
    exist_ok=True,
    patience=20,
    save=True,
    save_period=10,
    plots=True,
    verbose=True,
)

best = MODEL_DIR / "car_damage_seg" / "weights" / "best.pt"
if best.exists():
    shutil.copy2(best, MODEL_DIR / "best.pt")
    print(f"\n모델 저장 완료: {MODEL_DIR / 'best.pt'}")

print(f"mAP50: {results.results_dict.get('metrics/mAP50(M)', 'N/A')}")
