# config.py
# アプリ全体の設定を一元管理する

from pathlib import Path

# -------------------------
# パス設定
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"

# -------------------------
# YOLO設定
# -------------------------
# COCO想定: car=2, motorcycle=3, bus=5, truck=7
# 複数クラスを検出対象に
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # 車、バイク、バス、トラック
CAR_CLASS_ID = 2  # 後方互換性のため維持
CONF_TH = 0.20  # 検出感度を最大に

# -------------------------
# デバイス設定
# -------------------------
USE_GPU = True  # False にすると強制CPU

# -------------------------
# 描画設定
# -------------------------
BBOX_COLOR = (0, 255, 0)   # Green
BBOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# -------------------------
# UI設定
# -------------------------
WINDOW_NAME = "Vehicle Player"
SEEKBAR_HEIGHT = 40  # シークバー領域の高さ
BUTTON_AREA_HEIGHT = 40  # ボタン領域の高さ
