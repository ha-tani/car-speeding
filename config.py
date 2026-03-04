# config.py
# アプリ全体の設定を一元管理する

from pathlib import Path

# -------------------------
# パス設定
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

YOLO_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
YOLO_PLATE_MODEL_PATH = MODELS_DIR / "yolov8n-np.pt"  # ナンバープレート検出専用モデル
FAST_PLATE_OCR_MODEL_DIR = MODELS_DIR / "fast-plate-ocr"  # fast-plate-ocrモデル保存先

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

# 速度超過警告設定
SPEED_THRESHOLD_KMH = 60  # この速度を超えると赤色警告表示
SPEED_WARNING_COLOR = (0, 0, 255)  # Red (赤色)
SPEED_NORMAL_COLOR = (0, 255, 0)   # Green (緑色)

# ナンバープレートBBOX設定
PLATE_BBOX_NORMAL_COLOR = (255, 255, 0)   # Cyan (水色 - BGR)
PLATE_BBOX_THICKNESS = 2

# -------------------------
# UI設定
# -------------------------
WINDOW_NAME = "Vehicle Player"
SEEKBAR_HEIGHT = 40  # シークバー領域の高さ
BUTTON_AREA_HEIGHT = 40  # ボタン領域の高さ

# 表示サイズ制限（画面に収まるよう自動縮小）
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

# -------------------------
# 速度推定設定
# -------------------------
SPEED_SMOOTHING_WINDOW = 15   # 速度スムージングのフレーム数
SPEED_HISTORY_SIZE = 20       # 位置履歴の保持フレーム数
SPEED_MIN_THRESHOLD_KMH = 3.0 # この速度以下は停車とみなす [km/h]
SPEED_MIN_PIXEL_MOVEMENT = 3.0 # このピクセル以下の移動はノイズとみなす
