# detector_yolo.py
# YOLOv8による車検出専用モジュール

import torch
from ultralytics import YOLO
import numpy as np

import config


class YoloDetector:
    def __init__(self):
        self.device = self._select_device()
        self.model = YOLO(str(config.YOLO_MODEL_PATH))
        self.model.to(self.device)

    def _select_device(self):
        if config.USE_GPU and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def detect_cars(self, frame):
        """
        frame: BGR ndarray (OpenCV)
        return: list of dict
        [
          {
            "bbox": (x1, y1, x2, y2),
            "conf": float
          }
        ]
        """
        results = self.model(
            frame,
            conf=config.CONF_TH,
            device=self.device,
            verbose=False,
            iou=0.5,  # NMS IoU閾値
            agnostic_nms=True  # クラス無視NMS
        )

        cars = []

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, clss):
                # 複数の車両クラスを検出
                if cls_id not in config.VEHICLE_CLASS_IDS:
                    continue

                x1, y1, x2, y2 = map(int, box)
                
                # 最小サイズフィルタ（小さすぎる検出を除外）
                if (x2 - x1) < 30 or (y2 - y1) < 30:
                    continue
                
                cars.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(conf),
                    "class_id": cls_id
                })

        return cars

    def estimate_plate_region(self, car_bbox, frame_shape):
        """
        車のBBoxからナンバープレートの推定領域を返す。
        日本車の場合、ナンバープレートは通常車両の下部中央にある。
        
        car_bbox: (x1, y1, x2, y2)
        frame_shape: (height, width, channels)
        return: (x1, y1, x2, y2) or None
        """
        x1, y1, x2, y2 = car_bbox
        car_w = x2 - x1
        car_h = y2 - y1
        frame_h, frame_w = frame_shape[:2]
        
        # ナンバープレートは車の下部35%、水平方向は中央70%と推定（広めに取る）
        plate_x1 = int(x1 + car_w * 0.15)
        plate_x2 = int(x1 + car_w * 0.85)
        plate_y1 = int(y1 + car_h * 0.60)
        plate_y2 = int(y2 + car_h * 0.05)  # 少し下にはみ出させる
        
        # フレーム境界でクリップ
        plate_x1 = max(0, plate_x1)
        plate_y1 = max(0, plate_y1)
        plate_x2 = min(frame_w, plate_x2)
        plate_y2 = min(frame_h, plate_y2)
        
        # 領域が小さすぎる場合はNone
        if plate_x2 - plate_x1 < 20 or plate_y2 - plate_y1 < 10:
            return None
        
        return (plate_x1, plate_y1, plate_x2, plate_y2)

    def detect_cars_with_plates(self, frame):
        """
        車とナンバープレート推定領域を同時に返す
        return: list of dict
        [
          {
            "bbox": (x1, y1, x2, y2),
            "conf": float,
            "plate_bbox": (x1, y1, x2, y2) or None
          }
        ]
        """
        cars = self.detect_cars(frame)
        
        for car in cars:
            plate_region = self.estimate_plate_region(car["bbox"], frame.shape)
            car["plate_bbox"] = plate_region
        
        return cars
