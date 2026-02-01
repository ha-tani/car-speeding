# speed_estimator.py
# 車速算出専用モジュール

import math
from collections import deque
import time

import config


class SpeedEstimator:
    def __init__(self, history_size=10, smoothing_window=5):
        self.history = {}           # 位置履歴
        self.speed_history = {}     # 速度履歴（スムージング用）
        self.history_size = history_size
        self.smoothing_window = smoothing_window
        
        # 停止判定しきい値
        self.min_speed_threshold = 3.0  # km/h以下は停止とみなす
        self.min_pixel_movement = 3.0   # ピクセル以下の移動はノイズとみなす

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _bbox_size(self, bbox):
        """ピクセルサイズを返す（スケール補正用）"""
        x1, y1, x2, y2 = bbox
        return math.hypot(x2 - x1, y2 - y1)

    def update(self, tracked_cars):
        """
        tracked_cars:
        [
          { "track_id": int, "bbox": (...) }
        ]

        return:
        {
          track_id: speed_kmh (float)
        }
        """

        now = time.time()
        speeds = {}

        for car in tracked_cars:
            tid = car["track_id"]
            center = self._center(car["bbox"])
            bbox_size = self._bbox_size(car["bbox"])

            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.history_size)
                self.speed_history[tid] = deque(maxlen=self.smoothing_window)

            self.history[tid].append((center, now, bbox_size))

            if len(self.history[tid]) < 2:
                speeds[tid] = 0.0
                continue

            # 複数フレーム間の平均速度を計算（ジッター軽減）
            total_dist = 0.0
            total_time = 0.0
            
            history_list = list(self.history[tid])
            for i in range(1, len(history_list)):
                (x1, y1), t1, size1 = history_list[i - 1]
                (x2, y2), t2, size2 = history_list[i]
                
                dt = t2 - t1
                if dt <= 0:
                    continue
                
                pixel_dist = math.hypot(x2 - x1, y2 - y1)
                total_dist += pixel_dist
                total_time += dt
            
            if total_time <= 0:
                speeds[tid] = 0.0
                continue
            
            # 平均ピクセル速度
            avg_pixel_speed = total_dist / total_time
            
            # ノイズフィルタ: 1フレームあたりの移動量が小さすぎる場合は停止とみなす
            avg_frame_movement = total_dist / (len(history_list) - 1) if len(history_list) > 1 else 0
            if avg_frame_movement < self.min_pixel_movement:
                raw_kmh = 0.0
            else:
                # 仮スケール（1px = 0.05m）
                meters_per_sec = avg_pixel_speed * 0.05
                raw_kmh = meters_per_sec * 3.6
            
            # 低速度しきい値（3km/h以下は0とする）
            if raw_kmh < self.min_speed_threshold:
                raw_kmh = 0.0
            
            # スムージング: 速度履歴の移動平均
            self.speed_history[tid].append(raw_kmh)
            smoothed_kmh = sum(self.speed_history[tid]) / len(self.speed_history[tid])
            
            speeds[tid] = smoothed_kmh

        return speeds
