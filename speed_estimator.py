# speed_estimator.py
# 車速算出専用モジュール ─ 4点ホモグラフィ対応

import math
import numpy as np
from collections import deque

import config


class SpeedEstimator:
    def __init__(self):
        self.history = {}           # tid -> deque of (center, timestamp, bbox_size)
        self.speed_history = {}     # tid -> deque of raw_kmh
        self.history_size = config.SPEED_HISTORY_SIZE
        self.smoothing_window = config.SPEED_SMOOTHING_WINDOW
        self.calibration = None

        # 停止判定しきい値
        self.min_speed_threshold = config.SPEED_MIN_THRESHOLD_KMH
        self.min_pixel_movement = config.SPEED_MIN_PIXEL_MOVEMENT

        # シーク検知用
        self._last_time = None

    def set_calibration(self, calibration):
        self.calibration = calibration

    def _bottom_center(self, bbox):
        """BBoxの底辺中央を返す（タイヤ接地点に近く、道路平面上で安定）"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, float(y2))

    def _bbox_size(self, bbox):
        x1, y1, x2, y2 = bbox
        return math.hypot(x2 - x1, y2 - y1)

    def clear_history(self):
        """シーク時などに全履歴をクリア（速度爆発防止）"""
        self.history.clear()
        self.speed_history.clear()
        self._last_time = None

    def update(self, tracked_cars, video_time=None):
        """
        tracked_cars: トラッキング結果リスト
        video_time: 動画上の時刻 [秒] (frame_number / fps)
                    ※ time.time() ではなく動画時間を使い、
                      処理遅延による速度誤差を除去する
        """
        if video_time is None:
            # フォールバック（呼び出し元が対応していない場合）
            import time
            now = time.time()
        else:
            now = video_time
        speeds = {}

        # シーク検知: 動画時間が逆行 or 大きく飛んだ場合（>1秒）履歴をクリア
        if self._last_time is not None:
            tdiff = now - self._last_time
            if tdiff < -0.01 or tdiff > 1.0:
                self.history.clear()
                self.speed_history.clear()
        self._last_time = now

        # キャリブレーション状態
        cal = self.calibration
        use_homography = (cal is not None and cal.is_calibrated())
        dir_vec = None
        if use_homography:
            dir_vec = cal.get_direction_vector_world()

        active_ids = set()

        for car in tracked_cars:
            tid = car["track_id"]
            active_ids.add(tid)
            center = self._bottom_center(car["bbox"])
            bbox_size = self._bbox_size(car["bbox"])

            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.history_size)
                self.speed_history[tid] = deque(maxlen=self.smoothing_window)

            self.history[tid].append((center, now, bbox_size))

            if len(self.history[tid]) < 2:
                speeds[tid] = 0.0
                continue

            # 複数フレーム間の平均速度を計算
            total_dist = 0.0
            total_time = 0.0

            history_list = list(self.history[tid])
            for i in range(1, len(history_list)):
                (px1, py1), t1, sz1 = history_list[i - 1]
                (px2, py2), t2, sz2 = history_list[i]

                dt = t2 - t1
                if dt <= 0 or dt > 0.5:
                    # 時間がおかしいフレームペアはスキップ
                    continue

                pixel_dist = math.hypot(px2 - px1, py2 - py1)

                if use_homography:
                    w1 = cal.pixel_to_world(px1, py1)
                    w2 = cal.pixel_to_world(px2, py2)
                    if w1 is not None and w2 is not None:
                        dx_w, dy_w = w2[0] - w1[0], w2[1] - w1[1]
                        if dir_vec is not None:
                            dist_m = abs(dx_w * dir_vec[0] + dy_w * dir_vec[1])
                        else:
                            dist_m = math.sqrt(dx_w * dx_w + dy_w * dy_w)
                        total_dist += dist_m
                        total_time += dt
                        continue
                    # ホモグラフィ変換失敗 → ピクセルフォールバック

                # フォールバック: ピクセル距離
                total_dist += pixel_dist * 0.05  # 仮スケール 1px=0.05m
                total_time += dt

            if total_time <= 0:
                speeds[tid] = 0.0
                continue

            avg_speed_mps = total_dist / total_time
            raw_kmh = avg_speed_mps * 3.6

            # ノイズフィルタ: ピクセル移動が小さすぎる場合は停車とみなす
            if not use_homography:
                avg_frame_px = sum(
                    math.hypot(
                        history_list[i][0][0] - history_list[i-1][0][0],
                        history_list[i][0][1] - history_list[i-1][0][1],
                    )
                    for i in range(1, len(history_list))
                ) / max(len(history_list) - 1, 1)
                if avg_frame_px < self.min_pixel_movement:
                    raw_kmh = 0.0

            if raw_kmh < self.min_speed_threshold:
                raw_kmh = 0.0

            # スムージング
            self.speed_history[tid].append(raw_kmh)
            smoothed = sum(self.speed_history[tid]) / len(self.speed_history[tid])
            speeds[tid] = smoothed

        # 消えたトラックのクリーンアップ（メモリリーク防止）
        lost_ids = [tid for tid in self.history if tid not in active_ids]
        for tid in lost_ids:
            del self.history[tid]
            self.speed_history.pop(tid, None)

        return speeds
