# tracker_sort.py
# SORTによる車追跡専用モジュール

import math
import numpy as np
from collections import deque

import config

try:
    from sort import Sort
except ImportError:
    raise ImportError(
        "SORTが見つかりません。sort.py を同フォルダに置くか、"
        "https://github.com/abewley/sort を配置してください。"
    )


class SortTracker:
    def __init__(self):
        self.tracker = Sort()
        self._position_history = {}  # track_id -> deque of (cx, cy)

    def is_moving(self, track_id):
        """True: 動いている / False: 静止している（または履歴不足）"""
        hist = self._position_history.get(track_id)
        if hist is None or len(hist) < 2:
            # 履歴が十分でない間は動いているとみなし表示し続ける
            return True
        dist = math.hypot(hist[-1][0] - hist[0][0], hist[-1][1] - hist[0][1])
        return dist >= config.STATIONARY_MIN_DISPLACEMENT

    def update(self, cars):
        """
        cars: detector_yolo の出力
        [
          { "bbox": (x1,y1,x2,y2), "conf": float }
        ]

        return:
        [
          {
            "track_id": int,
            "bbox": (x1,y1,x2,y2)
          }
        ]
        """

        if not cars:
            self.tracker.update(np.empty((0, 5)))
            return []

        dets = []
        for car in cars:
            x1, y1, x2, y2 = car["bbox"]
            conf = car["conf"]
            dets.append([x1, y1, x2, y2, conf])

        dets = np.array(dets)

        tracks = self.tracker.update(dets)

        results = []
        for x1, y1, x2, y2, track_id in tracks:
            results.append({
                "track_id": int(track_id),
                "bbox": (int(x1), int(y1), int(x2), int(y2))
            })

        # 位置履歴を更新
        active_ids = set()
        for item in results:
            tid = item["track_id"]
            active_ids.add(tid)
            x1, y1, x2, y2 = item["bbox"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if tid not in self._position_history:
                self._position_history[tid] = deque(maxlen=config.STATIONARY_FILTER_FRAMES)
            self._position_history[tid].append((cx, cy))

        # 消滅したトラックの履歴を削除
        for tid in list(self._position_history.keys()):
            if tid not in active_ids:
                del self._position_history[tid]

        return results
