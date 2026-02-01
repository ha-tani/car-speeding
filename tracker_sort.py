# tracker_sort.py
# SORTによる車追跡専用モジュール

import numpy as np

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

        return results
