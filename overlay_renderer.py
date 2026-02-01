# overlay_renderer.py
# 描画専用モジュール（ロジック禁止）

import cv2
import config


class OverlayRenderer:
    def draw(self, frame, tracked_cars, speeds):
        """
        tracked_cars:
        [
          { "track_id": int, "bbox": (...) }
        ]

        speeds:
        { track_id: km/h }
        """

        for car in tracked_cars:
            tid = car["track_id"]
            x1, y1, x2, y2 = car["bbox"]
            speed = speeds.get(tid, 0.0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                config.BBOX_COLOR,
                config.BBOX_THICKNESS
            )

            label = f"ID:{tid} {speed:.1f} km/h"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.BBOX_COLOR,
                config.FONT_THICKNESS,
                cv2.LINE_AA
            )

        return frame
