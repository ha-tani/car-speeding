# overlay_renderer.py
# 描画専用モジュール（ロジック禁止）

import cv2
import config


class OverlayRenderer:
    def draw(self, frame, tracked_cars, speeds,
             show_car_bbox=True, show_plate_bbox=False, plate_bboxes=None):
        """
        tracked_cars:
        [
          { "track_id": int, "bbox": (...) }
        ]

        speeds:
        { track_id: km/h }

        show_car_bbox: 車BBOXの表示ON/OFF
        show_plate_bbox: ナンバープレートBBOXの表示ON/OFF
        plate_bboxes: { track_id: (x1, y1, x2, y2) } フレーム座標
        """
        if plate_bboxes is None:
            plate_bboxes = {}

        for car in tracked_cars:
            tid = car["track_id"]
            x1, y1, x2, y2 = car["bbox"]
            speed = speeds.get(tid, 0.0)
            
            # 速度超過判定：閾値を超えたら赤色、それ以外は通常色
            is_warning = speed > config.SPEED_THRESHOLD_KMH
            if is_warning:
                bbox_color = config.SPEED_WARNING_COLOR  # 赤色
                text_color = config.SPEED_WARNING_COLOR
                plate_color = config.SPEED_WARNING_COLOR  # 赤色
            else:
                bbox_color = config.SPEED_NORMAL_COLOR   # 緑色
                text_color = config.SPEED_NORMAL_COLOR
                plate_color = config.PLATE_BBOX_NORMAL_COLOR  # 水色

            # 車BBox描画
            if show_car_bbox:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    bbox_color,
                    config.BBOX_THICKNESS
                )

                # 速度のみ表示（IDは非表示）
                label = f"{speed:.1f} km/h"
                
                # 背景を追加して読みやすくする
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           config.FONT_SCALE, config.FONT_THICKNESS)[0]
                
                # テキスト背景（半透明の黒）
                bg_x1 = x1
                bg_y1 = y1 - text_size[1] - 10
                bg_x2 = x1 + text_size[0] + 10
                bg_y2 = y1 - 2
                
                # 背景矩形
                overlay = frame.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # テキスト描画
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.FONT_SCALE,
                    text_color,
                    config.FONT_THICKNESS,
                    cv2.LINE_AA
                )

            # ナンバープレートBBox描画
            if show_plate_bbox and tid in plate_bboxes:
                px1, py1, px2, py2 = plate_bboxes[tid]
                cv2.rectangle(
                    frame,
                    (px1, py1),
                    (px2, py2),
                    plate_color,
                    config.PLATE_BBOX_THICKNESS
                )

        return frame
