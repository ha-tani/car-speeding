# calibration.py
# 映像キャリブレーション管理 ─ 4点ホモグラフィ＋進行方向

import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog


class CalibrationManager:
    """
    道路平面上の4点（四角形）と実寸法からホモグラフィ変換を行い、
    画像上のピクセル移動を実世界メートルに変換する。

    トグル状態:
      STATE_NONE   (0): グレー  ─ 未設定
      STATE_ACTIVE (1): 青/オレンジ ─ クリック待ち
      STATE_HALF   (2): 青/オレンジ ─ 途中まで入力済み
      STATE_DONE   (3): 緑     ─ 設定完了
    """

    STATE_NONE = 0
    STATE_ACTIVE = 1
    STATE_HALF = 2
    STATE_DONE = 3

    def __init__(self):
        self.reset()

    def reset(self):
        """全リセット"""
        # --- 4点キャリブレーション ---
        self.quad_points = []           # 画像上の4点 [(x,y), ...]
        self.quad_state = self.STATE_NONE
        self.real_width_m = 3.5         # 四角形の幅 [m]（デフォルト: 車線幅）
        self.real_height_m = 10.0       # 四角形の奥行き [m]

        # --- 進行方向 (2クリック) ---
        self.dir_point1 = None
        self.dir_point2 = None
        self.dir_state = self.STATE_NONE

        # --- ホモグラフィ行列 ---
        self.homography = None          # 画像 → 実世界
        self.homography_inv = None      # 実世界 → 画像

        # --- 入力モード ---
        self.active_toggle = None       # "quad" | "direction" | None

    # ================================================================
    # トグル活性化
    # ================================================================
    def activate_quad(self):
        """4点指定モードを開始 (リセットして最初から)"""
        self.quad_points = []
        self.quad_state = self.STATE_ACTIVE
        self.active_toggle = "quad"
        self.homography = None
        self.homography_inv = None

    def activate_dir(self):
        """進行方向モードを開始"""
        self.dir_point1 = None
        self.dir_point2 = None
        self.dir_state = self.STATE_ACTIVE
        self.active_toggle = "direction"

    def cancel_active(self):
        """ESCキーで現在の入力モードをキャンセル"""
        if self.active_toggle == "quad":
            self.quad_points = []
            self.quad_state = self.STATE_NONE
            self.homography = None
            self.homography_inv = None
        elif self.active_toggle == "direction":
            self.dir_point1 = None
            self.dir_point2 = None
            self.dir_state = self.STATE_NONE
        self.active_toggle = None

    # ================================================================
    # クリック処理
    # ================================================================
    def handle_click(self, x, y):
        """映像上のクリック。消費したら True を返す"""
        if self.active_toggle == "quad":
            return self._handle_quad_click(x, y)
        elif self.active_toggle == "direction":
            return self._handle_dir_click(x, y)
        return False

    def _handle_quad_click(self, x, y):
        self.quad_points.append((x, y))
        n = len(self.quad_points)
        if n < 4:
            self.quad_state = self.STATE_HALF
        else:
            self.quad_state = self.STATE_DONE
            self.active_toggle = None
            self._prompt_dimensions()
            self._compute_homography()
        return True

    def _handle_dir_click(self, x, y):
        if self.dir_point1 is None:
            self.dir_point1 = (x, y)
            self.dir_state = self.STATE_HALF
        else:
            self.dir_point2 = (x, y)
            self.dir_state = self.STATE_DONE
            self.active_toggle = None
        return True

    # ================================================================
    # ホモグラフィ計算
    # ================================================================
    def _compute_homography(self):
        if len(self.quad_points) < 4:
            return
        img_pts = np.array(self.quad_points[:4], dtype=np.float32)
        w, h = self.real_width_m, self.real_height_m
        world_pts = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float32)
        self.homography, _ = cv2.findHomography(img_pts, world_pts)
        if self.homography is not None:
            self.homography_inv, _ = cv2.findHomography(world_pts, img_pts)

    # ================================================================
    # 座標変換
    # ================================================================
    def pixel_to_world(self, px, py):
        """画像座標 → 実世界座標 (m)"""
        if self.homography is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        w = cv2.perspectiveTransform(pt, self.homography)
        return (float(w[0][0][0]), float(w[0][0][1]))

    def world_distance(self, p1, p2):
        """2つの画像座標間の実世界距離 (m)"""
        w1 = self.pixel_to_world(*p1)
        w2 = self.pixel_to_world(*p2)
        if w1 is None or w2 is None:
            return None
        dx, dy = w2[0] - w1[0], w2[1] - w1[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def get_direction_vector_world(self):
        """進行方向の単位ベクトル (実世界座標系)"""
        if self.dir_point1 is None or self.dir_point2 is None:
            return None
        if self.homography is not None:
            w1 = self.pixel_to_world(*self.dir_point1)
            w2 = self.pixel_to_world(*self.dir_point2)
            if w1 is None or w2 is None:
                return None
            dx, dy = w2[0] - w1[0], w2[1] - w1[1]
        else:
            dx = self.dir_point2[0] - self.dir_point1[0]
            dy = self.dir_point2[1] - self.dir_point1[1]
        mag = np.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            return None
        return (dx / mag, dy / mag)

    def is_calibrated(self):
        return self.homography is not None

    # ================================================================
    # 寸法入力ダイアログ
    # ================================================================
    def _prompt_dimensions(self):
        """4点設定直後に幅・奥行きを入力させる"""
        root = tk.Tk()
        root.withdraw()
        w_str = simpledialog.askstring(
            "四角形の幅",
            f"4点で囲んだ道路四角形の幅 (横方向) [m]:\n(デフォルト: {self.real_width_m:.1f}m)",
            initialvalue=str(self.real_width_m), parent=root)
        if w_str:
            try:
                v = float(w_str)
                if v > 0:
                    self.real_width_m = v
            except ValueError:
                pass
        h_str = simpledialog.askstring(
            "四角形の奥行き",
            f"4点で囲んだ道路四角形の奥行き (進行方向) [m]:\n(デフォルト: {self.real_height_m:.1f}m)",
            initialvalue=str(self.real_height_m), parent=root)
        if h_str:
            try:
                v = float(h_str)
                if v > 0:
                    self.real_height_m = v
            except ValueError:
                pass
        root.destroy()

    def prompt_edit_width(self):
        root = tk.Tk()
        root.withdraw()
        s = simpledialog.askstring(
            "幅 [m]", f"四角形の幅 [m] (現在: {self.real_width_m:.1f})",
            initialvalue=str(self.real_width_m), parent=root)
        root.destroy()
        if s:
            try:
                v = float(s)
                if v > 0:
                    self.real_width_m = v
                    self._compute_homography()
            except ValueError:
                pass

    def prompt_edit_height(self):
        root = tk.Tk()
        root.withdraw()
        s = simpledialog.askstring(
            "奥行き [m]", f"四角形の奥行き [m] (現在: {self.real_height_m:.1f})",
            initialvalue=str(self.real_height_m), parent=root)
        root.destroy()
        if s:
            try:
                v = float(s)
                if v > 0:
                    self.real_height_m = v
                    self._compute_homography()
            except ValueError:
                pass

    # ================================================================
    # オーバーレイ描画
    # ================================================================
    def draw_overlays(self, frame):
        n = len(self.quad_points)
        done_color = (0, 200, 0)
        prog_color = (0, 165, 255)
        labels = ["Q1(TL)", "Q2(TR)", "Q3(BR)", "Q4(BL)"]

        for i, pt in enumerate(self.quad_points):
            c = done_color if self.quad_state == self.STATE_DONE else prog_color
            cv2.circle(frame, pt, 6, c, -1)
            cv2.circle(frame, pt, 6, (255, 255, 255), 1)
            cv2.putText(frame, labels[i], (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)

        if n >= 2:
            for i in range(min(n, 4)):
                if i + 1 < n:
                    c = done_color if n == 4 else prog_color
                    cv2.line(frame, self.quad_points[i], self.quad_points[i + 1], c, 1)
            if n == 4:
                cv2.line(frame, self.quad_points[3], self.quad_points[0], done_color, 1)
                # 半透明フィル
                overlay = frame.copy()
                pts = np.array(self.quad_points[:4], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 200, 0))
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                # 寸法表示
                mid_top = (
                    (self.quad_points[0][0] + self.quad_points[1][0]) // 2,
                    (self.quad_points[0][1] + self.quad_points[1][1]) // 2,
                )
                mid_left = (
                    (self.quad_points[0][0] + self.quad_points[3][0]) // 2,
                    (self.quad_points[0][1] + self.quad_points[3][1]) // 2,
                )
                cv2.putText(frame, f"W:{self.real_width_m:.1f}m",
                            (mid_top[0] - 20, mid_top[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, done_color, 1, cv2.LINE_AA)
                cv2.putText(frame, f"D:{self.real_height_m:.1f}m",
                            (mid_left[0] - 50, mid_left[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, done_color, 1, cv2.LINE_AA)

        # 進行方向矢印（マゼンタ）
        dir_c = (255, 0, 255)
        if self.dir_point1 is not None:
            cv2.circle(frame, self.dir_point1, 6, dir_c, -1)
            cv2.circle(frame, self.dir_point1, 6, (255, 255, 255), 1)
            cv2.putText(frame, "D1", (self.dir_point1[0] + 8, self.dir_point1[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, dir_c, 1, cv2.LINE_AA)
        if self.dir_point2 is not None:
            cv2.circle(frame, self.dir_point2, 6, dir_c, -1)
            cv2.circle(frame, self.dir_point2, 6, (255, 255, 255), 1)
            cv2.putText(frame, "D2", (self.dir_point2[0] + 8, self.dir_point2[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, dir_c, 1, cv2.LINE_AA)
            if self.dir_point1 is not None:
                cv2.arrowedLine(frame, self.dir_point1, self.dir_point2, dir_c, 2, tipLength=0.15)

        return frame
