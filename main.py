# main.py
# 起動時にUIを表示し、映像選択後に解析開始する版

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

import config
from video_player import VideoPlayer
from detector_yolo import YoloDetector
from tracker_sort import SortTracker
from speed_estimator import SpeedEstimator
from plate_ocr import PlateOCR
from overlay_renderer import OverlayRenderer
from click_selector import select_track_id
from plate_dialog import PlateDialog
from calibration import CalibrationManager


# -------------------------
# OpenCV マウスイベント
# -------------------------
class MouseHandler:
    def __init__(self):
        self.clicked_pos = None
        self.seekbar_dragging = False
        self.seekbar_y = 0
        self.button_area_y = 0
        self.request_file_select = False  # 映像選択リクエストフラグ
        self.toggle_car_bbox = False     # 車BBOXトグルリクエスト
        self.toggle_plate_bbox = False   # プレートBBOXトグルリクエスト
        self.toggle_quad = False         # 4点キャリブレーショントグル
        self.toggle_direction = False    # 進行方向トグルリクエスト
        self.edit_width = False          # 幅編集リクエスト
        self.edit_height = False         # 奥行き編集リクエスト
        self._seek_happened = False      # シークが発生したフラグ        
        self.cal_drag_active = False     # キャリブレーションポイントのドラッグ中
        self.calibration = None          # CalibrationManagerへの参照
    def callback(self, event, x, y, flags, param):
        player = param
        if event == cv2.EVENT_LBUTTONDOWN:
            # チェックボックスクリックの判定
            if button_manager.is_car_checkbox_clicked(x, y):
                self.toggle_car_bbox = True
            elif button_manager.is_plate_checkbox_clicked(x, y):
                self.toggle_plate_bbox = True
            # キャリブレーショントグルの判定
            elif button_manager.is_quad_toggle_clicked(x, y):
                self.toggle_quad = True
            elif button_manager.is_dir_toggle_clicked(x, y):
                self.toggle_direction = True
            elif button_manager.is_width_textbox_clicked(x, y):
                self.edit_width = True
            elif button_manager.is_height_textbox_clicked(x, y):
                self.edit_height = True
            # 映像選択ボタンクリックの判定
            elif button_manager.is_file_button_clicked(x, y):
                self.request_file_select = True
            # ボタンクリックの判定
            elif button_manager.is_play_button_clicked(x, y):
                player.toggle_pause()
            # シークバー領域のクリックか判定（ボタン領域より上）
            elif y >= self.seekbar_y and y < self.button_area_y:
                self.seekbar_dragging = True
                self._update_seek(x, player)
            elif y < self.seekbar_y:
                # キャリブレーションポイントのドラッグを先に試みる
                if (self.calibration is not None
                        and self.calibration.active_toggle is None
                        and self.calibration.start_drag(x, y)):
                    self.cal_drag_active = True
                else:
                    self.clicked_pos = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cal_drag_active and self.calibration is not None:
                self.calibration.update_drag(x, y)
            elif self.seekbar_dragging:
                self._update_seek(x, player)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.cal_drag_active and self.calibration is not None:
                self.calibration.end_drag()
                self.cal_drag_active = False
            elif self.seekbar_dragging:
                self._update_seek(x, player)
                self.seekbar_dragging = False
    
    def _update_seek(self, x, player):
        if player and player.frame_width > 0:
            ratio = max(0, min(1, x / player.frame_width))
            player.seek_ratio(ratio)
            # シーク時に速度履歴をクリア（速度爆発防止）
            self._seek_happened = True


# ボタン領域を管理するクラス
class ButtonManager:
    def __init__(self):
        self.play_button_rect = None  # (x1, y1, x2, y2)
        self.file_button_rect = None  # 映像選択ボタン
        self.car_checkbox_rect = None   # 車検出チェックボックス
        self.plate_checkbox_rect = None # プレート検出チェックボックス
        self.quad_toggle_rect = None    # 4点キャリブレーショントグル
        self.dir_toggle_rect = None     # 進行方向トグル
        self.width_textbox_rect = None  # 幅テキストボックス
        self.height_textbox_rect = None # 奥行きテキストボックス
    
    def set_play_button_rect(self, rect):
        self.play_button_rect = rect
    
    def set_file_button_rect(self, rect):
        self.file_button_rect = rect
    
    def set_car_checkbox_rect(self, rect):
        self.car_checkbox_rect = rect
    
    def set_plate_checkbox_rect(self, rect):
        self.plate_checkbox_rect = rect
    
    def is_play_button_clicked(self, x, y):
        if self.play_button_rect is None:
            return False
        x1, y1, x2, y2 = self.play_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def is_file_button_clicked(self, x, y):
        if self.file_button_rect is None:
            return False
        x1, y1, x2, y2 = self.file_button_rect
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def is_car_checkbox_clicked(self, x, y):
        if self.car_checkbox_rect is None:
            return False
        x1, y1, x2, y2 = self.car_checkbox_rect
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def is_plate_checkbox_clicked(self, x, y):
        if self.plate_checkbox_rect is None:
            return False
        x1, y1, x2, y2 = self.plate_checkbox_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def set_quad_toggle_rect(self, rect):
        self.quad_toggle_rect = rect

    def set_dir_toggle_rect(self, rect):
        self.dir_toggle_rect = rect

    def set_width_textbox_rect(self, rect):
        self.width_textbox_rect = rect

    def set_height_textbox_rect(self, rect):
        self.height_textbox_rect = rect

    def is_quad_toggle_clicked(self, x, y):
        if self.quad_toggle_rect is None:
            return False
        x1, y1, x2, y2 = self.quad_toggle_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def is_dir_toggle_clicked(self, x, y):
        if self.dir_toggle_rect is None:
            return False
        x1, y1, x2, y2 = self.dir_toggle_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def is_width_textbox_clicked(self, x, y):
        if self.width_textbox_rect is None:
            return False
        x1, y1, x2, y2 = self.width_textbox_rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def is_height_textbox_clicked(self, x, y):
        if self.height_textbox_rect is None:
            return False
        x1, y1, x2, y2 = self.height_textbox_rect
        return x1 <= x <= x2 and y1 <= y <= y2

button_manager = ButtonManager()


def draw_checkbox(frame, x, y, label, checked, size=14):
    """チェックボックスをフレームに描画し、矩形範囲を返す"""
    # ボックス描画
    bx1, by1 = x, y
    bx2, by2 = x + size, y + size
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
    if checked:
        # チェックマーク
        cv2.line(frame, (bx1 + 2, by1 + size // 2), (bx1 + size // 2, by2 - 2), (0, 255, 0), 2)
        cv2.line(frame, (bx1 + size // 2, by2 - 2), (bx2 - 2, by1 + 2), (0, 255, 0), 2)
    else:
        # 空欄（暗い塗りつぶし）
        cv2.rectangle(frame, (bx1 + 1, by1 + 1), (bx2 - 1, by2 - 1), (40, 40, 40), -1)
    # ラベル
    cv2.putText(frame, label, (bx2 + 5, by2 - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    # クリック判定用にラベル含む範囲を返す
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    return (bx1, by1, bx2 + 5 + text_size[0], by2)


def draw_toggle_button(frame, x, y, label, state, progress_text=None, w=72, h=16):
    """キャリブレーション用トグルボタンを描画し、矩形範囲を返す"""
    if state == CalibrationManager.STATE_NONE:
        bg = (80, 80, 80)       # グレー: 未設定
        indicator = ""            # インジケータなし
    elif state in (CalibrationManager.STATE_ACTIVE, CalibrationManager.STATE_HALF):
        bg = (200, 130, 0)      # オレンジ(BGR): クリック待ち
        indicator = "● "          # ● 点滅感
    elif state == CalibrationManager.STATE_DONE:
        bg = (50, 180, 50)      # 緑: 完了
        indicator = "✓ "          # チェックマーク
    else:
        bg = (80, 80, 80)
        indicator = ""

    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (180, 180, 180), 1)

    # ラベル（インジケータ付き）
    disp_label = label
    if progress_text:
        disp_label = progress_text
    text_size = cv2.getTextSize(disp_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    tx = x + (w - text_size[0]) // 2
    ty = y + (h + text_size[1]) // 2
    cv2.putText(frame, disp_label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return (x, y, x + w, y + h)


def draw_text_box(frame, x, y, label, value_text, w=90, h=16):
    """値表示テキストボックスを描画し、矩形範囲を返す（クリックで編集）"""
    bg = (60, 60, 60)
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (160, 160, 160), 1)

    display = f"{label}{value_text}"
    cv2.putText(frame, display, (x + 4, y + h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return (x, y, x + w, y + h)


def draw_seekbar(frame, player, seekbar_height=40, button_area_height=40):
    """フレームの下にシークバーとボタン領域を描画"""
    h, w = frame.shape[:2]
    total_height = seekbar_height + button_area_height
    
    # シークバー背景
    seekbar = np.zeros((total_height, w, 3), dtype=np.uint8)
    seekbar[:] = (40, 40, 40)  # ダークグレー
    
    # ========== シークバー部分 ==========
    # 進捗バー
    progress_ratio = player.get_position_ratio()
    progress_width = int(w * progress_ratio)
    
    # バー背景（薄いグレー）
    bar_y1 = seekbar_height // 2 - 4
    bar_y2 = seekbar_height // 2 + 4
    cv2.rectangle(seekbar, (10, bar_y1), (w - 10, bar_y2), (80, 80, 80), -1)
    
    # 進捗部分（緑）
    if progress_width > 10:
        cv2.rectangle(seekbar, (10, bar_y1), (progress_width, bar_y2), (0, 200, 0), -1)
    
    # 現在位置のノブ
    knob_x = max(10, min(w - 10, progress_width))
    cv2.circle(seekbar, (knob_x, seekbar_height // 2), 8, (255, 255, 255), -1)
    
    # 時間表示
    current_frame = player.get_current_frame()
    total_frames = player.get_total_frames()
    fps = player.get_fps()
    
    current_sec = current_frame / fps if fps > 0 else 0
    total_sec = total_frames / fps if fps > 0 else 0
    
    time_text = f"{int(current_sec // 60):02d}:{int(current_sec % 60):02d} / {int(total_sec // 60):02d}:{int(total_sec % 60):02d}"
    cv2.putText(
        seekbar,
        time_text,
        (w - 150, seekbar_height // 2 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )
    
    # ========== ボタン領域 ==========
    button_y = seekbar_height
    
    # 再生/一時停止ボタン
    btn_w, btn_h = 80, 30
    btn_x1 = w // 2 - btn_w // 2
    btn_y1 = button_y + (button_area_height - btn_h) // 2
    btn_x2 = btn_x1 + btn_w
    btn_y2 = btn_y1 + btn_h
    
    # ボタン領域を保存（フレーム座標系に変換して保存）
    button_manager.set_play_button_rect((btn_x1, h + btn_y1, btn_x2, h + btn_y2))
    
    # ボタン背景
    is_paused = player.is_paused()
    btn_color = (0, 150, 0) if is_paused else (150, 100, 0)  # 緑/オレンジ
    cv2.rectangle(seekbar, (btn_x1, btn_y1), (btn_x2, btn_y2), btn_color, -1)
    cv2.rectangle(seekbar, (btn_x1, btn_y1), (btn_x2, btn_y2), (200, 200, 200), 1)
    
    # ボタンテキスト
    btn_text = "Play" if is_paused else "Pause"
    text_size = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = btn_x1 + (btn_w - text_size[0]) // 2
    text_y = btn_y1 + (btn_h + text_size[1]) // 2
    cv2.putText(
        seekbar,
        btn_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    
    # フレームとシークバーを結合
    combined = np.vstack([frame, seekbar])
    return combined, total_height


def compute_display_scale(orig_w, orig_h, max_w, max_h):
    """原画が max_w x max_h に収まる縮小率を返す（等倍以下）"""
    if orig_w <= max_w and orig_h <= max_h:
        return 1.0
    return min(max_w / orig_w, max_h / orig_h)


# -------------------------
# 映像解析メインループ
# -------------------------
def run_video_app(video_path):
    # 初期化
    player = VideoPlayer()
    detector = YoloDetector()
    tracker = SortTracker()
    speed_estimator = SpeedEstimator()
    plate_ocr = PlateOCR()
    renderer = OverlayRenderer()

    player.open(video_path)
    player.set_paused(True)  # 最初は一時停止状態で開始

    calibration = CalibrationManager()
    speed_estimator.set_calibration(calibration)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)

    mouse = MouseHandler()
    cv2.setMouseCallback(config.WINDOW_NAME, mouse.callback, player)
    mouse.calibration = calibration  # ドラッグ処理用に参照を渡す

    tracked_cars = []
    speeds = {}  # 速度情報を保持
    plate_bboxes = {}  # track_id -> (x1, y1, x2, y2) ナンバープレートBBOX
    show_car_bbox = True
    show_plate_bbox = False
    last_frame_for_display = None  # 一時停止中の表示用フレーム
    open_dialogs = []  # 開いているダイアログのリスト
    display_scale = None  # フレーム縮小率
    frame_orig = None     # 高解像度プレート切り出し用の元フレーム

    while True:
        # ウィンドウが閉じられたかチェック
        if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        
        ret, frame = player.read_frame()
        
        # 映像終了時は最後のフレームで一時停止（アプリは閉じない）
        if not ret:
            player.set_paused(True)
            # 最後のフレームがあればそれを使用
            if player.last_frame is not None:
                frame = player.last_frame
            else:
                # フレームがない場合はキー入力待ち
                key = cv2.waitKey(100) & 0xFF
                if key == ord("q"):
                    break
                continue

        if frame is not None:
            # 元フレームを保持（高解像度でのプレート切り出し用）
            frame_orig = frame

            # 初回のみ縮小率を計算
            if display_scale is None:
                display_scale = compute_display_scale(
                    frame.shape[1], frame.shape[0],
                    config.MAX_DISPLAY_WIDTH, config.MAX_DISPLAY_HEIGHT
                )

            # 大きすぎる場合は縮小（cv2.resize は新しい配列を返す）
            if display_scale < 1.0:
                new_w = int(frame.shape[1] * display_scale)
                new_h = int(frame.shape[0] * display_scale)
                frame = cv2.resize(frame, (new_w, new_h))

            # フレームサイズを保存（シークバー用）
            player.frame_width = frame.shape[1]
            player.frame_height = frame.shape[0]
            
            # マウスハンドラにシークバーのY座標を設定
            mouse.seekbar_y = frame.shape[0]
            
            # シーク発生時に速度履歴をクリア（速度爆発防止）
            if mouse._seek_happened:
                mouse._seek_happened = False
                speed_estimator.clear_history()

            # 一時停止中は検出・追跡をスキップし、前回の結果を再利用
            if not player.is_paused():
                # 車検出
                cars = detector.detect_cars(frame)
                tracked_cars = tracker.update(cars)

                # 静止車両を除外（十分な履歴が溜まってから判定）
                tracked_cars = [tc for tc in tracked_cars if tracker.is_moving(tc["track_id"])]

                # 動画時間を算出（wall-clockではなく動画のFPSベース）
                fps = player.get_fps()
                video_time = player.get_current_frame() / fps if fps > 0 else 0.0
                speeds = speed_estimator.update(tracked_cars, video_time=video_time)
                
                # ナンバープレートBBOX検出（表示ON時のみ）
                if show_plate_bbox:
                    plate_bboxes = {}
                    for tc in tracked_cars:
                        tid = tc["track_id"]
                        pbbox = detector.detect_plate_in_car(frame, tc["bbox"])
                        if pbbox is not None:
                            plate_bboxes[tid] = pbbox
            
            # 描画（一時停止中も再生中も同じフレームに描画）
            display_base = frame.copy()
            display_base = renderer.draw(display_base, tracked_cars, speeds,
                                        show_car_bbox=show_car_bbox,
                                        show_plate_bbox=show_plate_bbox,
                                        plate_bboxes=plate_bboxes)
            
            # 映像選択ボタン（左上）
            file_btn_x1, file_btn_y1 = 10, 70
            file_btn_x2, file_btn_y2 = 110, 95
            cv2.rectangle(display_base, (file_btn_x1, file_btn_y1), (file_btn_x2, file_btn_y2), (80, 80, 80), -1)
            cv2.rectangle(display_base, (file_btn_x1, file_btn_y1), (file_btn_x2, file_btn_y2), (150, 150, 150), 1)
            cv2.putText(
                display_base,
                "Open Video",
                (file_btn_x1 + 5, file_btn_y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            button_manager.set_file_button_rect((file_btn_x1, file_btn_y1, file_btn_x2, file_btn_y2))

            # チェックボックス描画
            cb_x = 130
            cb_y1 = 72
            car_cb_rect = draw_checkbox(display_base, cb_x, cb_y1, "Car BBOX", show_car_bbox)
            button_manager.set_car_checkbox_rect(car_cb_rect)
            cb_y2 = cb_y1 + 18
            plate_cb_rect = draw_checkbox(display_base, cb_x, cb_y2, "Plate BBOX", show_plate_bbox)
            button_manager.set_plate_checkbox_rect(plate_cb_rect)

            # キャリブレーショントグルボタン
            tog_x = 250
            # 4点: 進捗表示 (0/4 〜 4/4)
            quad_n = len(calibration.quad_points)
            if calibration.quad_state == CalibrationManager.STATE_DONE:
                quad_label = "4-Pt OK"
            elif calibration.active_toggle == "quad":
                quad_label = f"4-Pt({quad_n}/4)"
            else:
                quad_label = "4-Point"
            quad_rect = draw_toggle_button(
                display_base, tog_x, cb_y1, quad_label, calibration.quad_state, w=64)
            button_manager.set_quad_toggle_rect(quad_rect)

            # 方向: ステータス表示
            if calibration.dir_state == CalibrationManager.STATE_DONE:
                dir_label = "Dir OK"
            elif calibration.active_toggle == "direction":
                dir_n = 1 if calibration.dir_point1 else 0
                dir_label = f"Dir({dir_n}/2)"
            else:
                dir_label = "Direction"
            dir_rect = draw_toggle_button(
                display_base, tog_x, cb_y2, dir_label, calibration.dir_state, w=64)
            button_manager.set_dir_toggle_rect(dir_rect)

            # 幅テキストボックス（クリックで編集）
            w_val = f"{calibration.real_width_m:.1f}m"
            w_rect = draw_text_box(
                display_base, tog_x + 70, cb_y1, "W:", w_val, w=68)
            button_manager.set_width_textbox_rect(w_rect)

            # 奥行きテキストボックス（クリックで編集）
            h_val = f"{calibration.real_height_m:.1f}m"
            h_rect = draw_text_box(
                display_base, tog_x + 70, cb_y2, "H:", h_val, w=68)
            button_manager.set_height_textbox_rect(h_rect)

            # キャリブレーション状態表示
            if calibration.is_calibrated():
                cv2.putText(display_base, "CAL OK",
                            (tog_x + 144, cb_y1 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 0), 1, cv2.LINE_AA)

            # 再生状態を表示
            status = "|| PAUSED" if player.is_paused() else "▶ PLAYING"
            cv2.putText(
                display_base,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            # 操作案内
            cv2.putText(
                display_base,
                "SPACE: Play/Pause | Q: Quit | ESC: Cancel | Click car for plate",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            # キャリブレーションオーバーレイ描画
            calibration.draw_overlays(display_base)

            # キャリブレーションモード時のヒント表示
            if calibration.active_toggle is not None:
                hint = ""
                if calibration.active_toggle == "quad":
                    n = len(calibration.quad_points)
                    labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
                    if n < 4:
                        hint = f"Click {n+1}/4: {labels[n]} of road rect  [ESC=cancel]"
                elif calibration.active_toggle == "direction":
                    if calibration.dir_state == CalibrationManager.STATE_ACTIVE:
                        hint = "Click travel START point  [ESC=cancel]"
                    else:
                        hint = "Click travel END point  [ESC=cancel]"
                if hint:
                    cv2.putText(display_base, hint,
                                (10, display_base.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 200, 255), 2, cv2.LINE_AA)

            # シークバーを下に追加
            display_frame, total_bar_height = draw_seekbar(display_base, player, config.SEEKBAR_HEIGHT, config.BUTTON_AREA_HEIGHT)
            mouse.button_area_y = display_base.shape[0] + config.SEEKBAR_HEIGHT  # ボタン領域のY開始位置
            
            cv2.imshow(config.WINDOW_NAME, display_frame)

        # クリック処理（シークバー外のクリック）
        if mouse.clicked_pos is not None:
            x, y = mouse.clicked_pos
            mouse.clicked_pos = None

            # キャリブレーションモード: 映像クリックを基準点設定に使用
            _cal_consumed = (calibration.active_toggle is not None
                             and calibration.handle_click(x, y))

            track_id = None if _cal_consumed else select_track_id(x, y, tracked_cars)
            if track_id is not None:
                # クリックされた車両に対してのみナンバープレート検出を実行
                for tracked in tracked_cars:
                    if tracked["track_id"] == track_id:
                        car_bbox = tracked["bbox"]
                        
                        # ===== 第1段階: 通常のYOLOナンバープレート検出 =====
                        plate_bbox = detector.detect_plate_in_car(frame, car_bbox)
                        
                        # ===== 第2段階: 検出失敗時、車の下半分で再検出 =====
                        if plate_bbox is None:
                            print(f"第1段階失敗: track_id={track_id}、車の下半分で再試行")
                            x1, y1, x2, y2 = car_bbox
                            h = y2 - y1
                            lower_half_bbox = (x1, y1 + h // 2, x2, y2)
                            plate_bbox = detector.detect_plate_in_car(frame, lower_half_bbox)
                        
                        # ===== 第3段階: グレースケール + 白黒反転で再検出 =====
                        if plate_bbox is None:
                            print(f"第2段階失敗: track_id={track_id}、グレースケール+反転で再試行")
                            x1, y1, x2, y2 = car_bbox
                            car_crop = frame[y1:y2, x1:x2]
                            gray_crop = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
                            inverted = cv2.bitwise_not(gray_crop)
                            inverted_bgr = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
                            temp_frame = frame.copy()
                            temp_frame[y1:y2, x1:x2] = inverted_bgr
                            plate_bbox = detector.detect_plate_in_car(temp_frame, car_bbox)
                        
                        # ===== 最終判定 =====
                        if plate_bbox is None:
                            print(f"❌ ナンバープレート検出失敗: track_id={track_id}")
                            import tkinter.messagebox as messagebox
                            messagebox.showwarning(
                                "検出失敗",
                                f"車両ID {track_id} のナンバープレートを検出できませんでした。"
                            )
                            break
                        
                        try:
                            # ダイアログが既に開いている場合は新たに開かない
                            if open_dialogs:
                                print(f"ダイアログが既に開いています。閉じてから再度クリックしてください。")
                                break

                            # ナンバープレート切り出し（元解像度から切り出して高品質を維持）
                            if display_scale is not None and display_scale < 1.0:
                                inv = 1.0 / display_scale
                                orig_pb = tuple(int(c * inv) for c in plate_bbox)
                                plate_crop = frame_orig[orig_pb[1]:orig_pb[3], orig_pb[0]:orig_pb[2]].copy()
                            else:
                                plate_crop = frame[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]].copy()
                            
                            # 手動操作ダイアログを表示
                            dialog = PlateDialog(track_id, plate_ocr, plate_crop)
                            dialog.show()
                            open_dialogs.append(dialog)
                            
                            print(f"Track ID {track_id} のダイアログを表示しました")
                            
                        except Exception as e:
                            print(f"ダイアログ処理エラー: {e}")
                            import traceback
                            traceback.print_exc()
                        break

        # チェックボックストグル処理
        if mouse.toggle_car_bbox:
            mouse.toggle_car_bbox = False
            show_car_bbox = not show_car_bbox
        if mouse.toggle_plate_bbox:
            mouse.toggle_plate_bbox = False
            show_plate_bbox = not show_plate_bbox

        # キャリブレーショントグル処理
        if mouse.toggle_quad:
            mouse.toggle_quad = False
            calibration.activate_quad()
        if mouse.toggle_direction:
            mouse.toggle_direction = False
            calibration.activate_dir()
        if mouse.edit_width:
            mouse.edit_width = False
            calibration.prompt_edit_width()
        if mouse.edit_height:
            mouse.edit_height = False
            calibration.prompt_edit_height()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            player.toggle_pause()
        elif key == 27:  # ESCキー
            if calibration.active_toggle is not None:
                calibration.cancel_active()
                print("キャリブレーション入力をキャンセルしました")
        
        # 開いているダイアログのTkinterイベントを処理
        open_dialogs = [d for d in open_dialogs if d.process_events()]
        
        # 映像選択リクエスト処理
        if mouse.request_file_select:
            mouse.request_file_select = False
            # Tkinterでファイル選択ダイアログ
            temp_root = tk.Tk()
            temp_root.withdraw()  # メインウィンドウを隠す
            new_path = filedialog.askopenfilename(
                title="動画を選択",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
            )
            temp_root.destroy()
            
            if new_path:
                # 新しい動画を開く
                player.release()
                player.open(new_path)
                player.set_paused(True)
                # 全ての状態をリセット
                tracker = SortTracker()
                speed_estimator = SpeedEstimator()
                speed_estimator.set_calibration(calibration)
                plate_ocr = PlateOCR()  # OCRもリセット
                tracked_cars = []
                speeds = {}
                plate_bboxes = {}
                mouse.clicked_pos = None
                mouse.seekbar_dragging = False
                display_scale = None  # 縮小率を再計算
                calibration.reset()

    player.release()
    cv2.destroyAllWindows()


# -------------------------
# 起動時UI（Tkinter）
# -------------------------
def main():
    root = tk.Tk()
    root.title("車両解析アプリ")
    root.geometry("300x150")

    def on_select_video():
        path = filedialog.askopenfilename(
            title="動画を選択",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not path:
            return

        root.destroy()          # ★ 入口UIを閉じる
        run_video_app(path)     # ★ OpenCV側へ制御を渡す

    label = tk.Label(
        root,
        text="解析する映像を選択してください",
        font=("Arial", 12)
    )
    label.pack(pady=20)

    btn = tk.Button(
        root,
        text="映像選択",
        font=("Arial", 12),
        width=15,
        command=on_select_video
    )
    btn.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
