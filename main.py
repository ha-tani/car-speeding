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

    def callback(self, event, x, y, flags, param):
        player = param
        if event == cv2.EVENT_LBUTTONDOWN:
            # 映像選択ボタンクリックの判定
            if button_manager.is_file_button_clicked(x, y):
                self.request_file_select = True
            # ボタンクリックの判定
            elif button_manager.is_play_button_clicked(x, y):
                player.toggle_pause()
            # シークバー領域のクリックか判定（ボタン領域より上）
            elif y >= self.seekbar_y and y < self.button_area_y:
                self.seekbar_dragging = True
                self._update_seek(x, player)
            elif y < self.seekbar_y:
                self.clicked_pos = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.seekbar_dragging:
                self._update_seek(x, player)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.seekbar_dragging:
                self._update_seek(x, player)
                self.seekbar_dragging = False
    
    def _update_seek(self, x, player):
        if player and player.frame_width > 0:
            ratio = max(0, min(1, x / player.frame_width))
            player.seek_ratio(ratio)


# ボタン領域を管理するクラス
class ButtonManager:
    def __init__(self):
        self.play_button_rect = None  # (x1, y1, x2, y2)
        self.file_button_rect = None  # 映像選択ボタン
    
    def set_play_button_rect(self, rect):
        self.play_button_rect = rect
    
    def set_file_button_rect(self, rect):
        self.file_button_rect = rect
    
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

button_manager = ButtonManager()


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

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    mouse = MouseHandler()
    cv2.setMouseCallback(config.WINDOW_NAME, mouse.callback, player)

    tracked_cars = []
    speeds = {}  # 速度情報を保持
    last_frame_for_display = None  # 一時停止中の表示用フレーム

    while True:
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
            # フレームサイズを保存（シークバー用）
            player.frame_width = frame.shape[1]
            player.frame_height = frame.shape[0]
            
            # マウスハンドラにシークバーのY座標を設定
            mouse.seekbar_y = frame.shape[0]
            
            # 一時停止中は検出・追跡をスキップし、前回の結果を再利用
            if not player.is_paused():
                # 車とナンバープレート領域を検出
                cars_with_plates = detector.detect_cars_with_plates(frame)
                
                # 追跡用にbboxとconfのみ抽出
                cars = [{"bbox": c["bbox"], "conf": c["conf"]} for c in cars_with_plates]
                tracked_cars = tracker.update(cars)
                
                # 追跡IDとナンバープレート領域を紐づけ（OCRは毎フレーム行わない）
                for tracked in tracked_cars:
                    tid = tracked["track_id"]
                    tx1, ty1, tx2, ty2 = tracked["bbox"]
                    
                    # 対応するナンバープレート領域を探す
                    for car in cars_with_plates:
                        cx1, cy1, cx2, cy2 = car["bbox"]
                        # BBoxが一致するか確認
                        if abs(cx1 - tx1) < 10 and abs(cy1 - ty1) < 10:
                            if car["plate_bbox"] is not None:
                                tracked["plate_bbox"] = car["plate_bbox"]
                            break
                
                speeds = speed_estimator.update(tracked_cars)
            
            # 描画（一時停止中も再生中も同じフレームに描画）
            display_base = frame.copy()
            display_base = renderer.draw(display_base, tracked_cars, speeds)
            
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
                "SPACE: Play/Pause | Q: Quit | Click car for plate",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

            # シークバーを下に追加
            display_frame, total_bar_height = draw_seekbar(display_base, player, config.SEEKBAR_HEIGHT, config.BUTTON_AREA_HEIGHT)
            mouse.button_area_y = display_base.shape[0] + config.SEEKBAR_HEIGHT  # ボタン領域のY開始位置
            
            cv2.imshow(config.WINDOW_NAME, display_frame)

        # クリック処理（シークバー外のクリック）
        if mouse.clicked_pos is not None:
            x, y = mouse.clicked_pos
            mouse.clicked_pos = None

            track_id = select_track_id(x, y, tracked_cars)
            if track_id is not None:
                # クリックされた車両のplate_bboxを取得してOCR実行
                for tracked in tracked_cars:
                    if tracked["track_id"] == track_id:
                        plate_bbox = tracked.get("plate_bbox")
                        if plate_bbox is not None:
                            plate_det = [{"bbox": plate_bbox}]
                            plate_ocr.update(frame, plate_det, track_id)
                        break
                
                plate_text = plate_ocr.get_plate_text(track_id)
                plate_image = plate_ocr.get_plate_image(track_id)
                PlateDialog(track_id, plate_text, plate_image).show()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            player.toggle_pause()
        
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
                # トラッカーと速度推定をリセット
                tracker = SortTracker()
                speed_estimator = SpeedEstimator()
                tracked_cars = []
                speeds = {}

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
