# video_player.py
# 動画再生とフレーム管理のみを担当

import cv2


class VideoPlayer:
    def __init__(self):
        self.cap = None
        self.paused = False
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30
        self.last_frame = None  # 一時停止中に表示するフレーム
        self.frame_width = 0
        self.frame_height = 0

    def open(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.current_frame = 0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.paused = False
        # 最初のフレームを読み込んでおく（一時停止開始時用）
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        # 先頭に戻す
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0

    def toggle_pause(self):
        self.paused = not self.paused

    def is_paused(self):
        return self.paused

    def set_paused(self, paused: bool):
        self.paused = paused

    def read_frame(self):
        """
        return:
          ret: bool
          frame: ndarray or None
        """
        if self.cap is None:
            return False, None

        if self.paused:
            return True, self.last_frame

        ret, frame = self.cap.read()
        if ret:
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.last_frame = frame
        return ret, frame

    def seek(self, frame_number: int):
        """指定フレームにシーク"""
        if self.cap is None:
            return
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
        # シーク後に1フレーム読み込んで表示用に保持
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def seek_ratio(self, ratio: float):
        """0.0〜1.0の比率でシーク"""
        frame_number = int(ratio * self.total_frames)
        self.seek(frame_number)

    def get_position_ratio(self) -> float:
        """現在の再生位置を0.0〜1.0で返す"""
        if self.total_frames <= 0:
            return 0.0
        return self.current_frame / self.total_frames

    def get_current_frame(self) -> int:
        return self.current_frame

    def get_total_frames(self) -> int:
        return self.total_frames

    def get_fps(self) -> float:
        return self.fps

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
