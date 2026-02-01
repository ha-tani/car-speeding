# utils.py
# 汎用ユーティリティ関数

import time


class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0.0

    def update(self):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            self.fps = 1.0 / dt
        return self.fps


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))
