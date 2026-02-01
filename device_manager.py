# device_manager.py
# デバイス(CPU / GPU)管理

import torch
import config


class DeviceManager:
    def __init__(self):
        self.device = self._detect_device()

    def _detect_device(self):
        if config.USE_GPU and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def get_device(self):
        return self.device

    def is_gpu(self):
        return self.device == "cuda"
