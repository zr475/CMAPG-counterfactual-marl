"""
图像采集模块
支持屏幕截图（默认）和 USB 摄像头两种模式
"""

import cv2
import numpy as np
from config import CAPTURE_REGION


class ScreenCapture:
    """使用 mss 或 PIL 截取屏幕区域"""

    def __init__(self, region: dict | None = None):
        self.region = region or CAPTURE_REGION
        self._mss = None
        self._using_mss = False
        try:
            from mss import mss
            self._mss = mss()
            self._using_mss = True
        except ImportError:
            pass

    def grab(self) -> np.ndarray:
        if self._using_mss:
            return self._grab_mss()
        return self._grab_pil()

    def _grab_mss(self) -> np.ndarray:
        monitor = {
            "top": self.region["top"],
            "left": self.region["left"],
            "width": self.region["width"],
            "height": self.region["height"],
        }
        img = self._mss.grab(monitor)
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

    def _grab_pil(self) -> np.ndarray:
        from PIL import ImageGrab
        r = self.region
        bbox = (r["left"], r["top"], r["left"] + r["width"], r["top"] + r["height"])
        img = ImageGrab.grab(bbox)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def update_region(self, region: dict):
        self.region = region


class CameraCapture:
    """USB 摄像头采集"""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def grab(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        self.cap.release()


def create_capture(mode: str = "screen", **kwargs):
    """
    mode: "screen" 或 "camera"
    """
    if mode == "camera":
        return CameraCapture(**kwargs)
    return ScreenCapture(**kwargs)
