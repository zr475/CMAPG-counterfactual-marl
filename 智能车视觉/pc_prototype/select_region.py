"""
屏幕区域选择工具
运行后拖拽鼠标框选游戏窗口，按 SPACE 确认，按 R 重选，按 Q 退出
选好后自动更新 config.py 中的 CAPTURE_REGION
"""

import cv2
import numpy as np
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.py"

# 截全屏
from PIL import ImageGrab
full = ImageGrab.grab()
img = cv2.cvtColor(np.array(full), cv2.COLOR_RGB2BGR)
h, w = img.shape[:2]

print(f"屏幕分辨率: {w}x{h}")
print("操作: 拖拽鼠标框选游戏区域 → SPACE 确认 → 自动更新 config.py")

# 缩放显示（适配屏幕）
scale = min(1.0, 1200 / w, 800 / h)
disp = cv2.resize(img, (int(w * scale), int(h * scale)))

roi = cv2.selectROI("框选游戏区域 - SPACE确认 R重选 Q退出", disp, False)
cv2.destroyAllWindows()

if roi[2] == 0 or roi[3] == 0:
    print("未选择区域，退出")
    exit()

# 缩放回原始坐标
x, y, rw, rh = [int(v / scale) for v in roi]
region = {"top": y, "left": x, "width": rw, "height": rh}
print(f"\n选择的区域: left={x}, top={y}, width={rw}, height={rh}")

# 更新 config.py
config_text = CONFIG_PATH.read_text(encoding="utf-8")

import re
old = r'CAPTURE_REGION = \{[^}]+\}'
new = f'CAPTURE_REGION = {{"top": {y}, "left": {x}, "width": {rw}, "height": {rh}}}'
config_text = re.sub(old, new, config_text)

CONFIG_PATH.write_text(config_text, encoding="utf-8")
print(f"已更新 {CONFIG_PATH}")
