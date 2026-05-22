"""HSV 颜色采样 —— 点击图像任意位置，输出 BGR 和 HSV 值"""
import cv2
import numpy as np

from capture import ScreenCapture
from config import TOPVIEW_REGION

cap = ScreenCapture()
frame = cap.grab()
tv = frame[TOPVIEW_REGION["top"]:TOPVIEW_REGION["top"] + TOPVIEW_REGION["height"],
           TOPVIEW_REGION["left"]:TOPVIEW_REGION["left"] + TOPVIEW_REGION["width"]]

print(f"俯视图尺寸: {tv.shape[1]}x{tv.shape[0]}")
print("点击俯视图上的元素查看 HSV 值，按 Q 退出")
print()

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = tv[y, x]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"  ({x:4d},{y:4d})  BGR=({bgr[0]:3d},{bgr[1]:3d},{bgr[2]:3d})  HSV=({hsv[0]:3d},{hsv[1]:3d},{hsv[2]:3d})")

cv2.namedWindow("HSV Inspector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Inspector", 600, 800)
cv2.setMouseCallback("HSV Inspector", on_mouse)

while True:
    cv2.imshow("HSV Inspector", tv)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
