"""诊断脚本：截一帧，可视化所有检测结果"""
import cv2
import numpy as np
from pathlib import Path

from capture import ScreenCapture
from config import (TOPVIEW_REGION, FPV_REGION, DEST_THRESHOLD, BOX_THRESHOLD,
                    FPV_DEST_THRESHOLD, FPV_BOX_THRESHOLD, GREEN_CAR_THRESHOLD, CYAN_CAR_THRESHOLD)

OUT = Path(__file__).parent / "diagnostic"
OUT.mkdir(exist_ok=True)

def _hsv_mask(img, threshold):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(threshold[0])
    upper = np.array(threshold[1])
    return cv2.inRange(hsv, lower, upper)

cap = ScreenCapture()
frame = cap.grab()
cv2.imwrite(str(OUT / "00_full_frame.png"), frame)
print(f"全屏尺寸: {frame.shape}")

# 俯视图
tv = frame[TOPVIEW_REGION["top"]:TOPVIEW_REGION["top"]+TOPVIEW_REGION["height"],
           TOPVIEW_REGION["left"]:TOPVIEW_REGION["left"]+TOPVIEW_REGION["width"]]
cv2.imwrite(str(OUT / "01_topview.png"), tv)
print(f"俯视图尺寸: {tv.shape}")

# 俯视检测
for name, thr in [("dest", DEST_THRESHOLD), ("box", BOX_THRESHOLD),
                   ("green_car", GREEN_CAR_THRESHOLD), ("cyan_car", CYAN_CAR_THRESHOLD)]:
    mask = _hsv_mask(tv, thr)
    cv2.imwrite(str(OUT / f"02_top_{name}_mask.png"), mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 50]
    print(f"俯视 {name}: {len(valid)} 个轮廓")
    # 画检测框
    vis = tv.copy()
    for c in valid:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(str(OUT / f"03_top_{name}_vis.png"), vis)

# FPV
fpv = frame[FPV_REGION["top"]:FPV_REGION["top"]+FPV_REGION["height"],
            FPV_REGION["left"]:FPV_REGION["left"]+FPV_REGION["width"]]
cv2.imwrite(str(OUT / "04_fpv.png"), fpv)
print(f"FPV尺寸: {fpv.shape}")

# FPV检测
for name, thr in [("fpv_dest", FPV_DEST_THRESHOLD), ("fpv_box", FPV_BOX_THRESHOLD)]:
    mask = _hsv_mask(fpv, thr)
    cv2.imwrite(str(OUT / f"05_{name}_mask.png"), mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > 50]
    print(f"FPV {name}: {len(valid)} 个轮廓")
    vis = fpv.copy()
    for c in valid:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(str(OUT / f"06_{name}_vis.png"), vis)

# FPV 目的地 ROI 细节
mask = _hsv_mask(fpv, FPV_DEST_THRESHOLD)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate([c for c in contours if cv2.contourArea(c) > 50]):
    x, y, w, h = cv2.boundingRect(c)
    m = 15
    fy, fx = fpv.shape[:2]
    x1 = max(0, x-m); y1 = max(0, y-m)
    x2 = min(fx, x+w+m); y2 = min(fy, y+h+m)
    roi = fpv[y1:y2, x1:x2]
    cv2.imwrite(str(OUT / f"07_dest_roi_{i}.png"), roi)
    # 白字提取
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 70, 255]))
    cv2.imwrite(str(OUT / f"08_dest_white_{i}.png"), white_mask)
    print(f"目的地 ROI {i}: 尺寸={roi.shape[1]}x{roi.shape[0]}, 白字像素={cv2.countNonZero(white_mask)}")

print(f"\n结果保存在: {OUT}")
