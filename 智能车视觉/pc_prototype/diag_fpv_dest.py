"""诊断 FPV 目的地检测：逐步骤打印过滤信息"""
import cv2
import numpy as np
from pathlib import Path

from capture import ScreenCapture
from config import FPV_REGION, FPV_DEST_THRESHOLD

OUT = Path(__file__).parent / "diagnostic"
OUT.mkdir(exist_ok=True)

def _hsv_mask(img, threshold):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(threshold[0])
    upper = np.array(threshold[1])
    return cv2.inRange(hsv, lower, upper)

cap = ScreenCapture()
frame = cap.grab()
fpv = frame[FPV_REGION["top"]:FPV_REGION["top"]+FPV_REGION["height"],
            FPV_REGION["left"]:FPV_REGION["left"]+FPV_REGION["width"]]
print(f"FPV尺寸: {fpv.shape}")

cv2.imwrite(str(OUT / "debug_fpv.png"), fpv)

mask = _hsv_mask(fpv, FPV_DEST_THRESHOLD)
cv2.imwrite(str(OUT / "debug_fpv_dest_mask.png"), mask)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nFPV_DEST_THRESHOLD 轮廓总数: {len(contours)}")

fy, fx = fpv.shape[:2]
max_area = (fx * fy) * 0.05

candidates = []
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area < 30:
        if i < 10:
            print(f"  [{i}] 拒绝: area={area:.0f} < 30")
        continue
    if area > max_area:
        print(f"  [{i}] 拒绝: area={area:.0f} > max={max_area:.0f} ({(area/max_area)*100:.0f}% of frame)")
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 10 or h < 10:
        print(f"  [{i}] 拒绝: size={w}x{h} too small")
        continue
    ar = w / h if h > 0 else 0
    if ar < 0.3 or ar > 3.0:
        print(f"  [{i}] 拒绝: AR={ar:.2f} out of [0.3, 3.0]")
        continue
    x1, y1 = max(0, x-10), max(0, y-10)
    x2, y2 = min(fx, x+w+10), min(fy, y+h+10)
    roi = fpv[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))
    white_ratio = cv2.countNonZero(white) / roi.size * 3 if roi.size > 0 else 0
    if white_ratio < 0.05:
        print(f"  [{i}] 拒绝: white_ratio={white_ratio:.3f} < 0.05, white_px={cv2.countNonZero(white)}")
        continue
    candidates.append((area, x, y, w, h, white_ratio))
    print(f"  [{i}] 通过: area={area:.0f} size={w}x{h} AR={ar:.2f} white={white_ratio:.3f}")
    # 保存通过筛选的ROI
    roi_full = fpv[y:y+h, x:x+w]
    cv2.imwrite(str(OUT / f"debug_candidate_{len(candidates)-1}_roi.png"), roi_full)
    # 带margin的ROI
    roi_margin = fpv[max(0,y-15):min(fy,y+h+15), max(0,x-15):min(fx,x+w+15)]
    cv2.imwrite(str(OUT / f"debug_candidate_{len(candidates)-1}_roi_margin.png"), roi_margin)
    cv2.imwrite(str(OUT / f"debug_candidate_{len(candidates)-1}_white.png"), white)

print(f"\n通过筛选总数: {len(candidates)}")

# 可视化
vis = fpv.copy()
for i, (_, x, y, w, h, wr) in enumerate(candidates):
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(vis, f"#{i} wr={wr:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
cv2.imwrite(str(OUT / "debug_fpv_candidates.png"), vis)

# 如果有候选人，跑匹配测试
if candidates:
    from detect import get_fpv_dest_matcher
    matcher = get_fpv_dest_matcher()
    candidates.sort(key=lambda c: c[5], reverse=True)
    for i, (_, x, y, w, h, _) in enumerate(candidates[:5]):
        margin = 15
        x1 = max(0, x-margin); y1 = max(0, y-margin)
        x2 = min(fx, x+w+margin); y2 = min(fy, y+h+margin)
        roi = fpv[y1:y2, x1:x2]
        idx, score = matcher.match(roi)
        print(f"\n  候选 #{i}: ({x},{y}) {w}x{h}, 匹配结果: id={idx} score={score:.4f}")
        if idx >= 0:
            print(f"  *** 识别成功! 数字={idx} ***")
else:
    print("\n无候选通过筛选!")

print(f"\n诊断图片保存在: {OUT}")
