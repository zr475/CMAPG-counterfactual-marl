"""诊断俯视图颜色：遍历所有像素的HSV值，找可能的目的地颜色"""
import cv2
import numpy as np
from capture import ScreenCapture
from config import TOPVIEW_REGION

cap = ScreenCapture()
frame = cap.grab()
tv = frame[TOPVIEW_REGION["top"]:TOPVIEW_REGION["top"]+TOPVIEW_REGION["height"],
           TOPVIEW_REGION["left"]:TOPVIEW_REGION["left"]+TOPVIEW_REGION["width"]]
hsv = cv2.cvtColor(tv, cv2.COLOR_BGR2HSV)

print(f"俯视图尺寸: {tv.shape}")
print(f"HSV 范围: H=[{hsv[:,:,0].min()},{hsv[:,:,0].max()}] S=[{hsv[:,:,1].min()},{hsv[:,:,1].max()}] V=[{hsv[:,:,2].min()},{hsv[:,:,2].max()}]")

# 找非黑/非白/非灰的"有色彩"像素（S>80, V>80）
colorful = (hsv[:,:,1] > 80) & (hsv[:,:,2] > 80)
color_hsv = hsv[colorful]

if len(color_hsv) > 0:
    print(f"\n有色彩像素数: {len(color_hsv)} / {hsv.size//3}")
    # 按 H 分组统计
    for h_range, label in [((0,20),"红/橙"), ((20,40),"橙/黄"), ((40,80),"绿"),
                            ((80,110),"蓝-青"), ((110,140),"蓝-紫"), ((140,170),"紫-粉")]:
        mask = (color_hsv[:,0] >= h_range[0]) & (color_hsv[:,0] < h_range[1])
        count = np.sum(mask)
        if count > 50:
            avg_h = color_hsv[mask,0].mean()
            avg_s = color_hsv[mask,1].mean()
            avg_v = color_hsv[mask,2].mean()
            print(f"  H{h_range[0]:3d}-{h_range[1]:3d} ({label:6s}): {count:5d} px, avg H={avg_h:.0f} S={avg_s:.0f} V={avg_v:.0f}")
else:
    print("无有色彩像素!")

# 找所有非黑像素(V>50)中最常见的颜色
bright = hsv[:,:,2] > 50
print(f"\n常见色版（采样）:")
sample = hsv[bright][::100]  # 每100个采样
if len(sample) > 0:
    for i in range(min(20, len(sample))):
        h_val, s_val, v_val = sample[i]
        print(f"  H={h_val:3d} S={s_val:3d} V={v_val:3d}")
