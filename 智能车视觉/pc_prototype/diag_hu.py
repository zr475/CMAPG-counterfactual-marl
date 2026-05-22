"""
诊断 Hu 矩数字识别 — 用已保存的截图测试每一步
输出到 D:/temp/
"""
import cv2
import numpy as np
from pathlib import Path
import os

from config import FPV_REGION, IMAGE_NUM_DIR

os.makedirs("D:/temp", exist_ok=True)
OUT = "D:/temp/diag_hu"

def _imread(path: str) -> np.ndarray | None:
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def _imsave(path, img):
    ext = os.path.splitext(path)[1]
    buf = cv2.imencode(ext, img)[1]
    with open(path, "wb") as f:
        f.write(buf)

# ========== Step 0: 加载参考图 ==========
def load_ref_contours():
    refs = {}
    for p in sorted(Path(IMAGE_NUM_DIR).glob("*.jpg")):
        idx = int(p.stem[:2])
        img = _imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            refs[idx] = max(contours, key=cv2.contourArea)
            print(f"  Ref {idx}: contour area={cv2.contourArea(refs[idx]):.0f}")
    return refs

def load_ref_images():
    refs = {}
    for p in sorted(Path(IMAGE_NUM_DIR).glob("*.jpg")):
        idx = int(p.stem[:2])
        img = _imread(str(p))
        if img is not None:
            refs[idx] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return refs

# ========== 加载最新截图 ==========
captures_dir = Path("captures")
frames = sorted(captures_dir.glob("frame_*.png"))
if not frames:
    print("ERROR: no captures found!")
    exit(1)

latest = frames[-1]
print(f"Loading: {latest}")

frame = cv2.imread(str(latest))
if frame is None:
    print(f"ERROR: cannot read {latest}")
    exit(1)

h, w = frame.shape[:2]
print(f"  Frame size: {w}x{h}")

# 保存完整截图供参考
_imsave(f"{OUT}_0_full_frame.png", frame)

# 提取 FPV 区域
fpv = frame[
    FPV_REGION["top"]:FPV_REGION["top"] + FPV_REGION["height"],
    FPV_REGION["left"]:FPV_REGION["left"] + FPV_REGION["width"],
]
_imsave(f"{OUT}_1_fpv_raw.png", fpv)
print(f"  FPV raw: {fpv.shape}")

# 保存 topview 区域
topview = frame[
    0:1200,
    0:960,
]
_imsave(f"{OUT}_1_topview.png", topview)

# ========== Step 1: 旋转 ==========
print("\n=== Step 1: Rotate FPV ===")
fpv_rot = cv2.rotate(fpv, cv2.ROTATE_90_COUNTERCLOCKWISE)
_imsave(f"{OUT}_2_fpv_rotated.png", fpv_rot)
print(f"  Rotated: {fpv_rot.shape}")

# ========== Step 2: 找白底标志牌 ==========
print("\n=== Step 2: Find white sign ===")
hsv = cv2.cvtColor(fpv_rot, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(fpv_rot, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# 放宽白色检测范围
white_mask = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 80, 255]))
white_mask[:int(h * 0.3), :] = 0  # exclude sky

kernel = np.ones((5, 5), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

_imsave(f"{OUT}_3_white_mask.png", white_mask)

# 可视化白色mask叠加
overlay = fpv_rot.copy()
overlay[white_mask > 0] = (0, 255, 255)
_imsave(f"{OUT}_3_white_overlay.png", overlay)

contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"  White regions: {len(contours)}")

# 画所有白色轮廓
disp_all = fpv_rot.copy()
for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, cw, ch = cv2.boundingRect(cnt)
    if area > 50:
        cv2.rectangle(disp_all, (x, y), (x + cw, y + ch), (0, 255, 0), 1)
        cv2.putText(disp_all, f"{area:.0f}", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
_imsave(f"{OUT}_3_all_white_rects.png", disp_all)

best = None
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 200 or area > (w * h) * 0.08:
        continue
    x, y, cw, ch = cv2.boundingRect(cnt)
    ar = cw / ch if ch > 0 else 0
    if ar < 0.2 or ar > 5.0:
        continue
    roi_g = gray[y:y + ch, x:x + cw]
    dark_ratio = np.sum(roi_g < 80) / roi_g.size if roi_g.size > 0 else 0
    print(f"  Candidate: area={area:.0f} ({x},{y}) {cw}x{ch} ar={ar:.2f} dark={dark_ratio:.3f}")
    if dark_ratio < 0.02:
        continue
    if best is None or area > best[0]:
        best = (area, x, y, cw, ch, dark_ratio)

if best is None:
    print("  [FAIL] No valid white sign found!")
    print("  Printing ALL contours for debug:")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        ar = cw / ch if ch > 0 else 0
        if area > 50:
            roi_g = gray[y:y + ch, x:x + cw]
            dark_ratio = np.sum(roi_g < 80) / roi_g.size if roi_g.size > 0 else 0
            print(f"    area={area:.0f} ({x},{y}) {cw}x{ch} ar={ar:.2f} dark={dark_ratio:.3f}")
    exit(1)

area, x, y, cw, ch, dr = best
print(f"\n  [OK] Best: area={area:.0f} ({x},{y}) {cw}x{ch} dark={dr:.3f}")

disp = fpv_rot.copy()
cv2.rectangle(disp, (x, y), (x + cw, y + ch), (0, 255, 0), 3)
_imsave(f"{OUT}_4_detected.png", disp)

# ========== Step 3: Extract ROI ==========
print("\n=== Step 3: Extract digit ROI ===")
roi = fpv_rot[y:y + ch, x:x + cw]
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_imsave(f"{OUT}_5_roi.png", roi)

print(f"  ROI stats: min={roi_gray.min()} max={roi_gray.max()} mean={roi_gray.mean():.1f}")

# 多阈值二值化
for thresh, label in [(60, "a_60"), (80, "b_80"), (100, "c_100"), (120, "d_120")]:
    _, bn = cv2.threshold(roi_gray, thresh, 255, cv2.THRESH_BINARY_INV)
    _imsave(f"{OUT}_6_bin_{label}.png", bn)

# OTSU
_, bn_otsu = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_imsave(f"{OUT}_6_bin_e_otsu.png", bn_otsu)

# 自适应
bn_adapt = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 3)
_imsave(f"{OUT}_6_bin_f_adaptive.png", bn_adapt)

# ========== Step 4: Hu Moments Match ==========
print("\n=== Step 4: Hu moments matching ===")
ref_contours = load_ref_contours()
ref_images = load_ref_images()

for name, thresh, binary in [("60", 60, roi_gray), ("80", 80, roi_gray), ("100", 100, roi_gray), ("120", 120, roi_gray)]:
    # 需要先二值化
    _, binary = cv2.threshold(roi_gray, int(name), 255, cv2.THRESH_BINARY_INV)
    kernel_m = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_m)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"  thresh={name}: no contours")
        continue

    cnt = max(contours, key=cv2.contourArea)
    cnt_area = cv2.contourArea(cnt)
    if cnt_area < 20:
        print(f"  thresh={name}: contour too small ({cnt_area:.0f})")
        continue

    results = []
    for idx, ref_cnt in ref_contours.items():
        dist = cv2.matchShapes(cnt, ref_cnt, cv2.CONTOURS_MATCH_I2, 0)
        results.append((dist, idx))
    results.sort()
    print(f"  thresh={name} (cnt_area={cnt_area:.0f}):")
    for dist, idx in results[:3]:
        conf = max(0.0, 1.0 - dist / 3.0)
        print(f"    {idx}: dist={dist:.4f} conf={conf:.2f}")

# ========== Step 5: Template matching fallback ==========
print("\n=== Step 5: Template matching ===")
roi_resized = cv2.resize(roi_gray, (340, 340))
for idx, ref_gray in ref_images.items():
    _, ref_bin = cv2.threshold(ref_gray, 128, 255, cv2.THRESH_BINARY_INV)
    _, roi_bin = cv2.threshold(roi_resized, 100, 255, cv2.THRESH_BINARY_INV)
    score = cv2.matchTemplate(roi_bin, ref_bin, cv2.TM_CCOEFF_NORMED)[0][0]
    print(f"  ref_{idx}: score={score:.4f}")

print(f"\nDone. Output in D:/temp/")
