"""
目的地数字参考图采集工具 v2
开车靠近目的地 → 按数字键 0-9 保存 ROI
改进：过滤天空/大片噪点，取白字占比最高的候选区域，方形裁剪
"""

import cv2
import numpy as np
from pathlib import Path

from capture import ScreenCapture
from config import FPV_REGION, FPV_DEST_THRESHOLD

REF_DIR = Path(__file__).parent / "fpv_dest_refs"

def _hsv_mask(img, threshold):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(threshold[0]), np.array(threshold[1]))


def _find_best_candidate(fpv: np.ndarray) -> tuple | None:
    """找最像目的地的候选区域，返回 (x, y, w, h) 或 None"""
    mask = _hsv_mask(fpv, FPV_DEST_THRESHOLD)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    fy, fx = fpv.shape[:2]
    max_area = (fx * fy) * 0.08  # 不超过画面 8%（排除天空）
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 60 or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 12 or h < 12:
            continue
        ar = w / h if h > 0 else 0
        if ar < 0.2 or ar > 5.0:
            continue
        # 检查白字
        m = 10
        x1, y1 = max(0, x - m), max(0, y - m)
        x2, y2 = min(fx, x + w + m), min(fy, y + h + m)
        roi = fpv[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 70, 255]))
        white_px = cv2.countNonZero(white)
        if white_px < 30:
            continue
        candidates.append((area, x, y, w, h, white_px))

    if not candidates:
        return None
    # 取白字最多的（而非面积最大的）
    candidates.sort(key=lambda c: c[5], reverse=True)
    _, x, y, w, h, _ = candidates[0]
    return (x, y, w, h)


def main():
    REF_DIR.mkdir(exist_ok=True)

    existing = sorted(int(p.stem) for p in REF_DIR.glob("*.png") if p.stem.isdigit())
    print("=== 目的地数字参考图采集 v2 ===")
    print(f"已有参考图: {existing if existing else '无'}")
    missing = sorted(set(range(10)) - set(existing))
    if missing:
        print(f"还需采集: {missing}")
    print("\n操作: 开车靠近目的地 → FPV窗口出现绿框 → 按对应数字键(0-9) → q退出\n")

    cap = ScreenCapture()
    cv2.namedWindow("FPV", cv2.WINDOW_NORMAL)

    while True:
        frame = cap.grab()
        fpv = frame[
            FPV_REGION["top"]:FPV_REGION["top"] + FPV_REGION["height"],
            FPV_REGION["left"]:FPV_REGION["left"] + FPV_REGION["width"]
        ]
        fy, fx = fpv.shape[:2]

        display = fpv.copy()
        roi = None
        cand = _find_best_candidate(fpv)

        if cand is not None:
            x, y, w, h = cand
            # 做方形裁剪：以中心为基准，取 max(w,h) 为边长
            size = max(w, h) + 20
            cx, cy = x + w // 2, y + h // 2
            half = size // 2
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(fx, cx + half)
            y2 = min(fy, cy + half)
            # 确保正方形（取较小的边长）
            side = min(x2 - x1, y2 - y1)
            x2 = x1 + side
            y2 = y1 + side

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"DEST {w}x{h}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            roi = fpv[y1:y2, x1:x2]

        # 已有/缺失
        existing_now = sorted(int(p.stem) for p in REF_DIR.glob("*.png") if p.stem.isdigit())
        miss = sorted(set(range(10)) - set(existing_now))
        info = [f"已有: {existing_now if existing_now else '无'}"]
        if miss:
            info.append(f"缺失: {miss}")
        for i, txt in enumerate(info):
            cv2.putText(display, txt, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow("FPV", display)
        if roi is not None and roi.size > 0:
            cv2.imshow("ROI (press 0-9 to save)", roi)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if ord("0") <= key <= ord("9"):
            digit = key - ord("0")
            if roi is not None and roi.size > 0:
                path = REF_DIR / f"{digit:02d}.png"
                _, buf = cv2.imencode(".png", roi)
                with open(str(path), "wb") as f:
                    f.write(buf)
                print(f"已保存 数字 {digit} → {path.name} (尺寸: {roi.shape[1]}x{roi.shape[0]})")
                existing_now = sorted(int(p.stem) for p in REF_DIR.glob("*.png") if p.stem.isdigit())
                miss = sorted(set(range(10)) - set(existing_now))
                if miss:
                    print(f"  还需采集: {miss}")
                else:
                    print("  *** 全部 0-9 采集完毕! ***")
            else:
                print("未检测到目的地 — 请把车开近一点，让蓝色标记在画面中清晰可见。")

    cv2.destroyAllWindows()
    print("退出。")


if __name__ == "__main__":
    main()
