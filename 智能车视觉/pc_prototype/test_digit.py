"""
FPV 数字识别实时测试
用 image_num/ 参考图做多尺度模板匹配，在 FPV 中识别目的地数字
"""
import cv2
import numpy as np
from pathlib import Path

from capture import ScreenCapture
from config import FPV_REGION, IMAGE_NUM_DIR


def load_refs():
    refs = {}
    for p in sorted(Path(IMAGE_NUM_DIR).glob("*.jpg")):
        idx = int(p.stem[:2])
        with open(str(p), "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, ref_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
            refs[idx] = ref_bin
    return refs


def detect_digit(fpv_gray, refs, min_score=0.35, min_gap=0.05):
    """
    多尺度滑动窗口模板匹配，返回 (digit_id, score, x, y, w, h) 或 None
    """
    h, w = fpv_gray.shape
    all_matches = []

    for idx, ref in refs.items():
        for scale in np.linspace(0.08, 0.6, 20):
            nw = int(ref.shape[1] * scale)
            nh = int(ref.shape[0] * scale)
            if nw < 15 or nh < 15 or nw > w or nh > h:
                continue
            tmpl = cv2.resize(ref, (nw, nh))
            result = cv2.matchTemplate(fpv_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            all_matches.append((score, idx, loc[0], loc[1], nw, nh))

    all_matches.sort(reverse=True)
    if not all_matches:
        return None

    best_score, best_idx, x, y, nw, nh = all_matches[0]
    # 找其他数字的最佳分数
    other_best = 0.0
    for s, idx, _, _, _, _ in all_matches:
        if idx != best_idx:
            other_best = s
            break

    gap = best_score - other_best
    if best_score < min_score or gap < min_gap:
        return None

    return best_idx, best_score, x, y, nw, nh


def main():
    refs = load_refs()
    print(f"已加载参考图: {sorted(refs.keys())}")
    print("\n=== FPV 数字识别实时测试 ===")
    print("Q=退出  S=保存当前帧  T=终端打印详情")
    print()

    cap = ScreenCapture()
    cv2.namedWindow("Digit Recognition Test", cv2.WINDOW_NORMAL)

    last_digit = -1
    frame_count = 0

    while True:
        frame = cap.grab()
        if frame is None:
            continue

        fpv = frame[
            FPV_REGION["top"]:FPV_REGION["top"] + FPV_REGION["height"],
            FPV_REGION["left"]:FPV_REGION["left"] + FPV_REGION["width"],
        ]
        gray = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)

        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        result = detect_digit(enhanced, refs)

        display = fpv.copy()

        if result:
            idx, score, x, y, nw, nh = result
            cv2.rectangle(display, (x, y), (x + nw, y + nh), (0, 255, 0), 2)
            label = f"D{idx} ({score:.2f})"
            cv2.putText(display, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if idx != last_digit:
                print(f"  [帧{frame_count:05d}] 识别: 数字 {idx}  置信度: {score:.3f}")
                last_digit = idx
        else:
            cv2.putText(display, "No digit detected", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.putText(display, f"Frame: {frame_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Digit Recognition Test", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"test_digit_frame_{frame_count:05d}.png"
            cv2.imwrite(path, display)
            print(f"  已保存: {path}")
        elif key == ord("t") and result:
            idx, score, x, y, nw, nh = result
            print(f"  [详情] 数字={idx} 置信度={score:.3f} 位置=({x},{y}) 尺寸={nw}x{nh}")

        frame_count += 1

    cv2.destroyAllWindows()
    print("退出。")


if __name__ == "__main__":
    main()
