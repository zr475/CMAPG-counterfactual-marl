"""
诊断FPV白底黑字区域 —— 目的地数字识别
在FPV中找白色矩形区域，匹配 image_num/ 参考图
"""
import cv2
import numpy as np
from pathlib import Path

from capture import ScreenCapture
from config import FPV_REGION, IMAGE_NUM_DIR

REF_DIR = Path(IMAGE_NUM_DIR)


def _imread(path: str) -> np.ndarray | None:
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_refs() -> dict[int, np.ndarray]:
    """加载 image_num/ 参考图，返回 {digit: BGR_image}"""
    refs = {}
    for p in sorted(REF_DIR.glob("*.jpg")):
        idx = int(p.stem[:2])
        img = _imread(str(p))
        if img is not None:
            refs[idx] = img
    return refs


def find_white_rects(fpv: np.ndarray):
    """在FPV中找白底矩形区域（候选目的地标记）"""
    hsv = cv2.cvtColor(fpv, cv2.COLOR_BGR2HSV)
    h, w = fpv.shape[:2]

    # 白色/亮色区域: 低饱和度 + 高亮度
    lower = np.array([0, 0, 150])
    upper = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # 形态学去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = (w * h) * 0.15  # 不超过画面15%
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 15 or ch < 15:
            continue
        ar = cw / ch if ch > 0 else 0
        if ar < 0.3 or ar > 3.0:
            continue

        # 检查该区域内是否有黑色内容（数字）
        roi_gray = cv2.cvtColor(fpv[y:y+ch, x:x+cw], cv2.COLOR_BGR2GRAY)
        dark_px = np.sum(roi_gray < 80)
        dark_ratio = dark_px / roi_gray.size if roi_gray.size > 0 else 0
        # 白底黑字：暗像素占比 5%-60% 之间
        if dark_ratio < 0.03 or dark_ratio > 0.6:
            continue

        candidates.append((area, x, y, cw, ch, dark_ratio))

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [(x, y, cw, ch, dr) for _, x, y, cw, ch, dr in candidates]


def match_refs(roi: np.ndarray, refs: dict[int, np.ndarray]) -> tuple[int, float]:
    """将 ROI 与 image_num 参考图做模板匹配"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    best_idx, best_score = -1, 0.0

    for idx, ref in refs.items():
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        # 统一缩放到参考图尺寸
        query = cv2.resize(gray, (ref_gray.shape[1], ref_gray.shape[0]))
        score = cv2.matchTemplate(query, ref_gray, cv2.TM_CCOEFF_NORMED)[0][0]
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx, best_score


def main():
    refs = load_refs()
    print(f"已加载参考图: {sorted(refs.keys())}")
    for idx, ref in refs.items():
        print(f"  {idx}: {ref.shape}")

    cap = ScreenCapture()
    print("\n=== FPV 白底黑字诊断 ===")
    print("窗口显示FPV + 候选区域(绿框)。按数字键0-9测试匹配。q退出。\n")

    cv2.namedWindow("FPV White Detect", cv2.WINDOW_NORMAL)

    while True:
        frame = cap.grab()
        fpv = frame[
            FPV_REGION["top"]:FPV_REGION["top"] + FPV_REGION["height"],
            FPV_REGION["left"]:FPV_REGION["left"] + FPV_REGION["width"]
        ]
        display = fpv.copy()

        candidates = find_white_rects(fpv)
        roi = None
        for i, (x, y, cw, ch, dr) in enumerate(candidates[:10]):
            color = (0, 255, 0) if i == 0 else (0, 200, 200)
            cv2.rectangle(display, (x, y), (x + cw, y + ch), color, 2)
            cv2.putText(display, f"#{i} a={dr:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # 对最佳候选做匹配
        if candidates:
            x, y, cw, ch, _ = candidates[0]
            roi = fpv[y:y+ch, x:x+cw]
            idx, score = match_refs(roi, refs)
            cv2.putText(display, f"Best match: {idx} ({score:.3f})",
                        (x, y + ch + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # 显示白色mask
        hsv = cv2.cvtColor(fpv, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 80, 255]))
        white_viz = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)

        cv2.putText(display, f"Candidates: {len(candidates)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("FPV White Detect", display)
        if roi is not None and roi.size > 0:
            cv2.imshow("ROI #0", roi)
            cv2.imshow("White Mask", white_viz)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('9'):
            digit = key - ord('0')
            if roi is not None:
                score = cv2.matchTemplate(
                    cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(refs[digit], cv2.COLOR_BGR2GRAY),
                    cv2.TM_CCOEFF_NORMED
                )[0][0]
                print(f"  手动对比: ROI vs ref_{digit}: score={score:.4f}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
