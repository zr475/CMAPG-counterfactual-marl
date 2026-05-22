"""
实时检测预览 —— 小窗口始终置顶，显示 FPV + 检测结果
Q=退出  S=保存当前帧
"""
import cv2
import numpy as np
import os
from collections import Counter, deque
from ultralytics import YOLO

from capture import ScreenCapture
from config import FPV_REGION, FPV_BOX_THRESHOLD, HIST_CONFIDENCE
from detect import classify_fpv_box

# 贴图中文名
CLASS_NAMES = {
    0: "米老鼠", 1: "皮卡丘", 2: "海绵宝宝", 3: "喜羊羊",
    4: "唐老鸭", 5: "哪吒", 6: "大头儿子", 7: "猪猪侠",
    8: "葫芦兄弟", 9: "灰太狼",
}

# YOLO 数字识别模型路径
_YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "数字识别", "best.pt")
# 标签映射：模型输出 → 真实数字
_YOLO_LABEL_MAP = {9: 0, 6: 1, 4: 2, 0: 3, 8: 4, 1: 5, 7: 6, 5: 7, 3: 8, 2: 9}


def main():
    os.makedirs("captures", exist_ok=True)

    cap = ScreenCapture()
    yolo_model = YOLO(_YOLO_MODEL_PATH)

    cv2.namedWindow("Live Detect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Detect", 500, 650)
    cv2.setWindowProperty("Live Detect", cv2.WND_PROP_TOPMOST, 1)

    print("=== 实时检测预览 ===")
    print("Q=退出  S=保存当前帧到 captures/")
    print()

    frame_count = 0
    last_box = -1
    box_history = deque(maxlen=10)

    while True:
        frame = cap.grab()
        if frame is None:
            continue

        fpv = frame[
            FPV_REGION["top"]:FPV_REGION["top"] + FPV_REGION["height"],
            FPV_REGION["left"]:FPV_REGION["left"] + FPV_REGION["width"],
        ]
        display = fpv.copy()

        # ========== 贴图识别 ==========
        cls_id, cls_score = classify_fpv_box(fpv)
        box_history.append((cls_id, cls_score))

        box_votes = Counter(b for b, s in box_history if b >= 0 and s > HIST_CONFIDENCE)
        smooth_box = box_votes.most_common(1)[0][0] if box_votes else -1

        h, w = display.shape[:2]

        # 箱子检测框
        if cls_id >= 0 and cls_score > HIST_CONFIDENCE:
            mask = cv2.inRange(
                cv2.cvtColor(fpv, cv2.COLOR_BGR2HSV),
                np.array(FPV_BOX_THRESHOLD[0]),
                np.array(FPV_BOX_THRESHOLD[1]),
            )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, bw, bh = cv2.boundingRect(cnt)
                margin = 20
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + bw + margin)
                y2 = min(h, y + bh + margin)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 255), 3)

            if cls_id != last_box:
                name = CLASS_NAMES.get(cls_id, str(cls_id))
                print(f"  [帧{frame_count:05d}] 贴图 {name} (class={cls_id}) score={cls_score:.3f}")
                last_box = cls_id

        # ========== YOLO 数字识别 ==========
        yolo_results = yolo_model(fpv, conf=0.25, verbose=False)
        yolo_digits = []
        if yolo_results and len(yolo_results[0].boxes) > 0:
            for box in yolo_results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                real_digit = _YOLO_LABEL_MAP.get(cls, cls)
                yolo_digits.append((real_digit, conf))
                # 蓝色检测框
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (255, 80, 0), 2)
                cv2.putText(display, f"{real_digit}", (bx1, by1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)

        # ========== 信息栏 ==========
        bar_h = 130
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar[:] = (20, 20, 30)

        cv2.putText(bar, f"Frame: {frame_count}", (15, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)

        # 贴图结果
        if smooth_box >= 0:
            name = CLASS_NAMES.get(smooth_box, str(smooth_box))
            text = f"Sticker: {name} (class={smooth_box})"
            cv2.putText(bar, text, (15, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 80), 2)
        else:
            cv2.putText(bar, "Sticker: ---", (15, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (120, 120, 120), 1)

        # YOLO数字结果
        if yolo_digits:
            text = "Digit: " + ", ".join(str(d) for d, _ in yolo_digits)
            cv2.putText(bar, text, (15, 110),
                        cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 180, 0), 2)
        else:
            cv2.putText(bar, "Digit: ---", (15, 110),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (120, 120, 120), 1)

        # 提示
        cv2.putText(bar, "Q=Quit | S=Save", (w - 200, bar_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        combined = np.vstack([display, bar])
        cv2.imshow("Live Detect", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"captures/live_{frame_count:05d}.png"
            cv2.imencode(".png", combined)[1].tofile(path)
            print(f"  已保存: {path}")

        frame_count += 1

    cv2.destroyAllWindows()
    print("退出。")


if __name__ == "__main__":
    main()
