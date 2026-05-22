import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageGrab

model = YOLO('D:/yolo_number2/runs/detect/runs/train/number_model_v1/weights/best.pt')

# 标签映射：模型输出 → 真实数字
# 模型: 0  1  2  3  4  5  6  7  8  9
# 真实: 9  6  4  0  8  1  7  5  3  2
LABEL_MAP = {0: 9, 1: 6, 2: 4, 3: 0, 4: 8, 5: 1, 6: 7, 7: 5, 8: 3, 9: 2}

# 获取屏幕分辨率，只截右半边
screen_w, screen_h = ImageGrab.grab().size
half_w = screen_w // 2
capture_bbox = (half_w, 0, screen_w, screen_h)  # 左, 上, 右, 下

print(f"屏幕: {screen_w}x{screen_h}，只识别右半边")
print("按 Q 退出...")

cv2.namedWindow('YOLO Detect', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO Detect', 480, 360)
cv2.moveWindow('YOLO Detect', 0, 0)  # 窗口放左上角

while True:
    img = ImageGrab.grab(bbox=capture_bbox)
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model(frame, conf=0.05, verbose=False)

    # 修正标签映射：用 names 字典欺骗 plot() 显示正确数字
    results[0].names = {9: '0', 6: '1', 4: '2', 0: '3', 8: '4',
                        1: '5', 7: '6', 5: '7', 3: '8', 2: '9'}

    annotated = results[0].plot()

    cv2.imshow('YOLO Detect', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("已退出")
