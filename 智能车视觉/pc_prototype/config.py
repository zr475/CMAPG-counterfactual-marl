"""
PC原型验证 - 配置文件
所有颜色值为 OpenCV HSV 范围 (H:0-180, S:0-255, V:0-255)
"""

from pathlib import Path

# ---- 路径 ----
BASE_DIR = Path(__file__).parent.parent
VR_DIR = BASE_DIR / "SmartCar_VR_V1.6"
IMAGE_CLASS_DIR = VR_DIR / "image_class"
IMAGE_NUM_DIR = VR_DIR / "image_num"
MAP_DIR = VR_DIR / "map_file"

# ---- 屏幕截图区域（推箱子游戏窗口区域，需根据实际显示器调整） ----
CAPTURE_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1200}

# 俯视图（全局地图）区域 —— 用于定位车/箱子/目的地/围墙
TOPVIEW_REGION = {"top": 0, "left": 0, "width": 960, "height": 1200}

# 第一人称视角区域 —— 靠近箱子/目的地时用于识别贴图和数字
FPV_REGION = {"top": 0, "left": 960, "width": 960, "height": 1200}

# ---- 游戏元素 HSV 颜色阈值 ----
# 黄色箱子 (实际采样 HSV≈30, 217, 255)
BOX_THRESHOLD = [(18, 150, 180), (42, 255, 255)]

# 紫红色目的地 (实际采样 H≈150, 像素分散需膨胀连接)
DEST_THRESHOLD = [(130, 60, 60), (170, 255, 255)]

# 绿色车模色块
GREEN_CAR_THRESHOLD = [(50, 100, 100), (85, 255, 255)]

# 青色车模色块
CYAN_CAR_THRESHOLD = [(85, 100, 100), (105, 255, 255)]

# 围墙 黑白灰混合（低饱和度 = 无彩色，排除蓝色地面）
WALL_THRESHOLD = [(0, 0, 0), (180, 50, 255)]

# FPV（第一人称）专用阈值 —— 3D 渲染下有光照偏移，范围更宽
FPV_BOX_THRESHOLD = [(5, 70, 140), (45, 255, 255)]
FPV_DEST_THRESHOLD = [(100, 60, 80), (170, 255, 255)]  # H扩到100-170覆盖3D蓝色偏移

# 直方图分类置信度阈值
HIST_CONFIDENCE = 0.35

# ---- 目标检测参数 ----
MIN_CONTOUR_AREA = 50        # 最小轮廓面积（过滤噪点）
BOX_AREA_RANGE = (100, 2000)  # 箱子轮廓面积范围
DEST_AREA_RANGE = (80, 1500)  # 目的地轮廓面积范围

# ---- 模板匹配阈值 ----
MATCH_THRESHOLD = 0.6  # matchTemplate 相关系数阈值（低于此值视为不匹配）

# ---- 网格参数 ----
GRID_CELL = 40  # 俯视图每个网格的像素大小

# ---- 通信 ----
UART_PORT = "COM3"
UART_BAUDRATE = 115200

# ---- 运动参数 ----
SPEED_FORWARD = 0.5     # 前进速度 m/s
SPEED_ROTATE = 0.8      # 旋转速度 rad/s
SPEED_PUSH = 0.3        # 推箱子时的速度（慢一些）
