"""
目标检测与模板匹配模块
从游戏画面中识别：车模位置/朝向、箱子、目的地、围墙、炸弹
"""

import cv2
import numpy as np
from pathlib import Path
from config import (
    BOX_THRESHOLD, DEST_THRESHOLD, GREEN_CAR_THRESHOLD, CYAN_CAR_THRESHOLD,
    WALL_THRESHOLD, MIN_CONTOUR_AREA, MATCH_THRESHOLD,
    IMAGE_CLASS_DIR, IMAGE_NUM_DIR,
    FPV_BOX_THRESHOLD, FPV_DEST_THRESHOLD, HIST_CONFIDENCE,
)


def _hsv_mask(img: np.ndarray, threshold: list) -> np.ndarray:
    """对 BGR 图像做 HSV 颜色阈值分割，返回二值 mask"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(threshold[0])
    upper = np.array(threshold[1])
    return cv2.inRange(hsv, lower, upper)


def _find_centers(mask: np.ndarray, min_area: int = MIN_CONTOUR_AREA) -> list[tuple[int, int]]:
    """从 mask 中提取连通域中心坐标"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    return centers


def detect_car(img: np.ndarray):
    """
    检测车模位置和朝向
    返回: (x, y, angle_deg) 或 None
    x,y = 红蓝两色块的中点, angle = 从红到蓝的方向
    """
    red_mask = _hsv_mask(img, GREEN_CAR_THRESHOLD)
    blue_mask = _hsv_mask(img, CYAN_CAR_THRESHOLD)

    red_pts = _find_centers(red_mask)
    blue_pts = _find_centers(blue_mask)

    if not red_pts or not blue_pts:
        return None

    # 取面积最大的绿/青色块, 朝向从绿到青
    rx, ry = red_pts[0]
    bx, by = blue_pts[0]

    cx = (rx + bx) // 2
    cy = (ry + by) // 2
    angle = np.degrees(np.arctan2(ry - by, rx - bx))
    return (cx, cy, angle)


def detect_boxes(img: np.ndarray) -> list[tuple[int, int]]:
    """检测可移动箱子（橙黄色块），返回中心坐标列表"""
    mask = _hsv_mask(img, BOX_THRESHOLD)
    return _find_centers(mask)


def detect_destinations(img: np.ndarray) -> list[tuple[int, int]]:
    """检测目的地（紫红色块），返回中心坐标列表。俯视图像素分散，需膨胀连接"""
    mask = _hsv_mask(img, DEST_THRESHOLD)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    h, w = mask.shape[:2]
    max_area = (w * h) * 0.02
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    return centers


def detect_walls(img: np.ndarray):
    """检测围墙（黑灰色），返回二值 mask 供后续路径规划使用"""
    return _hsv_mask(img, WALL_THRESHOLD)


# ---- 模板匹配（图片分类模式） ----

class TemplateMatcher:
    """加载参考图，对 ROI 图像做模板匹配，返回最佳匹配类别"""

    def __init__(self, template_dir: Path, prefix: str = ""):
        """
        template_dir: 参考图目录
        prefix: 类别名前缀（如 "class" → class00, class01...）
        """
        self.templates: dict[int, np.ndarray] = {}
        self._load(template_dir)

    def _load(self, template_dir: Path):
        for img_path in sorted(template_dir.glob("*.jpg")):
            stem = img_path.stem
            try:
                idx = int(stem[:2])
            except ValueError:
                continue
            template = _imread(str(img_path))
            if template is not None:
                self.templates[idx] = template

    def match(self, roi: np.ndarray):
        """
        对 roi 与所有模板做归一化相关系数匹配
        返回 (best_index, best_score)，若最高分 < MATCH_THRESHOLD 则返回 (-1, 0)
        """
        best_idx, best_score = -1, 0.0
        for idx, tmpl in self.templates.items():
            if roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
                roi_resized = cv2.resize(roi, (tmpl.shape[1], tmpl.shape[0]))
                score = cv2.matchTemplate(roi_resized, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
            else:
                result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_score < MATCH_THRESHOLD:
            return -1, best_score
        return best_idx, best_score


# 全局匹配器，启动时加载一次
_class_matcher: TemplateMatcher | None = None
_num_matcher: TemplateMatcher | None = None


def get_class_matcher() -> TemplateMatcher:
    global _class_matcher
    if _class_matcher is None:
        # image_class/ 的每个子目录里有一张参考图，汇总加载
        _class_matcher = TemplateMatcher(IMAGE_CLASS_DIR)
        # 对于按子目录组织的结构，需要特殊处理
        _class_matcher = _load_class_templates()
    return _class_matcher


def get_num_matcher() -> TemplateMatcher:
    global _num_matcher
    if _num_matcher is None:
        _num_matcher = TemplateMatcher(IMAGE_NUM_DIR)
    return _num_matcher


def _imread(path: str) -> np.ndarray | None:
    """cv2.imread 替代版，兼容中文路径"""
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _load_class_templates() -> TemplateMatcher:
    """image_class/ 下是按 00角色名/xxx.jpg 组织的"""
    matcher = TemplateMatcher.__new__(TemplateMatcher)
    matcher.templates = {}
    for subdir in sorted(IMAGE_CLASS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        idx = int(subdir.name[:2])
        for img_path in subdir.glob("*.jpg"):
            template = _imread(str(img_path))
            if template is not None:
                matcher.templates[idx] = template
            break
    return matcher


# ==================== HSV 直方图分类器 ====================

class HistogramMatcher:
    """HSV + LAB + 灰度 三特征直方图分类器"""

    def __init__(self, template_dir: Path, load_func=None):
        self.color_hists: dict[int, np.ndarray] = {}
        self.lab_hists: dict[int, np.ndarray] = {}
        self.gray_hists: dict[int, np.ndarray] = {}
        self._load(template_dir, load_func)

    @staticmethod
    def _enhance(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _compute_color_hist(self, img: np.ndarray) -> np.ndarray:
        img = self._enhance(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _compute_lab_hist(self, img: np.ndarray) -> np.ndarray:
        """LAB A-B 直方图"""
        img = self._enhance(img)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hist = cv2.calcHist([lab], [1, 2], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _compute_gray_hist(self, img: np.ndarray) -> np.ndarray:
        """灰度直方图 — 区分明暗分布"""
        img = self._enhance(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _load(self, template_dir: Path, load_func):
        for img_path in sorted(template_dir.glob("*.jpg")):
            stem = img_path.stem
            try:
                idx = int(stem[:2])
            except ValueError:
                continue
            img = _imread(str(img_path))
            if img is not None:
                self.color_hists[idx] = self._compute_color_hist(img)
                self.lab_hists[idx] = self._compute_lab_hist(img)
                self.gray_hists[idx] = self._compute_gray_hist(img)

    def match(self, roi: np.ndarray) -> tuple[int, float]:
        roi_hsv = self._compute_color_hist(roi)
        roi_lab = self._compute_lab_hist(roi)
        roi_gray = self._compute_gray_hist(roi)
        best_idx, best_score = -1, 0.0
        for idx in self.color_hists:
            hs = cv2.compareHist(self.color_hists[idx], roi_hsv, cv2.HISTCMP_CORREL)
            hi = cv2.compareHist(self.color_hists[idx], roi_hsv, cv2.HISTCMP_INTERSECT)
            ls = cv2.compareHist(self.lab_hists[idx], roi_lab, cv2.HISTCMP_CORREL)
            gs = cv2.compareHist(self.gray_hists[idx], roi_gray, cv2.HISTCMP_CORREL)
            score = max(hs, hi * 2.0, ls, gs)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx, best_score


def _load_class_hists() -> HistogramMatcher:
    """加载卡通角色参考图的 HSV 直方图"""
    matcher = HistogramMatcher.__new__(HistogramMatcher)
    matcher.color_hists = {}
    matcher.lab_hists = {}
    matcher.gray_hists = {}
    for subdir in sorted(IMAGE_CLASS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        idx = int(subdir.name[:2])
        for img_path in subdir.glob("*.jpg"):
            img = _imread(str(img_path))
            if img is not None:
                matcher.color_hists[idx] = matcher._compute_color_hist(img)
                matcher.lab_hists[idx] = matcher._compute_lab_hist(img)
                matcher.gray_hists[idx] = matcher._compute_gray_hist(img)
            break
    return matcher


# 全局直方图匹配器
_class_hist_matcher: HistogramMatcher | None = None
_num_hist_matcher: HistogramMatcher | None = None


def get_class_hist_matcher() -> HistogramMatcher:
    global _class_hist_matcher
    if _class_hist_matcher is None:
        _class_hist_matcher = _load_class_hists()
    return _class_hist_matcher


def get_num_hist_matcher() -> HistogramMatcher:
    global _num_hist_matcher
    if _num_hist_matcher is None:
        _num_hist_matcher = HistogramMatcher(IMAGE_NUM_DIR)
    return _num_hist_matcher


def classify_fpv_box(fpv: np.ndarray) -> tuple[int, float]:
    """在第一人称画面中检测箱子（3D渲染，用宽阈值）并做直方图分类"""
    mask = _hsv_mask(fpv, FPV_BOX_THRESHOLD)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1, 0.0
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    # 大幅扩展 ROI：颜色检测可能只命中贴图局部（如灰太狼的帽子），
    # 需扩到包含整张贴图才能提取有效特征
    margin = 20
    fy, fx = fpv.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(fx, x + w + margin)
    y2 = min(fy, y + h + margin)
    roi = fpv[y1:y2, x1:x2]
    if roi.size < 50:
        return -1, 0.0
    return get_class_hist_matcher().match(roi)


# ==================== Hu 矩数字分类器 ====================

def _extract_hu_moments(binary: np.ndarray) -> np.ndarray | None:
    """从二值图中提取最大轮廓的 log-transformed Hu 矩"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 15:
        return None
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments)
    # log transform for stable comparison
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.flatten()


class HuMomentsMatcher:
    """基于 Hu 矩的数字分类器 —— 对旋转/缩放/平移不变，对透视变形有一定容忍度"""

    def __init__(self, template_dir: Path):
        self.hu_ref: dict[int, np.ndarray] = {}
        self._load(template_dir)

    def _load(self, template_dir: Path):
        for img_path in sorted(template_dir.glob("*.jpg")):
            stem = img_path.stem
            try:
                idx = int(stem[:2])
            except ValueError:
                continue
            img = _imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 参考图：黑色数字/白色背景，取暗部做轮廓
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            hu = _extract_hu_moments(binary)
            if hu is not None:
                self.hu_ref[idx] = hu

    def match(self, roi: np.ndarray) -> tuple[int, float]:
        """
        从 ROI 中提取白色数字（高V低S），算 Hu 矩，与参考数字比对。
        返回 (digit_id, confidence) — confidence 越高越好。
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 白色数字：低饱和度 + 高亮度
        lower = np.array([0, 0, 170])
        upper = np.array([180, 70, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # 形态学去噪
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        hu = _extract_hu_moments(mask)
        if hu is None:
            return -1, 0.0

        best_idx, best_dist = -1, float("inf")
        for idx, ref_hu in self.hu_ref.items():
            dist = np.sum(np.abs(hu - ref_hu))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        # 距离 → 置信度映射。良好匹配 dist<5，差匹配 dist>15
        confidence = 1.0 / (1.0 + best_dist)
        if best_dist > 8.0:
            return -1, confidence
        return best_idx, confidence


# ==================== FPV 目的地数字识别 ====================
# FPV 画面向右旋转了90度（侧向显示），需旋转回来再处理
# 目的地标志牌：白底（低饱和度）+ 黑色数字，位于地面区域

# 全局缓存：image_num 参考图的数字轮廓（Hu矩匹配用）
_ref_digit_contours: dict[int, np.ndarray] | None = None


def _get_ref_digit_contours() -> dict[int, np.ndarray]:
    """加载 image_num/ 参考图，提取每个数字的最大轮廓（黑字白底→反转后提取）"""
    global _ref_digit_contours
    if _ref_digit_contours is not None:
        return _ref_digit_contours

    _ref_digit_contours = {}
    for img_path in sorted(IMAGE_NUM_DIR.glob("*.jpg")):
        try:
            idx = int(img_path.stem[:2])
        except ValueError:
            continue
        img = _imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            _ref_digit_contours[idx] = max(contours, key=cv2.contourArea)
    return _ref_digit_contours


def _find_dest_sign_rotated(fpv_rotated: np.ndarray) -> tuple | None:
    """
    在旋转后的FPV中找目的地标志牌。
    FPV已旋转90°CCW恢复为水平视角：天空在上，地面在下。
    标志牌特征：地面区域中的白底方块（低S高V），内含暗像素（数字）。
    返回 (x, y, w, h) 或 None。
    """
    hsv = cv2.cvtColor(fpv_rotated, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(fpv_rotated, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 地面区域（下方70%），找低饱和度+高亮度区域（白底标志牌）
    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))
    white_mask[:int(h * 0.3), :] = 0  # 排除天空

    kernel = np.ones((7, 7), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > (w * h) * 0.04:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        ar = cw / ch if ch > 0 else 0
        if ar < 0.3 or ar > 3.0:
            continue
        # 必须包含暗像素（数字笔画）
        roi_g = gray[y:y + ch, x:x + cw]
        dark_ratio = np.sum(roi_g < 80) / roi_g.size if roi_g.size > 0 else 0
        if dark_ratio < 0.03:
            continue
        if best is None or area > best[0]:
            best = (area, x, y, cw, ch)

    if best is None:
        return None
    return best[1:]  # (x, y, w, h)


def _match_digit_hu(roi_gray: np.ndarray) -> tuple[int, float]:
    """
    对标志牌ROI提取数字轮廓，用Hu矩与image_num参考轮廓匹配。
    返回 (digit_id, confidence)，confidence 越高越好，<0 表示不匹配。
    """
    ref_contours = _get_ref_digit_contours()
    if not ref_contours:
        return -1, 0.0

    # 固定阈值二值化提取黑字
    _, binary = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1, 0.0

    # 取面积最大的轮廓（数字主体）
    digit_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(digit_cnt) < 30:
        return -1, 0.0

    best_idx, best_dist = -1, float("inf")
    for idx, ref_cnt in ref_contours.items():
        dist = cv2.matchShapes(digit_cnt, ref_cnt, cv2.CONTOURS_MATCH_I2, 0)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    # 距离→置信度映射。dist<2.0 可接受，dist<1.0 良好
    confidence = max(0.0, 1.0 - best_dist / 3.0)
    if best_dist > 3.0:
        return -1, confidence
    return best_idx, confidence


def classify_fpv_dest(fpv: np.ndarray) -> tuple[int, float]:
    """
    识别FPV画面中的目的地数字。
    流程：旋转FPV恢复视角 → 在地面区域找白底标志牌 → 提取数字轮廓 → Hu矩匹配。
    返回 (digit_id, confidence)，未检测到返回 (-1, 0.0)。
    """
    # 旋转90°CCW恢复正常视角（FPV侧向显示）
    fpv_rot = cv2.rotate(fpv, cv2.ROTATE_90_COUNTERCLOCKWISE)

    loc = _find_dest_sign_rotated(fpv_rot)
    if loc is None:
        return -1, 0.0

    x, y, w, h = loc
    roi = cv2.cvtColor(fpv_rot[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    if roi.size < 50:
        return -1, 0.0

    return _match_digit_hu(roi)


def classify_box(frame: np.ndarray, box_pos: tuple[int, int],
                 class_matcher: TemplateMatcher, half: int = 25) -> tuple[int, float]:
    """截取箱子 ROI 做模板匹配，返回 (class_id, score)，失败返回 (-1, 0.0)"""
    x, y = box_pos
    h, w = frame.shape[:2]
    roi = frame[max(0, y - half):min(h, y + half), max(0, x - half):min(w, x + half)]
    if roi.size == 0:
        return -1, 0.0
    return class_matcher.match(roi)


def classify_destination(frame: np.ndarray, dest_pos: tuple[int, int],
                         num_matcher: TemplateMatcher, half: int = 25) -> tuple[int, float]:
    """截取目的地 ROI 做数字模板匹配，返回 (num_id, score)，失败返回 (-1, 0.0)"""
    x, y = dest_pos
    h, w = frame.shape[:2]
    roi = frame[max(0, y - half):min(h, y + half), max(0, x - half):min(w, x + half)]
    if roi.size == 0:
        return -1, 0.0
    return num_matcher.match(roi)
