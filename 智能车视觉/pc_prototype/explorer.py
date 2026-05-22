"""
探索控制流程 —— 遍历箱子/目的地，开车靠近后 FPV 识别并多帧投票记录

流程:
  1. 俯视图检测车/箱子/目的地/围墙
  2. 构建网格地图
  3. 对每个箱子: A* 导航靠近 → FPV 贴图识别 → 多帧投票
  4. 对每个目的地: A* 导航靠近 → FPV+YOLO 数字识别 → 多帧投票
  5. 记录 {网格坐标 → 类别/数字}
"""

import time
import math
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from capture import ScreenCapture
from config import TOPVIEW_REGION, FPV_REGION, GRID_CELL
from decision import GridMap, astar, WALL, EMPTY, BOX, DEST
from detect import (
    detect_car, detect_boxes, detect_destinations, detect_walls,
    classify_fpv_box, classify_fpv_dest,
)
from keyboard_control import KeyboardController, navigate_one_step
from map_loader import grid_to_pixel, pixel_to_grid


VOTE_FRAMES = 12          # 每个目标采集帧数
APPROACH_DIST = 2         # 靠近目标的网格距离
YOLO_CONF = 0.25

# YOLO 标签映射（与 live_view.py 一致）
_YOLO_LABEL_MAP = {9: 0, 6: 1, 4: 2, 0: 3, 8: 4, 1: 5, 7: 6, 5: 7, 3: 8, 2: 9}


@dataclass
class ExplorationResult:
    """探索阶段的结果"""
    box_class: dict[tuple[int, int], int] = field(default_factory=dict)    # 网格坐标 → 类别编号
    dest_num: dict[tuple[int, int], int] = field(default_factory=dict)     # 网格坐标 → 数字
    box_confidence: dict[tuple[int, int], float] = field(default_factory=dict)
    dest_confidence: dict[tuple[int, int], float] = field(default_factory=dict)


class Explorer:
    """探索控制器"""

    def __init__(self, yolo_model_path: str):
        self.cap = ScreenCapture()
        self.kc = KeyboardController()
        self.yolo = YOLO(yolo_model_path)
        self.result = ExplorationResult()
        self.grid_cell = 40  # 与 path_plan_view 一致

    def grab_topview(self):
        frame = self.cap.grab()
        if frame is None:
            return None
        return frame[
            TOPVIEW_REGION["top"]:TOPVIEW_REGION["top"] + TOPVIEW_REGION["height"],
            TOPVIEW_REGION["left"]:TOPVIEW_REGION["left"] + TOPVIEW_REGION["width"],
        ]

    def grab_fpv(self):
        frame = self.cap.grab()
        if frame is None:
            return None
        return frame[
            FPV_REGION["top"]:FPV_REGION["top"] + FPV_REGION["height"],
            FPV_REGION["left"]:FPV_REGION["left"] + FPV_REGION["width"],
        ]

    def build_grid(self, topview: np.ndarray) -> GridMap:
        """从俯视图构建网格地图"""
        car = detect_car(topview)
        boxes = detect_boxes(topview)
        dests = detect_destinations(topview)
        walls = detect_walls(topview)

        h, w = topview.shape[:2]
        gw = w // self.grid_cell
        gh = h // self.grid_cell

        gm = GridMap(gw, gh)

        # 围墙
        for gy in range(gh):
            for gx in range(gw):
                px = gx * self.grid_cell + self.grid_cell // 2
                py = gy * self.grid_cell + self.grid_cell // 2
                if 0 <= py < walls.shape[0] and 0 <= px < walls.shape[1]:
                    if walls[py, px] > 0:
                        gm.grid[gy][gx] = WALL

        # 箱子
        for bx, by in boxes:
            gx, gy = pixel_to_grid(bx, by, self.grid_cell)
            if gm.in_bounds(gx, gy) and gm.grid[gy][gx] != WALL:
                gm.boxes.add((gx, gy))
                gm.grid[gy][gx] = BOX

        # 目的地
        for dx, dy in dests:
            gx, gy = pixel_to_grid(dx, dy, self.grid_cell)
            if gm.in_bounds(gx, gy):
                gm.dests.add((gx, gy))
                if gm.grid[gy][gx] == EMPTY:
                    gm.grid[gy][gx] = DEST

        # 车
        if car:
            cx, cy, _ = car
            gx, gy = pixel_to_grid(cx, cy, self.grid_cell)
            if gm.in_bounds(gx, gy):
                gm.car_pos = (gx, gy)

        return gm

    def find_approach_pos(self, gm: GridMap, target: tuple[int, int]) -> tuple[int, int] | None:
        """
        找 target 附近的可行走位置（用于 FPV 靠近观察）。
        优先选离车最近的可通行邻格。
        """
        tx, ty = target
        candidates = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2),
                        (-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            nx, ny = tx + dx, ty + dy
            if gm.in_bounds(nx, ny) and gm.grid[ny][nx] in (EMPTY, DEST):
                if gm.car_pos:
                    dist = abs(nx - gm.car_pos[0]) + abs(ny - gm.car_pos[1])
                else:
                    dist = 0
                candidates.append((dist, (nx, ny)))

        candidates.sort(key=lambda x: x[0])
        for _, pos in candidates:
            if gm.car_pos:
                path = astar(gm, gm.car_pos, pos)
                if path:
                    return pos
        return None

    def wait_for_car_stable(self, timeout: float = 2.0) -> tuple | None:
        """等待车停稳，返回 (cx, cy, angle)"""
        t0 = time.time()
        last_pos = None
        stable_count = 0
        while time.time() - t0 < timeout:
            tv = self.grab_topview()
            if tv is None:
                continue
            car = detect_car(tv)
            if car is None:
                time.sleep(0.1)
                continue
            cx, cy, angle = car
            pos = (cx // self.grid_cell, cy // self.grid_cell)
            if pos == last_pos:
                stable_count += 1
                if stable_count >= 5:
                    return car
            else:
                stable_count = 0
                last_pos = pos
            time.sleep(0.1)
        return None

    def vote_box(self) -> tuple[int, float]:
        """多帧投票：FPV 贴图类别"""
        votes = Counter()
        best_conf = 0.0
        for _ in range(VOTE_FRAMES):
            fpv = self.grab_fpv()
            if fpv is None:
                continue
            cls_id, conf = classify_fpv_box(fpv)
            if cls_id >= 0:
                votes[cls_id] += 1
                if conf > best_conf:
                    best_conf = conf
            time.sleep(0.08)
        if votes:
            best_cls = votes.most_common(1)[0][0]
            return best_cls, best_conf
        return -1, 0.0

    def vote_dest(self) -> tuple[int, float]:
        """多帧投票：FPV 目的地数字（Hu矩 + YOLO 双重验证）"""
        hu_votes = Counter()
        yolo_votes = Counter()
        best_conf = 0.0
        for _ in range(VOTE_FRAMES):
            fpv = self.grab_fpv()
            if fpv is None:
                continue
            # Hu 矩
            digit, conf = classify_fpv_dest(fpv)
            if digit >= 0:
                hu_votes[digit] += 1
                if conf > best_conf:
                    best_conf = conf
            # YOLO
            results = self.yolo(fpv, conf=YOLO_CONF, verbose=False)
            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    real_digit = _YOLO_LABEL_MAP.get(cls, cls)
                    yolo_votes[real_digit] += 1
            time.sleep(0.08)

        # 优先 YOLO，其次 Hu 矩
        if yolo_votes:
            return yolo_votes.most_common(1)[0][0], best_conf
        if hu_votes:
            return hu_votes.most_common(1)[0][0], best_conf
        return -1, 0.0

    def navigate_to(self, gm: GridMap, target_grid: tuple[int, int]) -> bool:
        """
        导航车到目标网格位置。实时检测车位置，逐格移动。
        返回 True 到达，False 失败。
        """
        print(f"  导航到网格 {target_grid}...")
        max_steps = 60
        prev_grid = None
        stuck_count = 0

        for step in range(max_steps):
            tv = self.grab_topview()
            if tv is None:
                time.sleep(0.1)
                continue

            car = detect_car(tv)
            if car is None:
                print("    车未检测到")
                time.sleep(0.3)
                continue

            cx, cy, angle = car
            car_grid = pixel_to_grid(cx, cy, self.grid_cell)

            # 到达目标邻域
            if abs(car_grid[0] - target_grid[0]) <= 1 and abs(car_grid[1] - target_grid[1]) <= 1:
                print(f"    到达目标附近 {car_grid}")
                return True

            # 检测卡住
            if car_grid == prev_grid:
                stuck_count += 1
                if stuck_count > 10:
                    print("    可能卡住了，微调")
                    self.kc.turn_right(0.15)
                    self.kc.forward(0.1)
                    stuck_count = 0
            else:
                stuck_count = 0
            prev_grid = car_grid

            # 计算到目标的路径
            path = astar(gm, car_grid, target_grid)
            if not path or len(path) == 0:
                print("    无路径可达")
                return False

            # 取下一个格子，计算操作
            next_cell = path[0]
            next_px, next_py = grid_to_pixel(next_cell[0], next_cell[1], self.grid_cell)
            action = navigate_one_step((cx, cy), angle, (next_px, next_py))

            if action == "forward":
                self.kc.forward()
            elif action == "right":
                self.kc.turn_right()
            elif action == "left":
                self.kc.turn_left()
            else:
                self.kc.forward()

            time.sleep(0.04)

        print("    达到最大步数")
        return False

    def explore(self) -> ExplorationResult:
        """
        主探索流程：
        1. 扫描俯视图，定位所有箱子和目的地
        2. 依次开到每个前面做 FPV 识别
        3. 返回带类别/数字标注的结果
        """
        print("=" * 60)
        print("阶段一：探索扫描")
        print("=" * 60)

        # ---- 获取当前场景 ----
        tv = self.grab_topview()
        if tv is None:
            print("无法截取画面")
            return self.result

        gm = self.build_grid(tv)
        print(f"检测到: 车={gm.car_pos}, 箱子={len(gm.boxes)}个, 目的地={len(gm.dests)}个")
        print()

        # ---- 探索箱子 ----
        print("--- 探索箱子 ---")
        for i, box_pos in enumerate(sorted(gm.boxes)):
            print(f"[箱子 {i+1}/{len(gm.boxes)}] 网格位置 {box_pos}")
            approach = self.find_approach_pos(gm, box_pos)
            if approach is None:
                print(f"  无可用靠近位置，跳过")
                continue

            if not self.navigate_to(gm, approach):
                print(f"  导航失败，跳过")
                continue

            time.sleep(0.5)
            cls_id, conf = self.vote_box()
            if cls_id >= 0:
                self.result.box_class[box_pos] = cls_id
                self.result.box_confidence[box_pos] = conf
                print(f"  识别结果: class={cls_id}, conf={conf:.3f}")
            else:
                print(f"  未识别到贴图")

        # ---- 探索目的地 ----
        print()
        print("--- 探索目的地 ---")
        for i, dest_pos in enumerate(sorted(gm.dests)):
            print(f"[目的地 {i+1}/{len(gm.dests)}] 网格位置 {dest_pos}")
            approach = self.find_approach_pos(gm, dest_pos)
            if approach is None:
                print(f"  无可用靠近位置，跳过")
                continue

            if not self.navigate_to(gm, approach):
                print(f"  导航失败，跳过")
                continue

            time.sleep(0.5)
            digit, conf = self.vote_dest()
            if digit >= 0:
                self.result.dest_num[dest_pos] = digit
                self.result.dest_confidence[dest_pos] = conf
                print(f"  识别结果: digit={digit}, conf={conf:.3f}")
            else:
                print(f"  未识别到数字")

        print()
        print("探索完成")
        print(f"  箱子分类: {self.result.box_class}")
        print(f"  目的地数字: {self.result.dest_num}")
        return self.result


def run_exploration(yolo_model_path: str | None = None) -> ExplorationResult:
    """便捷入口：运行探索流程并返回结果"""
    import os
    if yolo_model_path is None:
        yolo_model_path = os.path.join(os.path.dirname(__file__), "..", "数字识别", "best.pt")
    explorer = Explorer(yolo_model_path)
    return explorer.explore()
