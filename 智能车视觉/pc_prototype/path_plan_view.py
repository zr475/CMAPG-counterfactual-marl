"""
路径规划预览 —— 俯视图（左半边）+ 网格化 + A* 寻路 + 推箱策略
Q=退出  S=保存当前帧
"""
import cv2
import numpy as np
import os
import time
from collections import deque

from capture import ScreenCapture
from config import TOPVIEW_REGION
from detect import detect_car, detect_boxes, detect_destinations, detect_walls
from decision import GridMap, astar, SokobanAI, WALL, EMPTY, BOX, DEST, DIRS

GRID_CELL = 40
# 已知的箱子类别和目的地数字（可从 FPV 识别获得，这里先给个模拟/手动映射）
# 实际使用时，需从 live_view 或 FPV 识别结果中获取
known_box_class: dict[tuple, int] = {}
known_dest_num: dict[tuple, int] = {}


def pixel_to_grid(px: int, py: int) -> tuple[int, int]:
    return px // GRID_CELL, py // GRID_CELL


def build_grid(wall_mask: np.ndarray, boxes: list, dests: list,
               car_pos: tuple | None, grid_w: int, grid_h: int) -> GridMap:
    gm = GridMap(grid_w, grid_h)
    # 采样围墙
    for gy in range(grid_h):
        for gx in range(grid_w):
            px = gx * GRID_CELL + GRID_CELL // 2
            py = gy * GRID_CELL + GRID_CELL // 2
            if 0 <= py < wall_mask.shape[0] and 0 <= px < wall_mask.shape[1]:
                if wall_mask[py, px] > 0:
                    gm.grid[gy][gx] = WALL
    # 箱子
    for bx, by in boxes:
        gx, gy = pixel_to_grid(bx, by)
        if gm.in_bounds(gx, gy) and gm.grid[gy][gx] != WALL:
            gm.boxes.add((gx, gy))
            gm.grid[gy][gx] = BOX
    # 目的地
    for dx, dy in dests:
        gx, gy = pixel_to_grid(dx, dy)
        if gm.in_bounds(gx, gy):
            gm.dests.add((gx, gy))
            if gm.grid[gy][gx] == EMPTY:
                gm.grid[gy][gx] = DEST
    # 车
    if car_pos:
        cx, cy = pixel_to_grid(car_pos[0], car_pos[1])
        if gm.in_bounds(cx, cy):
            gm.car_pos = (cx, cy)
    return gm


def draw_grid(display: np.ndarray, grid_w: int, grid_h: int) -> np.ndarray:
    """画网格线"""
    h, w = display.shape[:2]
    for gx in range(0, grid_w):
        px = gx * GRID_CELL
        cv2.line(display, (px, 0), (px, h), (50, 50, 50), 1)
    for gy in range(0, grid_h):
        py = gy * GRID_CELL
        cv2.line(display, (0, py), (w, py), (50, 50, 50), 1)
    return display


def draw_path(display: np.ndarray, path: list[tuple[int, int]],
              color: tuple = (0, 255, 255)):
    """在图上画 A* 路径"""
    if len(path) < 2:
        return display
    for i in range(len(path) - 1):
        x1 = path[i][0] * GRID_CELL + GRID_CELL // 2
        y1 = path[i][1] * GRID_CELL + GRID_CELL // 2
        x2 = path[i + 1][0] * GRID_CELL + GRID_CELL // 2
        y2 = path[i + 1][1] * GRID_CELL + GRID_CELL // 2
        cv2.line(display, (x1, y1), (x2, y2), color, 2)
        cv2.circle(display, (x1, y1), 3, color, -1)
    # 画最后一个点
    lx = path[-1][0] * GRID_CELL + GRID_CELL // 2
    ly = path[-1][1] * GRID_CELL + GRID_CELL // 2
    cv2.circle(display, (lx, ly), 4, (0, 0, 255), -1)
    return display


def main():
    os.makedirs("captures", exist_ok=True)

    cap = ScreenCapture()

    cv2.namedWindow("Path Plan", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Path Plan", 600, 800)
    cv2.setWindowProperty("Path Plan", cv2.WND_PROP_TOPMOST, 1)

    print("=== 路径规划预览 ===")
    print("Q=退出  S=保存当前帧到 captures/")
    print()

    frame_count = 0
    path_history = deque(maxlen=3)

    while True:
        frame = cap.grab()
        if frame is None:
            continue

        topview = frame[
            TOPVIEW_REGION["top"]:TOPVIEW_REGION["top"] + TOPVIEW_REGION["height"],
            TOPVIEW_REGION["left"]:TOPVIEW_REGION["left"] + TOPVIEW_REGION["width"],
        ]
        display = topview.copy()

        # 检测
        car = detect_car(topview)
        boxes = detect_boxes(topview)
        dests = detect_destinations(topview)
        walls = detect_walls(topview)

        # 构建网格
        tv_h, tv_w = topview.shape[:2]
        grid_w = tv_w // GRID_CELL
        grid_h = tv_h // GRID_CELL
        gm = build_grid(walls, boxes, dests,
                        (car[0], car[1]) if car else None,
                        grid_w, grid_h)

        # 画网格
        draw_grid(display, grid_w, grid_h)

        # 画围墙（半透明红）
        wall_overlay = np.zeros_like(display)
        wall_overlay[walls > 0] = (0, 0, 200)
        display = cv2.addWeighted(display, 1.0, wall_overlay, 0.3, 0)

        # 画目的地（紫色菱形）
        for dx, dy in dests:
            gx = dx // GRID_CELL * GRID_CELL + GRID_CELL // 2
            gy = dy // GRID_CELL * GRID_CELL + GRID_CELL // 2
            cv2.drawMarker(display, (gx, gy), (200, 0, 200),
                           cv2.MARKER_DIAMOND, 12, 2)

        # 画箱子（橙色圆）
        for bx, by in boxes:
            gx = bx // GRID_CELL * GRID_CELL + GRID_CELL // 2
            gy = by // GRID_CELL * GRID_CELL + GRID_CELL // 2
            cv2.circle(display, (gx, gy), 12, (0, 200, 255), 2)

        # 画车（绿色箭头）
        car_path_target = None
        if car:
            cx, cy, angle = car
            gx = cx // GRID_CELL * GRID_CELL + GRID_CELL // 2
            gy = cy // GRID_CELL * GRID_CELL + GRID_CELL // 2
            cv2.circle(display, (cx, cy), 14, (0, 255, 0), 2)
            tx = cx + int(20 * np.cos(np.radians(angle)))
            ty = cy + int(20 * np.sin(np.radians(angle)))
            cv2.arrowedLine(display, (cx, cy), (tx, ty), (0, 255, 0), 2)

        # A* 寻路 + 推箱策略
        path_info = ""
        planned_path = None
        if gm.car_pos and gm.boxes:
            # 如果有分类数据，用 SokobanAI 做配对推箱
            has_class_data = bool(gm.box_class) and bool(gm.dests)
            if has_class_data and gm.dests:
                ai = SokobanAI(gm)
                result = ai.solve_step()
                if result:
                    planned_path, push_dir = result
                    if planned_path:
                        draw_path(display, planned_path)
                        last_node = planned_path[-1]
                        bx = last_node[0] * GRID_CELL + GRID_CELL // 2
                        by = last_node[1] * GRID_CELL + GRID_CELL // 2
                        ddx, ddy = DIRS[push_dir]
                        ex = bx + ddx * GRID_CELL
                        ey = by + ddy * GRID_CELL
                        cv2.arrowedLine(display, (bx, by), (ex, ey),
                                        (255, 255, 0), 3, tipLength=0.3)
                        path_info = f"Path: {len(planned_path)} steps  Push: {['up','right','down','left'][push_dir]}"
                    else:
                        path_info = "At target, ready to push"
                else:
                    path_info = "All boxes on targets"
            else:
                # 无分类数据：简单 A* 到最近箱子（用于调试俯视图检测）
                nearest = min(gm.boxes, key=lambda b: abs(b[0]-gm.car_pos[0]) + abs(b[1]-gm.car_pos[1]))
                planned_path = astar(gm, gm.car_pos, nearest)
                if planned_path:
                    draw_path(display, planned_path, color=(0, 255, 255))
                    path_info = f"A* to box: {len(planned_path)} steps"
                else:
                    path_info = "No path to box (blocked)"
        elif not gm.car_pos:
            path_info = "Car not found"
        elif not gm.boxes:
            path_info = "No boxes found"

        # ========== 信息栏 ==========
        h, w = display.shape[:2]
        bar_h = 160
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        bar[:] = (20, 20, 30)

        cv2.putText(bar, f"Frame: {frame_count}", (15, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(bar, f"Car: {'OK' if car else '--'}  Boxes: {len(boxes)}  Dests: {len(dests)}",
                    (15, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 80), 2)
        cv2.putText(bar, path_info, (15, 115),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 200, 0), 2)

        cv2.putText(bar, "Q=Quit | S=Save", (w - 200, bar_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

        combined = np.vstack([display, bar])
        cv2.imshow("Path Plan", combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"captures/pathplan_{frame_count:05d}.png"
            cv2.imencode(".png", combined)[1].tofile(path)
            print(f"  已保存: {path}")

        frame_count += 1

    cv2.destroyAllWindows()
    print("退出。")


if __name__ == "__main__":
    main()
