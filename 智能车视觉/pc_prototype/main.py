"""
PC 原型验证 - 主程序
模式:
  - 检测模式(python main.py): 只显示检测结果，不发指令，用于调参
  - 运行模式(python main.py --run): 检测 + 决策 + 发送指令
"""

import sys
import os
import cv2
import numpy as np
import time

from capture import create_capture
from detect import (
    detect_car, detect_boxes, detect_destinations, detect_walls,
    get_class_matcher, get_num_matcher, classify_box, classify_destination,
    classify_fpv_box, classify_fpv_dest,
)
from decision import GridMap, SokobanAI, WALL, EMPTY, BOX, DEST, CAR
from uart_comm import create_uart
from config import TOPVIEW_REGION, FPV_REGION, BOX_THRESHOLD, DEST_THRESHOLD
from config import FPV_BOX_THRESHOLD, FPV_DEST_THRESHOLD, HIST_CONFIDENCE

# 屏幕区域 → 网格大小的映射比例（每个格子占多少像素，需要根据实际画面标定）
GRID_CELL = 40


def pixel_to_grid(px: int, py: int, origin_x: int = 0, origin_y: int = 0) -> tuple[int, int]:
    """像素坐标 → 网格坐标"""
    return ((px - origin_x) // GRID_CELL, (py - origin_y) // GRID_CELL)


def build_grid(wall_mask: np.ndarray, boxes: list, dests: list,
               car_pos: tuple, grid_w: int, grid_h: int,
               box_class: dict | None = None, dest_num: dict | None = None) -> GridMap:
    """从检测结果构建网格地图"""
    gm = GridMap(grid_w, grid_h)
    # 采样围墙
    for gy in range(grid_h):
        for gx in range(grid_w):
            px, py = gx * GRID_CELL + GRID_CELL // 2, gy * GRID_CELL + GRID_CELL // 2
            if 0 <= px < wall_mask.shape[1] and 0 <= py < wall_mask.shape[0]:
                if wall_mask[py, px] > 0:
                    gm.grid[gy][gx] = WALL
    # 箱子
    for bx, by in boxes:
        gx, gy = pixel_to_grid(bx, by)
        if gm.in_bounds(gx, gy):
            gm.boxes.add((gx, gy))
            gm.grid[gy][gx] = BOX
            if box_class and (bx, by) in box_class:
                gm.box_class[(gx, gy)] = box_class[(bx, by)]
    # 目的地
    for dx, dy in dests:
        gx, gy = pixel_to_grid(dx, dy)
        if gm.in_bounds(gx, gy):
            gm.dests.add((gx, gy))
            if gm.grid[gy][gx] == EMPTY:
                gm.grid[gy][gx] = DEST
            if dest_num and (dx, dy) in dest_num:
                gm.dest_num[(gx, gy)] = dest_num[(dx, dy)]
    # 车
    if car_pos:
        cx, cy = pixel_to_grid(car_pos[0], car_pos[1])
        if gm.in_bounds(cx, cy):
            gm.car_pos = (cx, cy)
    return gm


def draw_overlay(img: np.ndarray, car, boxes, dests, wall_mask,
                 box_class: dict | None = None, dest_num: dict | None = None):
    """在图上绘制检测结果"""
    display = img.copy()

    # 画围墙区域（半透明红）
    wall_overlay = np.zeros_like(display)
    wall_overlay[wall_mask > 0] = (0, 0, 200)
    display = cv2.addWeighted(display, 1.0, wall_overlay, 0.3, 0)

    # 画箱子
    for i, (bx, by) in enumerate(boxes):
        cv2.circle(display, (bx, by), 12, (0, 200, 255), 2)
        cls = box_class.get((bx, by), -1) if box_class else -1
        label = f"B{cls}" if cls >= 0 else "BOX"
        cv2.putText(display, label, (bx - 15, by - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    # 画目的地
    for dx, dy in dests:
        cv2.drawMarker(display, (dx, dy), (200, 0, 200),
                       cv2.MARKER_DIAMOND, 14, 2)
        num = dest_num.get((dx, dy), -1) if dest_num else -1
        label = f"D{num}" if num >= 0 else "DEST"
        cv2.putText(display, label, (dx - 17, dy - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 200), 1)

    # 画车
    if car:
        cx, cy, angle = car
        cv2.circle(display, (cx, cy), 15, (0, 255, 0), 2)
        tx = cx + int(20 * np.cos(np.radians(angle)))
        ty = cy + int(20 * np.sin(np.radians(angle)))
        cv2.arrowedLine(display, (cx, cy), (tx, ty), (0, 255, 0), 2)
        cv2.putText(display, f"CAR {angle:.0f}deg", (cx + 15, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 统计信息
    cv2.putText(display, f"Boxes:{len(boxes)} Dests:{len(dests)}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return display


def _crop_region(frame, region):
    """从全屏截图中裁剪指定区域"""
    r = region
    return frame[r["top"]:r["top"] + r["height"],
                 r["left"]:r["left"] + r["width"]].copy()


def run_detect_mode(cap):
    """检测模式：俯视图定位 + 第一人称分类"""
    print("[检测模式] 双视图: 俯视图(左)定位 + 第一人称(右)分类")
    print("  按 Q 退出, T 打印分类详情, C 手动触发分类")
    matcher_class = get_class_matcher()
    matcher_num = get_num_matcher()
    # 记录已分类的箱子和目的地
    known_box_class: dict[tuple, int] = {}
    known_dest_num: dict[tuple, int] = {}

    while True:
        frame = cap.grab()
        if frame is None:
            continue

        topview = _crop_region(frame, TOPVIEW_REGION)
        fpv = _crop_region(frame, FPV_REGION)

        # ---- 俯视图：位置检测 ----
        car = detect_car(topview)
        boxes = detect_boxes(topview)
        dests = detect_destinations(topview)
        walls = detect_walls(topview)

        # ---- 第一人称：HSV检测+直方图分类 ----
        cls_id, cls_score = classify_fpv_box(fpv)
        num_id, num_score = classify_fpv_dest(fpv)

        # ---- 绘制俯视图 ----
        top_disp = draw_overlay(topview, car, boxes, dests, walls)

        # ---- 绘制第一人称（叠加检测到的区域） ----
        fv_disp = fpv.copy()
        from detect import _hsv_mask
        box_mask = _hsv_mask(fpv, FPV_BOX_THRESHOLD)
        dest_mask = _hsv_mask(fpv, FPV_DEST_THRESHOLD)
        # 画箱子检测区域
        contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(fv_disp, (x, y), (x + w, y + h), (0, 200, 255), 2)
        # 画目的地检测区域
        contours, _ = cv2.findContours(dest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(fv_disp, (x, y), (x + w, y + h), (200, 0, 200), 2)

        cls_name = f"{cls_id}" if cls_id >= 0 else "?"
        num_name = f"{num_id}" if num_id >= 0 else "?"
        cv2.putText(fv_disp, f"Class: {cls_name} ({cls_score:.2f})",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(fv_disp, f"Num:   {num_name} ({num_score:.2f})",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)

        # 已记录的分类
        if known_box_class:
            lines = [f"B:{k}" for k in sorted(known_box_class.values())]
            cv2.putText(fv_disp, "Boxes: " + ",".join(lines),
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        if known_dest_num:
            lines = [f"D:{k}" for k in sorted(known_dest_num.values())]
            cv2.putText(fv_disp, "Dests: " + ",".join(lines),
                        (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 200), 1)

        # ---- 并排显示 ----
        th, tw = top_disp.shape[:2]
        fh, fw = fv_disp.shape[:2]
        target_h = max(th, fh, 300)
        if th != target_h:
            top_disp = cv2.resize(top_disp, (int(tw * target_h / th), target_h))
        if fh != target_h:
            fv_disp = cv2.resize(fv_disp, (int(fw * target_h / fh), target_h))
        combined = np.hstack([top_disp, fv_disp])

        cv2.putText(combined, f"Boxes:{len(boxes)} Dests:{len(dests)}",
                    (10, target_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("SmartCar Vision - Detect Mode", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # 手动触发：把当前 FPV 的分类结果记录下来
            if cls_id >= 0 and cls_score > 0.5:
                # 找离车最近的箱子，关联到这个类别
                if car and boxes:
                    cx_car, cy_car = car[0], car[1]
                    nearest = min(boxes, key=lambda b: (b[0]-cx_car)**2 + (b[1]-cy_car)**2)
                    known_box_class[nearest] = cls_id
                    print(f"[记录] 箱子 {nearest} → class {cls_id} (score={cls_score:.3f})")
            if num_id >= 0 and num_score > 0.5:
                if car and dests:
                    cx_car, cy_car = car[0], car[1]
                    nearest = min(dests, key=lambda d: (d[0]-cx_car)**2 + (d[1]-cy_car)**2)
                    known_dest_num[nearest] = num_id
                    print(f"[记录] 目的地 {nearest} → num {num_id} (score={num_score:.3f})")
        elif key == ord('t'):
            print(f"FPV center: class={cls_id}({cls_score:.3f}) num={num_id}({num_score:.3f})")
            print(f"Topview: {len(boxes)} boxes, {len(dests)} dests, car={'yes' if car else 'no'}")
            if car:
                print(f"  Car pos: ({car[0]},{car[1]}) angle={car[2]:.0f}deg")
            for bx, by in boxes:
                print(f"  Box at ({bx},{by})")
            for dx, dy in dests:
                print(f"  Dest at ({dx},{dy})")

    cv2.destroyAllWindows()


def run_detect_headless(cap, save_dir="captures"):
    """无窗口检测模式：不弹窗，定时保存截图到磁盘 + 终端打印事件
    解决单屏幕下"看游戏画面→py窗口被遮住"的问题
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[无窗口检测模式] 截图保存到 {save_dir}/, Ctrl+C 退出")
    print("  把车开到箱子面前→识别贴图  开到目的地面前→识别数字")

    frame_count = 0
    save_interval = 60
    last_cls, last_num = -1, -1

    # 跨帧累积已知分类
    known_boxes: dict[tuple, int] = {}
    known_dests: dict[tuple, int] = {}

    while True:
        frame = cap.grab()
        if frame is None:
            continue

        topview = _crop_region(frame, TOPVIEW_REGION)
        fpv = _crop_region(frame, FPV_REGION)

        car = detect_car(topview)
        boxes = detect_boxes(topview)
        dests = detect_destinations(topview)
        walls = detect_walls(topview)

        cls_id, cls_score = classify_fpv_box(fpv)
        num_id, num_score = classify_fpv_dest(fpv)

        # 高置信度时记录（只在新类别出现时打印）
        if cls_id >= 0 and cls_score > HIST_CONFIDENCE and cls_id != last_cls:
            # 关联到最近的车面对的箱子
            if car and boxes:
                cx_car, cy_car = car[0], car[1]
                nearest = min(boxes, key=lambda b: (b[0]-cx_car)**2 + (b[1]-cy_car)**2)
                if nearest not in known_boxes:
                    known_boxes[nearest] = cls_id
            print(f"  [帧{frame_count:05d}] >>> 箱子 class={cls_id} score={cls_score:.3f} (已记录 {len(known_boxes)} 个)")
            last_cls = cls_id

        if num_id >= 0 and num_score > HIST_CONFIDENCE and num_id != last_num:
            print(f"  [帧{frame_count:05d}] >>> 目的地 num={num_id} score={num_score:.3f} (已记录 {len(known_dests)} 个)")
            if car and dests:
                cx_car, cy_car = car[0], car[1]
                nearest = min(dests, key=lambda d: (d[0]-cx_car)**2 + (d[1]-cy_car)**2)
                if nearest not in known_dests:
                    known_dests[nearest] = num_id
            last_num = num_id

        # 定时打印分类进度
        if frame_count % 120 == 0 and frame_count > 0:
            print(f"  [进度] 已分类: 箱子 {len(known_boxes)}/?, 目的地 {len(known_dests)}/?")

        # 定时保存截图
        if frame_count % save_interval == 0:
            top_disp = draw_overlay(topview, car, boxes, dests, walls,
                                    known_boxes, known_dests)
            fv_disp = fpv.copy()
            from detect import _hsv_mask
            box_mask = _hsv_mask(fpv, FPV_BOX_THRESHOLD)
            dest_mask = _hsv_mask(fpv, FPV_DEST_THRESHOLD)
            contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(fv_disp, (x, y), (x + w, y + h), (0, 200, 255), 2)
            contours, _ = cv2.findContours(dest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(fv_disp, (x, y), (x + w, y + h), (200, 0, 200), 2)

            cv2.putText(fv_disp, f"C:{cls_id}({cls_score:.2f}) N:{num_id}({num_score:.2f})",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(fv_disp, f"Known B:{len(known_boxes)} D:{len(known_dests)}",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            th, tw = top_disp.shape[:2]
            fh, fw = fv_disp.shape[:2]
            target_h = max(th, fh, 300)
            if th != target_h:
                top_disp = cv2.resize(top_disp, (int(tw * target_h / th), target_h))
            if fh != target_h:
                fv_disp = cv2.resize(fv_disp, (int(fw * target_h / fh), target_h))
            combined = np.hstack([top_disp, fv_disp])

            path = os.path.join(save_dir, f"frame_{frame_count:05d}.png")
            cv2.imwrite(path, combined)

        frame_count += 1


def run_control_mode(cap, uart):
    """运行模式：双视图 — 俯视图定位 + 第一人称分类 + 配对推箱"""
    print("[运行模式] 双视图 + 扫描分类 + 配对推箱")
    uart.open()
    matcher_class = get_class_matcher()
    matcher_num = get_num_matcher()

    # 跨帧持久：已分类的箱子和目的地
    known_box_class: dict[tuple, int] = {}
    known_dest_num: dict[tuple, int] = {}
    # 扫描状态
    scan_phase = "box"  # "box" | "dest" | "push"
    scan_target_idx = 0

    while True:
        t0 = time.time()
        frame = cap.grab()
        if frame is None:
            continue

        topview = _crop_region(frame, TOPVIEW_REGION)
        fpv = _crop_region(frame, FPV_REGION)

        # ---- 俯视图定位 ----
        car = detect_car(topview)
        boxes = detect_boxes(topview)
        dests = detect_destinations(topview)
        walls = detect_walls(topview)

        # ---- 第一人称分类 ----
        cls_id, cls_score = classify_fpv_box(fpv)
        num_id, num_score = classify_fpv_dest(fpv)

        # ---- 自动记录分类（高置信度时） ----
        threshold = 0.7
        if car and boxes:
            cx_car, cy_car = car[0], car[1]
            # 找最近箱子
            unclassified = [b for b in boxes if b not in known_box_class]
            if unclassified:
                nearest = min(unclassified, key=lambda b: (b[0]-cx_car)**2 + (b[1]-cy_car)**2)
                dist = ((nearest[0]-cx_car)**2 + (nearest[1]-cy_car)**2) ** 0.5
                if dist < 80 and cls_id >= 0 and cls_score > threshold:
                    known_box_class[nearest] = cls_id
                    print(f"[自动] 箱子 {nearest} → class {cls_id} ({cls_score:.3f})")

        if car and dests:
            cx_car, cy_car = car[0], car[1]
            unclassified = [d for d in dests if d not in known_dest_num]
            if unclassified:
                nearest = min(unclassified, key=lambda d: (d[0]-cx_car)**2 + (d[1]-cy_car)**2)
                dist = ((nearest[0]-cx_car)**2 + (nearest[1]-cy_car)**2) ** 0.5
                if dist < 80 and num_id >= 0 and num_score > threshold:
                    known_dest_num[nearest] = num_id
                    print(f"[自动] 目的地 {nearest} → num {num_id} ({num_score:.3f})")

        # ---- 构建网格 + 决策 ----
        tv_h, tv_w = topview.shape[:2]
        grid_w, grid_h = tv_w // GRID_CELL, tv_h // GRID_CELL
        gm = build_grid(walls, boxes, dests,
                        (car[0], car[1]) if car else None,
                        grid_w, grid_h, known_box_class, known_dest_num)

        if gm.car_pos and gm.boxes and gm.dests:
            # 只在有足够分类信息时做配对推箱
            classified_boxes = sum(1 for b in gm.boxes if b in gm.box_class)
            classified_dests = sum(1 for d in gm.dests if d in gm.dest_num)
            if classified_boxes >= len(gm.boxes) and classified_dests >= len(gm.dests):
                ai = SokobanAI(gm)
                result = ai.solve_step()
                if result:
                    path, push_dir = result
                    if path:
                        cx_g, cy_g = gm.car_pos
                        tx, ty = path[0]
                        vx = 0.3 * (tx - cx_g)
                        vy = 0.3 * (ty - cy_g)
                        uart.send_velocity(vx, vy, 0)
            else:
                # 还没分类完，打印进度
                print(f"[扫描] 箱子:{classified_boxes}/{len(gm.boxes)} 目的地:{classified_dests}/{len(gm.dests)}")

        # ---- 显示 ----
        top_disp = draw_overlay(topview, car, boxes, dests, walls,
                               known_box_class, known_dest_num)
        fv_disp = fpv.copy()
        from detect import _hsv_mask
        box_mask = _hsv_mask(fpv, FPV_BOX_THRESHOLD)
        dest_mask = _hsv_mask(fpv, FPV_DEST_THRESHOLD)
        contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(fv_disp, (x, y), (x + w, y + h), (0, 200, 255), 2)
        contours, _ = cv2.findContours(dest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(fv_disp, (x, y), (x + w, y + h), (200, 0, 200), 2)
        cv2.putText(fv_disp, f"Class: {cls_id} ({cls_score:.2f})",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(fv_disp, f"Num:   {num_id} ({num_score:.2f})",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)
        cv2.putText(fv_disp, f"Known B:{len(known_box_class)} D:{len(known_dest_num)}",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        th, tw = top_disp.shape[:2]
        fh, fw = fv_disp.shape[:2]
        target_h = max(th, fh, 300)
        if th != target_h:
            top_disp = cv2.resize(top_disp, (int(tw * target_h / th), target_h))
        if fh != target_h:
            fv_disp = cv2.resize(fv_disp, (int(fw * target_h / fh), target_h))
        combined = np.hstack([top_disp, fv_disp])

        fps = 1.0 / (time.time() - t0) if (time.time() - t0) > 0 else 0
        cv2.putText(combined, f"FPS:{fps:.0f}",
                    (10, target_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("SmartCar Vision - Control Mode", combined)

        # MCU 上行帧
        for line in uart.recv():
            if line.startswith("$E,stall"):
                print("[!] 堵转报警！")
                uart.send_stop()
            elif line.startswith("$P,"):
                pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    uart.send_stop()
    uart.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    headless = "headless" in sys.argv
    run_mode = "--run" in sys.argv

    if "camera" in sys.argv:
        cap = create_capture("camera", index=0)
    else:
        cap = create_capture("screen")

    if headless:
        # 无窗口模式：不弹 OpenCV 窗口，截图存磁盘
        if run_mode:
            print("[无窗口运行模式] 暂不支持，请先用检测模式调参")
        else:
            run_detect_headless(cap)
    elif run_mode:
        uart = create_uart()
        run_control_mode(cap, uart)
    else:
        run_detect_mode(cap)
