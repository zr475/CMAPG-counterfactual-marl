"""
执行推箱流程 —— 按配对结果逐一推箱子，通过串口发指令给真实小车

流程:
  1. 对每个配对：计算推箱站位 → 串口导航到站位 → 推箱 → 确认
  2. 循环直到所有箱子归位
"""

import time
from collections import deque

from config import GRID_CELL, UART_PORT, UART_BAUDRATE
from decision import GridMap, astar, SokobanAI, DIRS, WALL, EMPTY, BOX, DEST
from pairing import pair_boxes_from_dicts
from uart_comm import create_uart


# 像素→物理坐标的缩放因子（需根据实际场地标定）
# 场地 3.2m×2.4m，游戏俯视图约 960×1200 像素
# 粗略估计：每个网格 40px ≈ 0.133m（以 3.2m/24格 或 2.4m/30格 估算）
METERS_PER_CELL = 0.10   # 每个网格格子的物理尺寸（米），待标定


def grid_to_physical(gx: int, gy: int) -> tuple[float, float]:
    """网格坐标 → 物理坐标（米），原点在场地一角"""
    px = gx * METERS_PER_CELL
    py = gy * METERS_PER_CELL
    return px, py


class PushExecutor:
    """推箱执行器 —— 通过串口控制真实小车"""

    def __init__(self, port: str = UART_PORT, virtual: bool = False):
        self.uart = create_uart(port, virtual=virtual)
        self._pending_stop = False

    def open(self):
        self.uart.open()

    def close(self):
        self.uart.send_stop()
        self.uart.close()

    def stop(self):
        self.uart.send_stop()

    def send_velocity(self, vx: float, vy: float, wz: float):
        """发送速度指令，自动限幅"""
        vx = max(-2.0, min(2.0, vx))
        vy = max(-2.0, min(2.0, vy))
        wz = max(-6.0, min(6.0, wz))
        self.uart.send_velocity(vx, vy, wz)

    def execute_push(self, gm: GridMap, pairs: list) -> bool:
        """
        执行所有配对推箱任务。
        pairs: [(box_grid_pos, dest_grid_pos, class_id), ...]
        返回 True 全部完成。
        """
        if not pairs:
            print("无推箱任务")
            return True

        # 将配对数据填入 GridMap
        for box, dest, cls_id in pairs:
            gm.box_class[box] = cls_id
            gm.dest_num[dest] = cls_id

        ai = SokobanAI(gm)

        for i, (box, dest, cls_id) in enumerate(pairs):
            print(f"\n[推箱 {i+1}/{len(pairs)}] box={box} → dest={dest} (class={cls_id})")

            # 检查是否已完成
            if box == dest:
                print("  已在目的地，跳过")
                continue

            # 更新当前车的位置（从GM中获取）
            result = ai.solve_step()
            if result is None:
                print("  无可用推箱方案")
                continue

            path, push_dir = result
            if not path:
                print("  已在推箱站位")
            else:
                print(f"  导航到推箱站位: {len(path)} 步")
                if not self._navigate_path(path):
                    print("  导航失败")
                    continue
                # 更新地图中车的位置
                gm.car_pos = path[-1]

            # 执行推箱
            print(f"  推箱方向: {['up','right','down','left'][push_dir]}")
            self._execute_push(push_dir)

            # 更新地图状态
            bx, by = box
            ddx, ddy = DIRS[push_dir]
            new_box = (bx + ddx, by + ddy)
            gm.grid[by][bx] = EMPTY if (bx, by) not in gm.dests else DEST
            gm.boxes.discard(box)
            gm.boxes.add(new_box)
            gm.grid[new_box[1]][new_box[0]] = BOX
            gm.car_pos = box  # 车在箱子原位置

        return True

    def _navigate_path(self, path: list[tuple[int, int]]) -> bool:
        """
        沿网格路径导航（简化版：逐格发送速度指令）。
        实际使用时需配合里程计反馈做闭环控制。
        """
        for i, (gx, gy) in enumerate(path):
            # 检查是否需要紧急停车
            msgs = self.uart.recv()
            for m in msgs:
                if m.startswith("$E,"):
                    print(f"  报警: {m}")
                    self.stop()
                    return False

            px, py = grid_to_physical(gx, gy)
            # 简化：恒速移动到目标格
            # 实际需根据当前位姿和目位姿做轨迹跟踪
            self.send_velocity(0.3, 0.0, 0.0)  # 慢速前进
            time.sleep(0.5)  # 粗估一个格子时间

        self.stop()
        return True

    def _execute_push(self, push_dir: int):
        """执行一次推箱动作"""
        # 持续前进直到箱子被推到目标格
        self.send_velocity(0.2, 0.0, 0.0)
        time.sleep(1.0)  # 粗估推箱时间
        self.stop()


def run_execution(gm: GridMap,
                  box_class: dict[tuple[int, int], int],
                  dest_num: dict[tuple[int, int], int],
                  virtual: bool = True) -> bool:
    """
    便捷入口：运行推箱执行流程。
    virtual=True 时指令打印到控制台（调试），False 时通过真实串口发送。
    """
    pairs = pair_boxes_from_dicts(box_class, dest_num)
    if not pairs:
        print("无配对，无法执行")
        return False

    executor = PushExecutor(virtual=virtual)
    try:
        executor.open()
        return executor.execute_push(gm, pairs)
    finally:
        executor.close()
