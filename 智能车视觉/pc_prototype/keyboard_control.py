"""
键盘模拟控制 —— 使用 keyboard 库的驱动级注入（兼容 DirectInput 游戏）

方向键: ↑=前进  ↓=后退  ←=左转  →=右转
"""

import time
import math
import keyboard


# 按键持续时间（秒）
TURN_PRESS = 0.08
FORWARD_PRESS = 0.15

ANGLE_TOLERANCE = 15  # 角度容差（度）


class KeyboardController:
    """keyboard 库驱动级按键模拟"""

    def _press(self, key: str, duration: float):
        keyboard.press(key)
        time.sleep(duration)
        keyboard.release(key)

    def turn_left(self, duration: float = TURN_PRESS):
        self._press("left", duration)

    def turn_right(self, duration: float = TURN_PRESS):
        self._press("right", duration)

    def forward(self, duration: float = FORWARD_PRESS):
        self._press("up", duration)

    def backward(self, duration: float = FORWARD_PRESS):
        self._press("down", duration)


def angle_diff(deg1: float, deg2: float) -> float:
    """两个角度（度）的最小差，范围 [-180, 180)"""
    d = (deg2 - deg1) % 360
    if d > 180:
        d -= 360
    return d


def navigate_one_step(car_pos: tuple[int, int], car_angle: float,
                      target: tuple[int, int]) -> str | None:
    """
    计算从车当前位置到目标格子的单步操作。
    返回 "left" / "right" / "forward" / None（已对齐）。
    """
    tx, ty = target
    cx, cy = car_pos
    dx = tx - cx
    dy = ty - cy

    if dx == 0 and dy == 0:
        return None

    target_angle = math.degrees(math.atan2(dy, dx))
    diff = angle_diff(car_angle, target_angle)

    if abs(diff) <= ANGLE_TOLERANCE:
        return "forward"
    elif diff > 0:
        return "right"
    else:
        return "left"


def navigate_path(path: list[tuple[int, int]],
                  get_car_state,
                  kc: KeyboardController,
                  step_callback=None) -> bool:
    """
    沿 A* 路径逐格导航。
    返回 True 到达终点，False 中断。
    """
    for i, target in enumerate(path):
        attempts = 0
        max_attempts = 8

        while attempts < max_attempts:
            state = get_car_state()
            if state is None:
                print("  [navigate] 车丢失，等待重检测...")
                time.sleep(0.3)
                attempts += 1
                continue

            cx, cy, angle = state
            action = navigate_one_step((cx, cy), angle, target)
            if action is None:
                break

            if action == "forward":
                kc.forward()
                break
            elif action == "right":
                kc.turn_right()
            elif action == "left":
                kc.turn_left()

            attempts += 1
            time.sleep(0.05)

        if attempts >= max_attempts:
            print(f"  [navigate] 卡在格子 {i}，跳过")

        if step_callback:
            step_callback(i, target)

    return True
