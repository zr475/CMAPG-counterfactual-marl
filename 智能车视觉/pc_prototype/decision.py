"""
推箱子决策模块
- 游戏状态网格化
- A* 寻路
- 推箱子策略（绕后+推出）
"""

from collections import deque
import heapq
import math
from config import SPEED_FORWARD, SPEED_ROTATE, SPEED_PUSH

# 网格符号
EMPTY, WALL, BOX, DEST, CAR = 0, 1, 2, 3, 4
SYMBOLS = {EMPTY: " ", WALL: "#", BOX: "$", DEST: ".", CAR: "@"}

# 四方向：上右下左
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = ["up", "right", "down", "left"]


class GridMap:
    """游戏状态网格"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[EMPTY] * width for _ in range(height)]
        self.car_pos: tuple[int, int] | None = None
        self.boxes: set[tuple[int, int]] = set()
        self.dests: set[tuple[int, int]] = set()
        self.box_class: dict[tuple[int, int], int] = {}   # 箱子位置 → 类别编号
        self.dest_num: dict[tuple[int, int], int] = {}    # 目的地位置 → 数字编号
        self.bombs: set[tuple[int, int]] = set()          # 炸弹位置

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_passable(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return False
        return self.grid[y][x] not in (WALL, BOX)

    def is_pushable(self, bx: int, by: int, dx: int, dy: int) -> bool:
        """检查箱子 (bx,by) 能否向 (dx,dy) 方向推一格"""
        nx, ny = bx + dx, by + dy
        if not self.in_bounds(nx, ny):
            return False
        return self.grid[ny][nx] == EMPTY or self.grid[ny][nx] == DEST

    def get_neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = pos
        result = []
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if self.is_passable(nx, ny):
                result.append((nx, ny))
        return result

    def display(self):
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) == self.car_pos:
                    row.append("@")
                elif (x, y) in self.boxes:
                    row.append("$")
                elif (x, y) in self.dests:
                    row.append(".")
                else:
                    row.append(SYMBOLS[self.grid[y][x]])
            print(" ".join(row))


def astar(grid: GridMap, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """A* 寻路，返回从 start 到 goal 的路径（不含起点）"""
    if start == goal:
        return []

    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_set = [(h(start), 0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {start: 0}

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current == goal:
            # 回溯路径
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for nb in grid.get_neighbors(current):
            tentative = g_score[current] + 1
            if nb not in g_score or tentative < g_score[nb]:
                g_score[nb] = tentative
                came_from[nb] = current
                heapq.heappush(open_set, (tentative + h(nb), tentative, nb))
    return None


class SokobanAI:
    """推箱子 AI：按箱子图片类别匹配目的地数字，绕后推出"""

    def __init__(self, grid: GridMap):
        self.grid = grid

    def solve_step(self):
        """
        按 class_id == num_id 配对箱子与目的地，选路径最短的配对执行。
        返回 (path, push_dir) 或 None（全部完成/无解）
        """
        car = self.grid.car_pos
        if car is None:
            return None

        # 过滤已在匹配目的地上的箱子
        pending = []
        for box in self.grid.boxes:
            cls = self.grid.box_class.get(box, -1)
            if cls < 0:
                continue
            # 找到匹配的目的地
            target_dest = None
            for dest in self.grid.dests:
                if self.grid.dest_num.get(dest, -1) == cls:
                    target_dest = dest
                    break
            if target_dest is None:
                continue
            # 已经在目的地上了，跳过
            if box == target_dest:
                continue
            pending.append((box, cls, target_dest))

        if not pending:
            return None

        best_result = None
        best_dist = float("inf")

        for box, cls_id, target_dest in pending:
            result = self._plan_push(car, box, target_dest)
            if result is None:
                continue
            path, push_dir = result
            dist = len(path)
            if dist < best_dist:
                best_dist = dist
                best_result = result

        return best_result

    def _plan_push(self, car, box, dest):
        """为指定箱子→目的地规划一次推动，返回 (path, push_dir) 或 None"""
        bx, by = box
        dx, dy = dest

        # 确定推箱方向（箱子 → 目的地）
        push_dir = self._best_push_dir(box, dest)
        if push_dir is None:
            return None
        pdx, pdy = DIRS[push_dir]

        # 车需要站在箱子反方向
        stand_x, stand_y = bx - pdx, by - pdy

        # 如果站位不可通行，尝试其他方向
        if not self.grid.is_passable(stand_x, stand_y):
            found = False
            for alt_dir in range(4):
                adx, ady = DIRS[alt_dir]
                ax, ay = bx - adx, by - ady
                if self.grid.is_passable(ax, ay) and self.grid.is_pushable(bx, by, adx, ady):
                    stand_x, stand_y = ax, ay
                    push_dir = alt_dir
                    found = True
                    break
            if not found:
                return None

        path = astar(self.grid, car, (stand_x, stand_y))
        if path is None:
            return None
        return path, push_dir

    def _best_push_dir(self, box, dest):
        """选箱子→目的地的最优推动方向"""
        bx, by = box
        dx, dy = dest
        # 优先选减小曼哈顿距离的方向
        best = None
        best_gain = float("-inf")
        for i, (ddx, ddy) in enumerate(DIRS):
            nx, ny = bx + ddx, by + ddy
            if not self.grid.is_pushable(bx, by, ddx, ddy):
                continue
            old_dist = abs(bx - dx) + abs(by - dy)
            new_dist = abs(nx - dx) + abs(ny - dy)
            gain = old_dist - new_dist
            if gain > best_gain:
                best_gain = gain
                best = i
        return best


def cmd_to_speed(path: list[tuple[int, int]], car_pos: tuple[int, int]) -> list[tuple[float, float, float]]:
    """
    将网格路径转为 (vx, vy, wz) 速度指令序列
    简化版本：每个格子一步
    """
    speeds = []
    cx, cy = car_pos
    for tx, ty in path:
        vx = 0.3 if tx > cx else (-0.3 if tx < cx else 0)
        vy = 0.3 if ty > cy else (-0.3 if ty < cy else 0)
        speeds.append((vx, vy, 0.0))
        cx, cy = tx, ty
    return speeds
