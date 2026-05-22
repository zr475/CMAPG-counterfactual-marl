"""
地图文件解析
符号: #=围墙  $=箱子  .=目的地  *=炸弹  -=空地
"""

from pathlib import Path
from decision import GridMap, WALL, EMPTY, BOX, DEST


def load_map(filepath: str | Path) -> GridMap:
    """从 map_file/*.txt 加载地图，返回 GridMap"""
    with open(filepath, encoding="utf-8") as f:
        lines = [line.rstrip("\n").rstrip("\r") for line in f if line.strip()]

    if not lines:
        raise ValueError("地图文件为空")

    height = len(lines)
    width = max(len(line) for line in lines)

    gm = GridMap(width, height)
    gm.bombs = set()

    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == "#":
                gm.grid[y][x] = WALL
            elif ch == "$":
                gm.grid[y][x] = BOX
                gm.boxes.add((x, y))
            elif ch == ".":
                gm.grid[y][x] = DEST
                gm.dests.add((x, y))
            elif ch == "*":
                gm.grid[y][x] = EMPTY  # 炸弹位置可通行，炸掉后变空地
                gm.bombs.add((x, y))
            elif ch == "-":
                gm.grid[y][x] = EMPTY
            # 其他字符视为空地

    return gm


def grid_to_pixel(gx: int, gy: int, cell_size: int) -> tuple[int, int]:
    """网格坐标 → 像素坐标（格子中心）"""
    return gx * cell_size + cell_size // 2, gy * cell_size + cell_size // 2


def pixel_to_grid(px: int, py: int, cell_size: int) -> tuple[int, int]:
    """像素坐标 → 网格坐标"""
    return px // cell_size, py // cell_size
