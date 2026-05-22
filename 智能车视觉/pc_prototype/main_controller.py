"""
主控制循环 —— 整合四阶段流程

阶段:
  1. 探索扫描 — 键盘控制 VR 游戏中的车，遍历箱子/目的地，FPV 识别
  2. 记录投票 — 多帧投票（已内嵌在 explorer 中）
  3. 配对     — 箱子类别 ↔ 目的地数字
  4. 执行推箱 — 串口发指令给真实小车

用法:
  python main_controller.py                  # 完整流程
  python main_controller.py --virtual        # 虚拟串口模式（调试）
  python main_controller.py --map map1.txt   # 指定地图文件
  python main_controller.py --explore-only   # 仅探索+配对（不推箱）
  python main_controller.py --exec-only      # 仅推箱（需已有配对数据）
"""

import argparse
import json
import os
import sys
import time

from config import MAP_DIR, GRID_CELL
from map_loader import load_map
from explorer import Explorer
from pairing import pair_boxes, print_pairs
from executor import PushExecutor
from decision import GridMap


def phase_explore(explorer: Explorer) -> dict:
    """阶段 1+2：探索扫描 + 记录投票"""
    result = explorer.explore()
    return {
        "box_class": {f"{x},{y}": c for (x, y), c in result.box_class.items()},
        "dest_num": {f"{x},{y}": n for (x, y), n in result.dest_num.items()},
        "box_conf": {f"{x},{y}": c for (x, y), c in result.box_confidence.items()},
        "dest_conf": {f"{x},{y}": c for (x, y), c in result.dest_confidence.items()},
    }


def phase_pair(result: dict):
    """阶段 3：配对"""
    from explorer import ExplorationResult
    er = ExplorationResult()
    er.box_class = {tuple(map(int, k.split(","))): v for k, v in result.get("box_class", {}).items()}
    er.dest_num = {tuple(map(int, k.split(","))): v for k, v in result.get("dest_num", {}).items()}
    pairs = pair_boxes(er)
    return pairs, er


def phase_execute(gm: GridMap, er, pairs: list, virtual: bool = True):
    """阶段 4：执行推箱"""
    if not pairs:
        print("无配对，跳过执行")
        return
    executor = PushExecutor(virtual=virtual)
    try:
        executor.open()
        executor.execute_push(gm, pairs)
    finally:
        executor.close()


def load_pairing_cache(cache_path: str) -> dict | None:
    """加载之前保存的探索结果（跳过重复探索）"""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_pairing_cache(cache_path: str, data: dict):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="推箱子四阶段主控")
    parser.add_argument("--map", default="map1.txt", help="地图文件名")
    parser.add_argument("--virtual", action="store_true", help="虚拟串口模式（打印指令）")
    parser.add_argument("--explore-only", action="store_true", help="仅探索+配对，不执行推箱")
    parser.add_argument("--exec-only", action="store_true", help="仅执行推箱，使用缓存配对")
    parser.add_argument("--cache", default="cache/pairing_result.json", help="配对结果缓存路径")
    args = parser.parse_args()

    map_path = MAP_DIR / args.map
    if not map_path.exists():
        print(f"地图文件不存在: {map_path}")
        print(f"可用地图: {[f.name for f in MAP_DIR.glob('*.txt')]}")
        sys.exit(1)

    print(f"地图: {args.map}")
    gm = load_map(map_path)
    gm.display()
    print(f"箱子: {len(gm.boxes)}  目的地: {len(gm.dests)}  炸弹: {len(gm.bombs)}")
    print()

    # ---- 阶段 1+2: 探索+记录 ----
    if not args.exec_only:
        input("请切换到 VR 游戏窗口，按 Enter 开始探索...")
        print("提示：3 秒后开始，请确保游戏窗口在前台！")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

        yolo_path = os.path.join(os.path.dirname(__file__), "..", "数字识别", "best.pt")
        explorer = Explorer(yolo_path)
        pairing_data = phase_explore(explorer)
        save_pairing_cache(args.cache, pairing_data)
        print(f"\n配对数据已缓存: {args.cache}")
    else:
        pairing_data = load_pairing_cache(args.cache)
        if pairing_data is None:
            print(f"缓存不存在: {args.cache}，请先运行探索")
            sys.exit(1)
        print(f"从缓存加载配对数据: {args.cache}")

    # ---- 阶段 3: 配对 ----
    pairs, er = phase_pair(pairing_data)
    print_pairs(pairs)
    print()

    if args.explore_only:
        print("--explore-only 模式，跳过推箱执行")
        return

    # ---- 阶段 4: 执行推箱 ----
    # 将配对数据填入 GridMap
    for box, dest, cls_id in pairs:
        gm.box_class[box] = cls_id
        gm.dest_num[dest] = cls_id

    # 推箱执行需要知道车的位置，此处假设车在探索结束后停在地图某处
    # 实际运行时需从俯视图检测获取初始车位置
    print("初始化推箱执行...")
    if not args.virtual:
        input("请确认串口已连接，按 Enter 开始推箱...")

    phase_execute(gm, er, pairs, virtual=args.virtual)
    print("\n全部完成！")


if __name__ == "__main__":
    main()
