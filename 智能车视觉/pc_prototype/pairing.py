"""
配对逻辑 —— 根据命名规则将箱子类别与目的地数字配对
规则: image_class/ 目录名 00mickey_mouse → class_id=0 ↔ dest_num=0
"""

from explorer import ExplorationResult


def pair_boxes(exploration: ExplorationResult) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    """
    将箱子与目的地配对。
    返回 [(箱子网格坐标, 目的地网格坐标, class_id), ...]
    """
    pairs = []

    # 建立目的地数字→位置的索引
    num_to_dest: dict[int, tuple[int, int]] = {}
    for dest_pos, num in exploration.dest_num.items():
        num_to_dest[num] = dest_pos

    for box_pos, cls_id in exploration.box_class.items():
        if cls_id in num_to_dest:
            dest_pos = num_to_dest[cls_id]
            pairs.append((box_pos, dest_pos, cls_id))
        else:
            print(f"  [配对] 箱子 {box_pos} class={cls_id} 无匹配目的地（可用: {list(num_to_dest.keys())}）")

    return pairs


def pair_boxes_from_dicts(
    box_class: dict[tuple[int, int], int],
    dest_num: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    """直接从字典配对（不依赖 ExplorationResult）"""
    pairs = []
    num_to_dest = {num: pos for pos, num in dest_num.items()}
    for box_pos, cls_id in box_class.items():
        if cls_id in num_to_dest:
            pairs.append((box_pos, num_to_dest[cls_id], cls_id))
    return pairs


def print_pairs(pairs: list):
    """打印配对结果"""
    if not pairs:
        print("无配对结果")
        return
    print("配对结果:")
    for i, (box, dest, cls_id) in enumerate(pairs):
        print(f"  {i+1}. 箱子{box} (class={cls_id}) → 目的地{dest}")
