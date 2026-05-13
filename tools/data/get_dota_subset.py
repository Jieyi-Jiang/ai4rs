# -*- coding: utf-8 -*-
"""
make_dota_subset.py

只适配如下目录结构：

data/DOTA/
├── train/
│   ├── images/
│   └── labelTxt/
├── val/
│   ├── images/
│   └── labelTxt/
└── test/
    └── images/

功能：
1. 从 DOTA 原始大图中抽取 train / val 子集；
2. 采用 DOTA 15 类分层随机抽样，尽量保证类别覆盖；
3. 保留完整标注，不删除任何类别；
4. 输出到 data/DOTA_subset；
5. 输出类别统计文件，方便检查和写论文。

注意：
先抽原始大图，再切 patch。
不要先切 patch 再抽。
"""

import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# 配置区：只需要改这里
# ============================================================

# 原始 DOTA 路径
DOTA_ROOT = Path("data/DOTA")

# 输出子集路径
OUTPUT_ROOT = Path("data/DOTA_subset")

# 使用哪个标注文件夹
# 你的目录里有 labelTxt / labelTxt-v1.0 / labelTxt-v1.5
# 一般先用 labelTxt
# 如果你确定要用 v1.0，就改成 "labelTxt-v1.0"
LABEL_DIR_NAME = "labelTxt"

# 抽样数量
# 推荐先用 200 / 60
# 如果训练太慢，改成 150 / 50
# 如果结果波动太大，改成 300 / 80
TRAIN_NUM = 200
VAL_NUM = 60

# 每个类别尽量至少覆盖多少张图
# 子集较小时不要设太高
MIN_IMAGES_PER_CLASS_TRAIN = 8
MIN_IMAGES_PER_CLASS_VAL = 4

# 固定随机种子，保证每次抽出来一样
SEED = 42

# 是否清空已有输出目录
# 第一次运行建议 False
# 如果你想重新生成子集，改成 True
CLEAN_OUTPUT = False

# 图片后缀
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


# DOTA-v1.0 15 类
DOTA_CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]


# ============================================================
# 基础函数
# ============================================================

def parse_dota_label(label_path: Path):
    """
    解析 DOTA 标注文件。

    常见格式：
    x1 y1 x2 y2 x3 y3 x4 y4 class difficulty

    前两行可能是：
    imagesource:GoogleEarth
    gsd:0.5
    """
    classes = []

    with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            lower = line.lower()
            if lower.startswith("imagesource") or lower.startswith("gsd"):
                continue

            parts = line.split()

            if len(parts) < 9:
                continue

            cls_name = parts[8]
            classes.append(cls_name)

    return classes


def find_image_by_stem(image_dir: Path, stem: str):
    """
    根据标注名寻找对应图片。
    例如 P0001.txt -> P0001.png / P0001.jpg / P0001.tif
    """
    for ext in IMAGE_EXTS:
        image_path = image_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path

    return None


def load_split_infos(split: str):
    """
    读取 train 或 val 的图片和标注信息。
    """
    image_dir = DOTA_ROOT / split / "images"
    label_dir = DOTA_ROOT / split / LABEL_DIR_NAME

    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在：{image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"标注目录不存在：{label_dir}")

    infos = {}

    label_files = sorted(label_dir.glob("*.txt"))

    for label_path in label_files:
        stem = label_path.stem
        image_path = find_image_by_stem(image_dir, stem)

        if image_path is None:
            print(f"[Warning] 找不到对应图片：{stem}")
            continue

        classes = parse_dota_label(label_path)

        # 没有目标的图不抽
        if len(classes) == 0:
            continue

        infos[stem] = {
            "image_path": image_path,
            "label_path": label_path,
            "classes": classes,
            "class_set": set(classes),
            "num_objects": len(classes),
        }

    return infos


def count_classes(infos, selected_ids):
    """
    统计选中图片中：
    1. 每类出现在多少张图中；
    2. 每类有多少个实例。
    """
    image_counter = Counter()
    instance_counter = Counter()

    for img_id in selected_ids:
        info = infos[img_id]
        image_counter.update(info["class_set"])
        instance_counter.update(info["classes"])

    return image_counter, instance_counter


def stratified_sample(infos, target_num, min_images_per_class, seed):
    """
    分层随机抽样。

    逻辑：
    1. 先尽量保证 DOTA 15 类都有覆盖；
    2. 稀有类别优先；
    3. 剩余名额再随机填满；
    4. 不删标注，不只保留某些类别。
    """
    rng = random.Random(seed)

    all_image_ids = list(infos.keys())
    rng.shuffle(all_image_ids)

    target_num = min(target_num, len(all_image_ids))

    class_to_images = defaultdict(list)

    for img_id, info in infos.items():
        for cls in info["class_set"]:
            class_to_images[cls].append(img_id)

    print("\n各类别可用图片数量：")
    for cls in DOTA_CLASSES:
        print(f"  {cls}: {len(class_to_images.get(cls, []))}")

    selected = set()

    # 稀有类别优先
    class_order = sorted(
        DOTA_CLASSES,
        key=lambda c: len(class_to_images.get(c, []))
    )

    # 第一阶段：保证类别覆盖
    for cls in class_order:
        candidates = class_to_images.get(cls, [])
        candidates = [x for x in candidates if x not in selected]
        rng.shuffle(candidates)

        # 优先选择类别丰富、目标较多的图
        candidates.sort(
            key=lambda img_id: (
                len(infos[img_id]["class_set"]),
                infos[img_id]["num_objects"],
                rng.random(),
            ),
            reverse=True
        )

        while len(selected) < target_num:
            image_counter, _ = count_classes(infos, selected)

            if image_counter.get(cls, 0) >= min_images_per_class:
                break

            if not candidates:
                break

            selected.add(candidates.pop(0))

    # 第二阶段：填满剩余数量
    remaining = [x for x in all_image_ids if x not in selected]

    def fill_score(img_id):
        image_counter, _ = count_classes(infos, selected)
        info = infos[img_id]

        # 当前覆盖越少的类别，优先级越高
        balance_score = 0.0
        for cls in info["class_set"]:
            balance_score += 1.0 / (image_counter.get(cls, 0) + 1.0)

        # 目标多一点的图稍微优先
        object_score = min(info["num_objects"], 200) / 200.0

        return balance_score, object_score, rng.random()

    remaining.sort(key=fill_score, reverse=True)

    for img_id in remaining:
        if len(selected) >= target_num:
            break

        selected.add(img_id)

    return sorted(selected)


def prepare_output_dir():
    """
    准备输出目录。
    """
    if OUTPUT_ROOT.exists():
        if CLEAN_OUTPUT:
            if "subset" not in OUTPUT_ROOT.name.lower():
                raise RuntimeError(
                    f"为了防止误删，输出目录名必须包含 subset：{OUTPUT_ROOT}"
                )

            print(f"[Info] 清空已有输出目录：{OUTPUT_ROOT}")
            shutil.rmtree(OUTPUT_ROOT)
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[Info] 输出目录已存在，将继续写入：{OUTPUT_ROOT}")
    else:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path):
    """
    复制文件。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    shutil.copy2(src, dst)


def export_split(split: str, infos, selected_ids):
    """
    导出 train 或 val 子集。
    """
    out_image_dir = OUTPUT_ROOT / split / "images"
    out_label_dir = OUTPUT_ROOT / split / LABEL_DIR_NAME

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for img_id in selected_ids:
        info = infos[img_id]

        src_image = info["image_path"]
        src_label = info["label_path"]

        dst_image = out_image_dir / src_image.name
        dst_label = out_label_dir / src_label.name

        copy_file(src_image, dst_image)
        copy_file(src_label, dst_label)


def write_selected_list(split: str, selected_ids):
    """
    保存抽到的图片名，方便复现实验。
    """
    path = OUTPUT_ROOT / f"{split}_selected_images.txt"

    with open(path, "w", encoding="utf-8") as f:
        for img_id in selected_ids:
            f.write(f"{img_id}\n")


def write_stats(split: str, infos, selected_ids):
    """
    输出类别统计。
    """
    image_counter, instance_counter = count_classes(infos, selected_ids)

    stats = {
        "split": split,
        "num_images": len(selected_ids),
        "num_instances": int(sum(instance_counter.values())),
        "image_count_per_class": dict(sorted(image_counter.items())),
        "instance_count_per_class": dict(sorted(instance_counter.items())),
        "selected_images": selected_ids,
    }

    json_path = OUTPUT_ROOT / f"{split}_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    csv_path = OUTPUT_ROOT / f"{split}_class_stats.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class,image_count,instance_count\n")
        for cls in DOTA_CLASSES:
            f.write(
                f"{cls},"
                f"{image_counter.get(cls, 0)},"
                f"{instance_counter.get(cls, 0)}\n"
            )

    return stats


def print_stats(stats):
    """
    打印统计结果。
    """
    print("\n" + "-" * 60)
    print(f"{stats['split']} 子集统计")
    print("-" * 60)
    print(f"图片数量：{stats['num_images']}")
    print(f"实例数量：{stats['num_instances']}")

    print("\n各类别实例数：")
    for cls in DOTA_CLASSES:
        count = stats["instance_count_per_class"].get(cls, 0)
        print(f"  {cls}: {count}")


def process_split(split: str, target_num: int, min_images_per_class: int, seed: int):
    """
    处理 train 或 val。
    """
    print("\n" + "=" * 70)
    print(f"开始处理：{split}")
    print("=" * 70)

    infos = load_split_infos(split)

    print(f"[Info] 读取到有效图片数：{len(infos)}")
    print(f"[Info] 计划抽取图片数：{target_num}")

    selected_ids = stratified_sample(
        infos=infos,
        target_num=target_num,
        min_images_per_class=min_images_per_class,
        seed=seed,
    )

    print(f"[Info] 实际抽取图片数：{len(selected_ids)}")

    export_split(split, infos, selected_ids)
    write_selected_list(split, selected_ids)
    stats = write_stats(split, infos, selected_ids)
    print_stats(stats)

    return selected_ids, stats


def check_overlap(train_ids, val_ids):
    """
    检查 train 和 val 是否有重名图片。
    """
    overlap = set(train_ids) & set(val_ids)

    if len(overlap) == 0:
        print("\n[Info] train / val 没有重名图片，正常。")
    else:
        print("\n[Warning] train / val 存在重名图片：")
        for img_id in sorted(overlap)[:20]:
            print(f"  {img_id}")
        print(f"共 {len(overlap)} 张重叠，请检查数据。")


def write_summary(train_stats, val_stats):
    """
    输出总览文件。
    """
    summary = {
        "dataset": "DOTA_subset",
        "source_root": str(DOTA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "label_dir_name": LABEL_DIR_NAME,
        "sampling_strategy": "stratified random sampling with class coverage",
        "seed": SEED,
        "train_num": train_stats["num_images"],
        "val_num": val_stats["num_images"],
        "dota_classes": DOTA_CLASSES,
        "train": train_stats,
        "val": val_stats,
    }

    path = OUTPUT_ROOT / "subset_summary.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 70)
    print("DOTA 子集抽样脚本")
    print("=" * 70)

    print(f"DOTA_ROOT: {DOTA_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"LABEL_DIR_NAME: {LABEL_DIR_NAME}")
    print(f"TRAIN_NUM: {TRAIN_NUM}")
    print(f"VAL_NUM: {VAL_NUM}")
    print(f"SEED: {SEED}")

    if not DOTA_ROOT.exists():
        raise FileNotFoundError(f"DOTA_ROOT 不存在：{DOTA_ROOT}")

    prepare_output_dir()

    train_ids, train_stats = process_split(
        split="train",
        target_num=TRAIN_NUM,
        min_images_per_class=MIN_IMAGES_PER_CLASS_TRAIN,
        seed=SEED,
    )

    val_ids, val_stats = process_split(
        split="val",
        target_num=VAL_NUM,
        min_images_per_class=MIN_IMAGES_PER_CLASS_VAL,
        seed=SEED + 1,
    )

    check_overlap(train_ids, val_ids)

    write_summary(train_stats, val_stats)

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

    print("\n输出目录：")
    print(f"  {OUTPUT_ROOT}")

    print("\n重点检查这些文件：")
    print(f"  {OUTPUT_ROOT / 'train_class_stats.csv'}")
    print(f"  {OUTPUT_ROOT / 'val_class_stats.csv'}")
    print(f"  {OUTPUT_ROOT / 'train_selected_images.txt'}")
    print(f"  {OUTPUT_ROOT / 'val_selected_images.txt'}")
    print(f"  {OUTPUT_ROOT / 'subset_summary.json'}")

    print("\n下一步：")
    print("  1. 打开 train_class_stats.csv 和 val_class_stats.csv")
    print("  2. 检查 15 类是否都有覆盖")
    print("  3. 再对 data/DOTA_subset 进行切图")
    print("  4. 用切图后的数据训练 Baseline 和最终方法")


if __name__ == "__main__":
    main()