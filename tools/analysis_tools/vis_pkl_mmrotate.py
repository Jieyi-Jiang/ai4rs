import os
import cv2
import mmcv
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from mmengine.fileio import load

# ====== 配置 ======
config_file = 'projects/OrientedFormer-DINO/configs/orientedformer_dino_r50_dior-4090_np300.py'
pkl_file = 'work_dirs/pkl_files/OF_OCDN_np300_boxdn0.5.pkl'
out_dir = 'work_dirs/vis_from_pkl_OCDN_thr0.2'
score_thr = 0.2

draw_text = False        # 是否画类别+分数
print_progress = True    # 是否打印进度
print_every = 50         # 每多少张打印一次日志

# ==================

os.makedirs(out_dir, exist_ok=True)

cfg = Config.fromfile(config_file)
results = load(pkl_file)

# classes
classes = None
if hasattr(cfg, 'metainfo'):
    classes = cfg.metainfo.get('classes', None)

if classes is None:
    try:
        classes = cfg.test_dataloader.dataset.metainfo['classes']
    except Exception:
        classes = None

num_classes = len(classes) if classes is not None else 20

# ====== 颜色表 ======
def get_color_map(num_classes):
    np.random.seed(42)  # 固定颜色
    colors = np.random.randint(0, 256, size=(num_classes, 3))
    return {i: tuple(map(int, colors[i])) for i in range(num_classes)}

color_map = get_color_map(num_classes)

# ====== 旋转框转四点 ======
def rbox_to_poly(cx, cy, w, h, angle):
    angle_deg = angle * 180.0 / np.pi
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle_deg))
    pts = cv2.boxPoints(rect)
    return np.int32(pts)

# ====== 主循环 ======
iterator = tqdm(results) if print_progress else results

for i, item in enumerate(iterator):
    img_path = item['img_path']
    img = mmcv.imread(img_path)  # BGR

    pred = item['pred_instances']
    bboxes = pred['bboxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()

    keep = scores >= score_thr
    bboxes = bboxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    for idx, (box, label, score) in enumerate(zip(bboxes, labels, scores)):
        label = int(label)
        color = color_map[label]

        poly = rbox_to_poly(*box)

        # 画框
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=2)

        # ====== 可选：画文字 ======
        if draw_text:
            cls_name = str(label) if classes is None else classes[label]
            text = f'{cls_name} {score:.2f}'

            x, y = poly[0]

            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            y_text = max(int(y) - 4, th + 2)

            cv2.rectangle(
                img,
                (int(x), y_text - th - 2),
                (int(x) + tw, y_text + 2),
                color,
                -1
            )

            cv2.putText(
                img,
                text,
                (int(x), y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    out_file = os.path.join(out_dir, os.path.basename(img_path))
    mmcv.imwrite(img, out_file)

    # ====== 打印日志 ======
    if (not print_progress) and (i % print_every == 0):
        print(f'[vis] {i}/{len(results)}: {out_file}')

print(f'\nDone. Saved to {out_dir}')