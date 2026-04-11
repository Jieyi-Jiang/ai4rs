import json
import matplotlib.pyplot as plt

logs = {
    # "baseline": "work_dirs/analyse_log/vis_data_baseline/vis_data_baseline.json",
    # "baseline_np200": "work_dirs/analyse_log/orientedformer_r50_dior-4090_np200/vis_data/20260402_134910.json",
    # "OCDN": "work_dirs/analyse_log/vis_data_OCDN/vis_data_OCDN.json",
    # "OCDN_np200": "/home/jieyi/projects/ai4rs/work_dirs/analyse_log/orientedformer_dino_r50_dior-4090_np200/vis_data/20260331_004008.json",
    # "box_noise=0.3" : "work_dirs/analyse_log/orientedformer_dino_r50_dior-4090_np300_lownoise/vis_data/20260401_174239.json",
    # "label_noise=0.001": "work_dirs/analyse_log/orientedformer_dino_r50_dior-4090_np300_no_clsdn/vis_data/20260401_185848.json",
    # "bn=0.3,ln=0.001": "work_dirs/analyse_log/orientedformer_dino_r50_dior-4090_np300_box0.3_cls0.001/vis_data/20260402_135142.json",
    # "curriculum": "work_dirs/analyse_log/orientedformer_cur_r50_dior-4090_np300/20260406_143325/vis_data/20260406_143325.json",
    # "curriculum_lr": "work_dirs/analyse_log/orientedformer_cur_r50_dior-4090_np300_lr/vis_data/20260407_081006.json",
    # "rotated-dino": "work_dirs/analyse_log/rotated_dino_4scale_r50_2xb4_12e_dior_4090/vis_data/20260407_095204.json",
    # "rotated-rtdetr": "work_dirs/analyse_log/o2_rtdetr_r50vd_2xb4_12e_dior_4090/vis_data/20260407_181726.json",
    "curriculum": "work_dirs/analyse_log/orientedformer_cur_r50_dior-4090_np300_36e/vis_data/20260408_003803.json",
    "rotated-rtdetr": "work_dirs/analyse_log/o2_rtdetr_r50vd_2xb4_36e_dior_4090/vis_data/20260407_235752.json"
}

metric = "dota/AP90"
save_path = f"./work_dirs/analyse_log/results/cur_rtdetr_36e/AP90.png"

plt.figure(figsize=(8, 5))

for name, path in logs.items():
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        eval_idx = 0
        for line in f:
            data = json.loads(line)
            if metric in data:
                eval_idx += 1
                xs.append(eval_idx)
                ys.append(data[metric])

    if not ys:
        print(f"[WARN] {name} has no metric: {metric}")
        continue

    # plt.plot(xs, ys, marker='o', label=name)
    plt.plot(xs, ys,  linewidth=1.5, label=name)

plt.xlabel("Eval index")
plt.ylabel(metric)
plt.title(metric)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(save_path, dpi=200)
print(f"saved to {save_path}")