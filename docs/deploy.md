# AI4RS Deployment


## TensorRT

### Installation

#### Install ai4rs

Please follow the [installation guide](../README.md#installation-%EF%B8%8F) to install ai4rs.

#### Install mmdeploy

**Method I**: Install precompiled package

1.install MMDeploy model converter
```
pip install mmdeploy==1.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2.install MMDeploy sdk inference
```
#   support onnxruntime-gpu, tensorrt
pip install mmdeploy-runtime-gpu==1.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3.install inference engine

check CUDA  version
```
python -c "import torch; print(torch.version.cuda)"
```
check python version

download TensorRT CUDA x.x tar package from [NVIDIA](https://developer.nvidia.com/tensorrt), and extract it to the current directory

```
# For example, TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
tar -xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
pip install TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
export TENSORRT_DIR=$(pwd)/TensorRT-8.6.1.6
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
```
check
```
echo $LD_LIBRARY_PATH
echo $TENSORRT_DIR
# /root/TensorRT-8.6.1.6/lib:/usr/local/cuda-10.2/lib64:
# /root/TensorRT-8.6.1.6
```


### Convert tensorrt model

|  Method  |  Pth     |  Command |
| :-----:  | :-----:  | :-----:  |
|`rotated_rtmdet_s-3x-dota.py` | [pth](https://download.openmmlab.com/mmrotate/v1.0/rotated_rtmdet/rotated_rtmdet_s-3x-dota/rotated_rtmdet_s-3x-dota-11f6ccf5.pth) | python /root/mmdeploy/tools/deploy.py /root/mmdeploy/configs/mmrotate/rotated-detection_tensorrt-fp16_static-1024x1024.py configs/rotated_rtmdet/rotated_rtmdet_s-3x-dota.py rotated_rtmdet_s-3x-dota-11f6ccf5.pth demo/demo.jpg --work-dir mmdeploy_models/ai4rs/rtmdet_s --device cuda:0 --dump-info|





### Model inference

#### SDK model inference
```
from mmdeploy_runtime import RotatedDetector
import cv2
import numpy as np
img = cv2.imread('./dota_demo.jpg')
detector = RotatedDetector(model_path='./mmdeploy_models/ai4rs/fasterrcnn', device_name='cuda', device_id=0)
det = detector(img)

boxes = det[0][:,:-1]
labels = det[1]
scores = det[0][:,-1]

class_names = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter']
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=int)

def draw_obb(image, obb, label, score, color):
    cx, cy, w, h, angle = obb
    theta = angle * 180 / np.pi
    rect = ((cx, cy), (w, h), theta)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(image, [box], True, color, 2)
    cv2.putText(image, f'{label}:{score:.2f}', 
                (box[0][0], box[0][1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

for box, label_id, score in zip(boxes, labels, scores):
    label = class_names[label_id] if label_id < len(class_names) else str(label_id)
    color = tuple(colors[label_id % len(colors)].tolist())
    draw_obb(img, box, label, score, color)

cv2.imwrite('vis_rotated_result_with_label.jpg', img)
print('saved to vis_rotated_result_with_label.jpg')
```

####  evaluate model
```
python tools/test.py \
${DEPLOY_CFG} \
${MODEL_CFG} \
--model ${BACKEND_MODEL_FILES} \
[--out ${OUTPUT_PKL_FILE}] \
[--format-only] \
[--metrics ${METRICS}] \
[--show] \
[--show-dir ${OUTPUT_IMAGE_DIR}] \
[--show-score-thr ${SHOW_SCORE_THR}] \
--device ${DEVICE} \
[--cfg-options ${CFG_OPTIONS}] \
[--metric-options ${METRIC_OPTIONS}]
[--log2file work_dirs/output.txt]
[--batch-size ${BATCH_SIZE}]
[--speed-test] \
[--warmup ${WARM_UP}] \
[--log-interval ${LOG_INTERVERL}] \
```
For example
```
python tools/test.py \
    configs/mmrotate/rotated-detection_tensorrt-fp16_static-1024x1024.py \
    /root/ai4rs/configs/rotated_rtmdet/rotated_rtmdet_s-3x-dota.py \
    --model  mmdeploy_models/ai4rs/rtmdet_s/end2end.engine \
    --device cuda:0 \
    --speed-test
```
Out
```
mmengine - INFO - Epoch(test) [   50/10833]    eta: 0:28:13  time: 0.1191  data_time: 0.0456  memory: 39  
mmengine - INFO - Epoch(test) [  100/10833]    eta: 0:23:52  time: 0.1192  data_time: 0.0125  memory: 39  
mmengine - INFO - [tensorrt]-110 times per count: 21.84 ms, 45.79 FPS
mmengine - INFO - Epoch(test) [  150/10833]    eta: 0:22:14  time: 0.1098  data_time: 0.0207  memory: 39  
mmengine - INFO - Epoch(test) [  200/10833]    eta: 0:21:29  time: 0.1204  data_time: 0.0457  memory: 39  
mmengine - INFO - [tensorrt]-210 times per count: 23.65 ms, 42.29 FPS
mmengine - INFO - Epoch(test) [  250/10833]    eta: 0:21:07  time: 0.1095  data_time: 0.0199  memory: 39  
mmengine - INFO - Epoch(test) [  300/10833]    eta: 0:20:44  time: 0.1005  data_time: 0.0197  memory: 39  
mmengine - INFO - [tensorrt]-310 times per count: 27.82 ms, 35.94 FPS
mmengine - INFO - Epoch(test) [  350/10833]    eta: 0:20:23  time: 0.1106  data_time: 0.0113  memory: 39  
mmengine - INFO - Epoch(test) [  400/10833]    eta: 0:20:03  time: 0.1086  data_time: 0.0121  memory: 39  
mmengine - INFO - [tensorrt]-410 times per count: 26.91 ms, 37.15 FPS
mmengine - INFO - Epoch(test) [  450/10833]    eta: 0:19:51  time: 0.1189  data_time: 0.0286  memory: 39  
mmengine - INFO - Epoch(test) [  500/10833]    eta: 0:19:34  time: 0.1106  data_time: 0.0370  memory: 39  
```

### Model profile
```
python tools/profiler.py \
${DEPLOY_CFG} \
${MODEL_CFG} \
${IMAGE_DIR} \
--model ${BACKEND_MODEL_FILES} \
--device ${DEVICE} \
--shape ${SHAPE} \
--warmup ${WARM_UP} \
--num-iter ${NUM_ITER} \
--batch-size ${BATCH_SIZE} \
--img-ext ${IMG_EXT}
```
For example
```
python tools/profiler.py \
    configs/mmrotate/rotated-detection_tensorrt-fp16_static-1024x1024.py \
    /root/ai4rs/configs/rotated_rtmdet/rotated_rtmdet_s-3x-dota.py \
    /root/split_ss_dota/test/images/ \
    --model mmdeploy_models/ai4rs/rtmdet_s/end2end.engine \
    --device cuda:0 \
    --shape 1024x1024 \
    --warmup 100  \
    --num-iter 800
```
Out
```

- mmengine - WARNING - Failed to search registry with scope "mmrotate" in the "Codebases" registry tree. As a workaround, the current "Codebases" registry in "mmdeploy" is used to build instance. This may cause unexpected failure when running the built modules. Please check whether "mmrotate" is a correct scope, or whether the registry is initialized.
- mmengine - WARNING - Failed to search registry with scope "mmrotate" in the "mmrotate_tasks" registry tree. As a workaround, the current "mmrotate_tasks" registry in "mmdeploy" is used to build instance. This may cause unexpected failure when running the built modules. Please check whether "mmrotate" is a correct scope, or whether the registry is initialized.
- mmengine - WARNING - Failed to search registry with scope "mmrotate" in the "backend_detectors" registry tree. As a workaround, the current "backend_detectors" registry in "mmdeploy" is used to build instance. This may cause unexpected failure when running the built modules. Please check whether "mmrotate" is a correct scope, or whether the registry is initialized.
- mmengine - INFO - Successfully loaded tensorrt plugins from /root/anaconda3/envs/ai4rs/lib/python3.10/site-packages/mmdeploy/lib/libmmdeploy_tensorrt_ops.so
- mmengine - INFO - Successfully loaded tensorrt plugins from /root/anaconda3/envs/ai4rs/lib/python3.10/site-packages/mmdeploy/lib/libmmdeploy_tensorrt_ops.so
[TRT] [W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
11/18 19:39:04 - mmengine - INFO - Found totally 10833 image files in /root/split_ss_dota/test/images/
11/18 19:39:28 - mmengine - INFO - [tensorrt]-120 times per count: 4.83 ms, 207.25 FPS
11/18 19:39:31 - mmengine - INFO - [tensorrt]-140 times per count: 4.89 ms, 204.47 FPS
...
11/18 19:41:57 - mmengine - INFO - [tensorrt]-880 times per count: 7.03 ms, 142.15 FPS
11/18 19:42:01 - mmengine - INFO - [tensorrt]-900 times per count: 7.14 ms, 140.06 FPS
----- Settings:
+------------+-----------+
| batch size |     1     |
|   shape    | 1024x1024 |
| iterations |    800    |
|   warmup   |    100    |
+------------+-----------+
----- Results:
+--------+------------+---------+
| Stats  | Latency/ms |   FPS   |
+--------+------------+---------+
|  Mean  |   7.140    | 140.062 |
| Median |   4.735    | 211.192 |
|  Min   |   4.136    | 241.755 |
|  Max   |   85.282   |  11.726 |
+--------+------------+---------+
```


### BUG

BUG 1:
```
raise TypeError(f'engine should be str or trt.ICudaEngine, \ TypeError: engine should be str or trt.ICudaEngine, but given: <class 'NoneType'>
```
Solution 1:
```
export TENSORRT_DIR=$(pwd)/TensorRT-8.6.1.6
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
```