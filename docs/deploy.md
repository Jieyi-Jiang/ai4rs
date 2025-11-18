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

download TensorRT CUDA x.x tar package from [NVIDIA](https://developer.nvidia.com/tensorrt), and extract it to the current directory

```
# For example, TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
tar -xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
pip install TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl
export TENSORRT_DIR=$(pwd)/TensorRT-8.6.1.6
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
```

### Convert tensorrt model
```
cd mmdeploy

# for example, download rotated-faster-rcnn model from mmrotate model zoo
wget https://github.com/open-mmlab/mmrotate/raw/main/demo/dota_demo.jpg

# convert model
python tools/deploy.py \
configs/mmrotate/rotated-detection_tensorrt-fp16_static-1024x1024.py \
/root/ai4rs/configs/rotated_faster_rcnn/rotated-faster-rcnn-le90_r50_fpn_1x_dota.py \
rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth \
dota_demo.jpg \
--work-dir mmdeploy_models/ai4rs/fasterrcnn \
--device cuda \
--dump-info
```

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
    rotated-faster-rcnn-le90_r50_fpn_1x_dota.py \
    --model ./mmdeploy_models/ai4rs/ort/end2end.engine \
    --device cuda \
    --speed-test
```