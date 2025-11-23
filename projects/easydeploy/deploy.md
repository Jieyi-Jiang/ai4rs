
## TensorRT

### Installation

#### Install ai4rs

Please follow the [installation guide](../README.md#installation-%EF%B8%8F) to install ai4rs.

#### Install onnx and tensorrt

Install onnx
```
pip install onnx onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install onnx-simplifier -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Install tensorrt  
NOTE: Ensure TensorRT version (e.g., -cu12 suffix) matches the CUDA version in the current conda environment!!  
NOTE: Ensure TensorRT version (e.g., -cu12 suffix) matches the CUDA version in the current conda environment!!   
```
pip install tensorrt-cu12  -i https://pypi.tuna.tsinghua.edu.cn/simple
```
If install `tensorrt-cu12`:
```
tensorrt-cu12             10.14.1.48.post1          pypi_0    pypi
tensorrt-cu12-bindings    10.14.1.48.post1          pypi_0    pypi
tensorrt-cu12-libs        10.14.1.48.post1          pypi_0    pypi
```

### Export

#### Onnx

```
python projects/easydeploy/tools/export_onnx_rtdetr.py
```

#### Onnx -> Tensorrt

```
python projects/easydeploy/tools/build_engine_rtdetr.py
```

#### Visual
```
python projects/easydeploy/tools/image_demo_rtdetr.py
```

### Lantency
install inference engine

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
use trtexec
```
/root/TensorRT-8.6.1.6/bin/trtexec --onnx=/root/ai4rs/work_dirs/easydeploy/rtdetr/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.onnx --fp16
```