
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