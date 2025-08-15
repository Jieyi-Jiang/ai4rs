# YOLOX

> [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)

<!-- [ALGORITHM] -->

## Abstract

In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector -- YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanced detection techniques, i.e., a decoupled head and the leading label assignment strategy SimOTA to achieve state-of-the-art results across a large scale range of models: For YOLO-Nano with only 0.91M parameters and 1.08G FLOPs, we get 25.3% AP on COCO, surpassing NanoDet by 1.8% AP; for YOLOv3, one of the most widely used detectors in industry, we boost it to 47.3% AP on COCO, outperforming the current best practice by 3.0% AP; for YOLOX-L with roughly the same amount of parameters as YOLOv4-CSP, YOLOv5-L, we achieve 50.0% AP on COCO at a speed of 68.9 FPS on Tesla V100, exceeding YOLOv5-L by 1.8% AP. Further, we won the 1st Place on Streaming Perception Challenge (Workshop on Autonomous Driving at CVPR 2021) using a single YOLOX-L model. We hope this report can provide useful experience for developers and researchers in practical scenes, and we also provide deploy versions with ONNX, TensorRT, NCNN, and Openvino supported.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/144001736-9fb303dd-eac7-46b0-ad45-214cfa51e928.png"/>
</div>

## Results and models

**DOTA**

|         Backbone         |  mAP  | AP50 | AP75 | Angle | lr schd |  Scale | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------: | :-: | :--------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| CSPDarknet <br> (1024,1024,200) | 46.27 | 74.94  |   49.49    |   `le90`   |      `300e`      |  single <br> scale  | 8=2gpu*<br>4img/gpu   | [rotated_yolox_s_300e_dota_le90.py](./configs/rotated_yolox_s_300e_dota_le90.py) | [all clpt](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [log](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_s_300e_dota_le90/20250809_224227.log) \| [result](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_s_300e_dota_le90/Task1.zip) \| [last ckpt](https://modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yolox/rotated_yolox_s_300e_dota_le90/epoch_300.pth) |

This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.749433522358599  
ap of each class: plane:0.8642951917278126, baseball-diamond:0.8293671917667135, bridge:0.4792212695427707, ground-track-field:0.6550645193578333, small-vehicle:0.7802050660379458, large-vehicle:0.8497052490370042, ship:0.8906889639787817, tennis-court:0.9083821116390618, basketball-court:0.8798755073052649, storage-tank:0.8600542432660736, soccer-ball-field:0.5829021529662977, roundabout:0.6362592197722562, harbor:0.7323143096770864, swimming-pool:0.6980757951055564, helicopter:0.5950920441985256  
COCO style result:  
AP50: 0.749433522358599
AP75: 0.4948977689064668
mAP: 0.46267007542788247

**Note**: Since we configured `auto_scale_lr = dict(base_batch_size=64, enable=True)` in `rotated_yolox_s_300e_dota_le90.py`, the learning rate will be automatically scaled. Specifically, with our actual batch size of `8=2gpu*4img/gpu`, the learning rate is automatically reduced to 1/8 of the original value.

## Citation

```latex
@article{yolox2021,
  title={{YOLOX}: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
