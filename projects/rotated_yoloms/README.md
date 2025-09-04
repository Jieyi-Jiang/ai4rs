# YOLO-MS (TPAMI 2025)

> IEEE Link [YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection](https://ieeexplore.ieee.org/document/10872821)
> ArXiv Link [YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-Time Object Detection](https://arxiv.org/abs/2308.05480)

<!-- [ALGORITHM] -->

## Abstract

We aim at providing the object detection community with an efficient and performant object detector, termed YOLO-MS. The core design is based on a series of investigations on how multi-branch features of the basic block and convolutions with different kernel sizes affect the detection performance of objects at different scales. The outcome is a new strategy that can significantly enhance multi-scale feature representations of real-time object detectors. To verify the effectiveness of our work, we train our YOLO-MS on the MS COCO dataset from scratch without relying on any other large-scale datasets, like ImageNet or pre-trained weights. Without bells and whistles, our YOLO-MS outperforms the recent state-of-the-art real-time object detectors, including YOLO-v7, RTMDet, and YOLO-v8. Taking the XS version of YOLO-MS as an example, it can achieve an AP score of 42+% on MS COCO, which is about 2% higher than RTMDet with the same model size. Furthermore, our work can also serve as a plug-and-play module for other YOLO models. Typically, our method significantly advances the APs, APl, and AP of YOLOv8-N from 18%+, 52%+, and 37%+ to 20%+, 55%+, and 40%+, respectively, with even fewer parameters and MACs.

<div align=center>
<img src="https://i-blog.csdnimg.cn/img_convert/a6ae33d000432d002f498bc79f75cc2b.png" height="360"/>
</div>

## Results and Models

### DOTA-v1.0

|  Model  | pretrain |  Aug  | mAP  | AP50 | AP75 | Params(M) | FLOPS(G) | batch size |                          Config                          |                                                                                                                                                                       Download                                                                                                                                                                       |
| :---------: | :------: | :---: | :---: | :---: | :---: | :-------: | :------: | :------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLO-MS-S |    COCO   |  RR   | 50.46 | 76.55 | 55.65 |   8.03    |  39.68   |    8=2gpu*<br>4img/gpu     |        [config](./configs/yoloms-s_syncbn_fast_2xb4-36e_dota_previous.py)        |                         [last epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-s_syncbn_fast_2xb4-36e_dota_previous/epoch_36.pth) \| [log](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-s_syncbn_fast_2xb4-36e_dota_previous/20250903_223206/20250903_223206.log) \|<br> [all epoch](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/files) \| [result](https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/rotated_yoloms/yoloms-s_syncbn_fast_2xb4-36e_dota_previous/Task1.zip)                           |


This is your evaluation result for task 1 (VOC metrics):  
mAP: 0.765460659633011  
ap of each class: plane:0.8936621878842959, baseball-diamond:0.8392404068426194, bridge:0.5326113192062201, ground-track-field:0.7353638093780249, small-vehicle:0.8174603067689339, large-vehicle:0.8494447803894849, ship:0.889119814337507, tennis-court:0.90887829545426, basketball-court:0.8833521616607203, storage-tank:0.8785790124897477, soccer-ball-field:0.619967107911772, roundabout:0.5697002557520482, harbor:0.7753673808345287, swimming-pool:0.7565379369479264, helicopter:0.5326251186370747  
COCO style result:  
AP50: 0.765460659633011  
AP75: 0.5564636731501145  
mAP: 0.5045991125776231


**Note**:

1. We follow the latest metrics from the DOTA evaluation server, original voc format mAP is now mAP50.
2. `IN` means ImageNet pretrain, `COCO` means COCO pretrain.
3. By default, DOTA-v1.0 dataset trained with 3x schedule and image size 1024\*1024.

## Citation

```
@article{Chen2025,
  title = {YOLO-MS: Rethinking Multi-Scale Representation Learning for Real-time Object Detection},
  ISSN = {1939-3539},
  url = {http://dx.doi.org/10.1109/TPAMI.2025.3538473},
  DOI = {10.1109/tpami.2025.3538473},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Chen, Yuming and Yuan, Xinbin and Wang, Jiabao and Wu, Ruiqi and Li, Xiang and Hou, Qibin and Cheng, Ming-Ming},
  year = {2025},
  pages = {1–14}
}
```
