from mmrotate.registry import MODELS
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.models.utils.misc import unpack_gt_instances
from mmdet.structures import SampleList
from mmengine.structures import InstanceData
import torch
from torch import Tensor
from mmdet.utils import OptConfigType
from .oriented_dino_layers import OrientedCdnQueryGenerator
@MODELS.register_module()
class OrientedDDQRCNN(TwoStageDetector):

    def __init__(self, *args, dn_cfg: OptConfigType = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            # dn_cfg['num_classes'] = self.bbox_head.num_classes
            # dn_cfg['embed_dims'] = self.embed_dims
            # dn_cfg['num_matching_queries'] = self.num_queries
            dn_cfg['num_classes'] = self.roi_head.bbox_head[0].num_classes
            dn_cfg['embed_dims'] = self.roi_head.content_dim
            dn_cfg['num_matching_queries'] = self.rpn_head.num_proposals
        self.dn_query_generator = OrientedCdnQueryGenerator(**dn_cfg)
    
    def loss(self,
             batch_inputs: Tensor,
             batch_data_samples: SampleList):
        '''
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        
        OrientedFormer 训练损失计算主流程：
            1. 提取图像特征 (Backbone + FPN)。
            2. RPN 阶段：计算初步候选框损失，并生成供第二阶段使用的旋转 Query。
            3. 结果组装：将 RPN 输出的位置 (xyzrt) 和内容 (content) 封装。
            4. RoI 阶段：基于 RPN 的输出进行精细化的旋转框回归与分类，计算最终损失。
            5. 返回包含两个阶段所有损失分量的字典。
        '''
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        # 处理 GT ，这个在 _forward 中不需要，因为 _forward 不需要监督信息
        gt_bboxes, gt_labels = [], []
        for i in range(len(batch_gt_instances)):
            gt_bboxes.append(batch_gt_instances[i].bboxes)
            gt_labels.append(batch_gt_instances[i].labels)

        # 收集并记录模型各部分产生的损失分量，用于后面加权求和计算总损失以进行反向传播，
        # 同时供训练日志记录和监控。
        losses = dict()
        # list(level), each level has shape (bs, c, h, w)
        x = self.extract_feat(batch_inputs) 
        rpn_x = x
        roi_x = x

         # 这里调用 loss_and_predict，_forward 和 predtic 中调用 predict
        rpn_losses, imgs_whwht, distinc_query_dict = \
            self.rpn_head.loss_and_predict( 
                rpn_x,
                batch_img_metas,
                gt_bboxes,
                gt_labels)
        ddq_query_xyzrt = distinc_query_dict['query_xyzrt']     # (bs, 300, 256)
        ddq_query_content = distinc_query_dict['query_content'] # (bs, 300, 5)
        # 这里插入 dn_query 的构建
        # if self.training: # 这个判断不太必要，loss 里面默认就是训练状态
        #     pass
        dn_query_content, dn_query_xyzrt, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
        # 这里插入 dn_query 的构建

        # dn_query 和前面生成的 query 拼接
        query_xyzrt = torch.cat([dn_query_xyzrt, ddq_query_xyzrt], dim=1)
        query_content = torch.cat([dn_query_content, ddq_query_content], dim=1)

        # 将 RPN 损失存入总字典，加上 rpn_ 前缀以示区分
        # （在 _forward 和 predict 方法里面没用用到 rpn_losses）
        for k, v in rpn_losses.items():
            losses[f'rpn_{k}'] = v

        rpn_results_list = []
        for idx in range(len(batch_img_metas)):
            rpn_results = InstanceData()
            rpn_results.query_xyzrt = query_xyzrt[idx]
            rpn_results.imgs_whwht = imgs_whwht[idx].repeat(
                len(query_xyzrt[idx]), 1)
            rpn_results.query_content = query_content[idx]
            rpn_results_list.append(rpn_results)

        # 这里插入 dn_loss 计算
        roi_losses = self.roi_head.loss(
            roi_x, rpn_results_list, batch_data_samples,
            dn_mask=dn_mask,
            dn_meta=dn_meta)
        losses.update(roi_losses)

        return losses

    # 除了打包 batch_data_samples，其他差不多
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: list,
                rescale: bool = True):
        '''
        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            rescale (bool): True
        Returns:
            batch_data_samples:
        '''
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        x = self.extract_feat(batch_inputs)
        rpn_x = x
        roi_x = x

        rpn_losses, imgs_whwht, distinc_query_dict = \
            self.rpn_head.predict(
                rpn_x, batch_img_metas)

        query_xyzrt = distinc_query_dict['query_xyzrt']
        query_content = distinc_query_dict['query_content']

        rpn_results_list = []
        for idx in range(len(batch_img_metas)):
            rpn_results = InstanceData()
            rpn_results.query_xyzrt = query_xyzrt[idx]
            rpn_results.imgs_whwht = imgs_whwht[idx].repeat(
                len(query_xyzrt[idx]), 1)
            rpn_results.query_content = query_content[idx]
            rpn_results_list.append(rpn_results)
        # 图片进网络前被 resize（比如 1024×768 → 800×800）
        # 网络输出的 bbox 坐标是在 resize 后的尺寸空间里
        # 用户拿到的结果要在原图上画框，所以必须 rescale 回去
        results_list = self.roi_head.predict(roi_x,
                                             rpn_results_list,
                                             batch_data_samples,
                                             rescale=rescale)  # 这里 resacle，为什么？
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


    def _forward(self, batch_inputs, batch_data_samples) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        assert batch_data_samples != None, 'Copy the code get_flops.py from mmdetection-3.x to mmrotate-1.x'
        results = ()
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        # 特征维度转换调用 ChannelMapperWithGN(neck)，在 extract_feat 内部自动完成
        # 每个尺度的特征都是默认 256 维
        x = self.extract_feat(batch_inputs) # ResNet, LSKNet, SwinTransformer
        rpn_x = x
        roi_x = x
        
        # OrientedAdaMixerDDQ
        # imgs_whwht - 图像尺寸归一化基准
        # distinc_query_dict - 筛选后的 distinct query 这是 DDQ 的核心输出，
        #   包含 top-k 的 imgs_whwht 和 distinc_query_dict
        rpn_losses, imgs_whwht, distinc_query_dict = \
            self.rpn_head.predict(
                rpn_x, batch_img_metas)

        # query_xyzrt 和 query_content 在 oriented_adamixer_ddq 里面生成
        query_xyzrt = distinc_query_dict['query_xyzrt']     # 位置 queries
        query_content = distinc_query_dict['query_content'] # 内容 queries
        
        rpn_results_list = []
        for idx in range(len(batch_img_metas)):
            rpn_results = InstanceData()
            # 自动在 rpn_results 中创建 'query_xyzrt' 属性
            rpn_results.query_xyzrt = query_xyzrt[idx]
            # 根据 query_xyzrt[idx] 的长度（queries）个数，复制 len(query_xyzrt[idx]) 次
            # 对齐 queries 和 imgs_whwht 的维度，方便后面做诸如归一化的操作
            # （也其实可以不复制，用 pytorch 的传播机制就行）
            rpn_results.imgs_whwht = imgs_whwht[idx].repeat(
                len(query_xyzrt[idx]), 1)
            rpn_results.query_content = query_content[idx]
            rpn_results_list.append(rpn_results)

        # OrientedAdaMixerDecoder
        roi_outs = self.roi_head.forward(roi_x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )

        return results
