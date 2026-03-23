from mmrotate.registry import MODELS
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.models.utils.misc import unpack_gt_instances
from mmdet.structures import SampleList
from mmengine.structures import InstanceData
from torch import Tensor

@MODELS.register_module()
class OrientedDDQRCNN(TwoStageDetector):

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
        '''
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        gt_bboxes, gt_labels = [], []
        for i in range(len(batch_gt_instances)):
            gt_bboxes.append(batch_gt_instances[i].bboxes)
            gt_labels.append(batch_gt_instances[i].labels)

        losses = dict()
        x = self.extract_feat(batch_inputs) # list(level), each level has shape (bs, c, h, w)
        rpn_x = x
        roi_x = x

        rpn_losses, imgs_whwht, distinc_query_dict = \
            self.rpn_head.loss_and_predict(
                rpn_x,
                batch_img_metas,
                gt_bboxes,
                gt_labels)
        query_xyzrt = distinc_query_dict['query_xyzrt']
        query_content = distinc_query_dict['query_content']

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

        roi_losses = self.roi_head.loss(
            roi_x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses)

        return losses

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

        results_list = self.roi_head.predict(roi_x,
                                             rpn_results_list,
                                             batch_data_samples,
                                             rescale=rescale)
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
        rpn_losses, imgs_whwht, distinc_query_dict = \
            self.rpn_head.predict(
                rpn_x, batch_img_metas)

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
