# from torch.optim.adamw import AdamW
# from mmengine.config import read_base
# from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
# from mmengine.optim.optimizer import OptimWrapper
# from mmengine.optim.scheduler.lr_scheduler import LinearLR
# from mmengine.hooks.ema_hook import EMAHook
# from mmcv.transforms import LoadImageFromFile, RandomApply
# from mmdet.models.data_preprocessors import DetDataPreprocessor
# from mmdet.models.necks import ChannelMapper
# from mmdet.models.losses import L1Loss, GIoULoss
# from mmdet.models.task_modules import FocalLossCost, BBoxL1Cost, IoUCost, HungarianAssigner
# from mmdet.models.layers.ema import ExpMomentumEMA
# from mmdet.datasets.transforms import (FilterAnnotations, Resize,
#                                        RandomFlip, PackDetInputs, LoadAnnotations)
# from projects.rtdetr.rtdetr import (BatchSyncRandomResize, PhotoMetricDistortion, MinIoURandomCrop, Expand,
#                                     RTDETR, RTDETRFPN, RTDETRHead, RTDETRVarifocalLoss, ResNetV1dPaddle)


# with read_base():
#     from .coco_detection import *
#     from .default_runtime import *

_base_ = [
    '../../../configs/_base_/datasets/dior.py',
    # '../../../configs/_base_/schedules/schedule_2x.py',
    '../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.rtdetr_o.rtdetr'], allow_failed_imports=False)

pretrained = ('https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/'
              'master/rtdetr/resnet50vd_ssld_v2_pretrained_d037e232.pth')  # noqa

base_size_repeat = 3

angle_cfg = dict(
    width_longer=True,
    start_angle=0,
)
angle_factor=3.1415926535897932384626433832795
angle_version = 'le90'

batch_size = 8
batch_size_val = 8
num_workers = 4
max_epochs = 96

model = dict(
    type='RotatedRTDETR',
    num_queries=300,  # num_matching_queries, 900 for DINO
    # spatial_shapes=((80, 80), (40, 40), (
    #     20, 20)),  # for strdies (8, 16, 32) with image_size 640x640. # noqa
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        batch_augments=[
            dict(
                type='RotatedBatchSyncRandomResize',
                interval=1,
                interpolations='nearest',
                random_sizes=[480, 512, 544, 576, 608] +
                [640] * base_size_repeat + [672, 704, 736, 768, 800])
        ],
        mean=[0, 0, 0],  # [123.675, 116.28, 103.53] for DINO
        std=[255, 255, 255],  # [58.395, 57.12, 57.375] for DINO
        bgr_to_rgb=True,
        pad_size_divisor=1,
        boxtype2tensor=False),
    backbone=dict(
        type='ResNetV1dPaddle',  # ResNet for DINO
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=0,  # -1 for DINO
        norm_cfg=dict(type='BN', requires_grad=False),  # BN for DINO
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),  # GN for DINO
        num_outs=3,  # 4 for DINO
        init_cfg=dict(
            type='Kaiming',
            layer='Conv2d',
            a=5**0.5,
            distribution='uniform',
            mode='fan_in',
            nonlinearity='leaky_relu')),
    encoder=dict(
        use_encoder_idx=[-1],
        num_encoder_layers=1,
        in_channels=[256, 256, 256],
        fpn_cfg=dict(
            type='RTDETRFPN',
            in_channels=[256, 256, 256],
            out_channels=256,
            expansion=1.0,
            norm_cfg=dict(type='BN', requires_grad=True)),
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 2048 for DINO
                ffn_drop=0.0,
                act_cfg=dict(type='GELU')))),  # ReLU for DINO
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=3,  # 4 for DINO
                dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,  # 2048 for DINO
                ffn_drop=0.0)),
        post_norm_cfg=None),
    bbox_head=dict(
        ######################## 这里要改 ###################################################################
        type='RotatedRTDETRHead',
        num_classes=20,
        angle_cfg=angle_cfg,
        angle_factor=angle_factor,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            # type='RTDETRVarifocalLoss',  # FocalLoss in DINO
            # use_sigmoid=True,
            # alpha=0.75,
            # gamma=2.0,
            # iou_weighted=True,
            # loss_weight=1.0),
            type='mmdet.FocalLoss',  # FocalLoss in DINO
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=5.0),
        ######################## 这里要改 ####################################################################
        loss_iou=dict(
            type='GDLoss',
            loss_type='kld',
            fun='log1p',
            tau=1,
            sqrt=False,
            loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
        ######################## 这里要改 ####################################################################
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(type='mmdet.FocalLossCost', weight=2.0),
                # dict(type='RBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='RBoxL1Cost', weight=5.0, box_format='xywha', angle_factor = angle_factor),
                # dict(type='mmdet.IoUCost', iou_mode='giou', weight=2.0)
                dict(
                    type='GDCost', 
                    loss_type='kld', 
                    fun='log1p', 
                    tau=1, 
                    sqrt=False,
                    weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
backend_args = None
_base_.train_pipeline = None
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    ######################## 这里要改 ####################################################################
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='PhotoMetricDistortion',
        hue_delta=12.75,
        clip_val=255,
        force_float32=False),
    dict(type='Expand', mean=[0, 0, 0]),
    dict(
        type='mmdet.RandomApply',
        transforms=dict(
            type='MinIoURandomCrop', cover_all_box=False, trials=40),
        prob=0.8),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=False),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]

_base_.val_pipeline = None
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=False),
    # avoid bboxes being resized
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = [
    # dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(640, 640), keep_ratio=True),
    # dict(
    #     type='mmdet.PackDetInputs',
    #     meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
    #                'scale_factor'))
]


train_dataloader = dict(
    drop_last=True,
    pin_memory=True,
    batch_size=batch_size,
    num_workers=num_workers,
)

val_dataloader = dict(
    batch_size=batch_size_val,
    num_workers=num_workers,)

test_dataloader = val_dataloader

# # optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1.0e-4, weight_decay=1e-6),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'in_proj_bias': dict(decay_mult=0)
        },
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True))

# learning policy
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000)
# ]
# optimizer
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

# optim_wrapper = dict(
#     optimizer=dict(
#         _delete_=True, type='AdamW', lr=1.0e-4, weight_decay=1e-6),  # 1 RTX 4090
#     clip_grad=dict(max_norm=1, norm_type=2))
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
auto_scale_lr = dict(enable=False, base_batch_size=batch_size)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

# load_from = 'models/rtdetr_r50vd_8xb2-72e_coco_ad2bdcfe.pth'