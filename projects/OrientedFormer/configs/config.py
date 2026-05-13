angle_version = 'le90'
backend_args = None
batch_size = 4
seed=42
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.OrientedFormer.orientedformer',
    ])
data_root = 'data/hrsc/'
dataset_type = 'HRSCDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        boxtype2tensor=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        num_outs=5,
        out_channels=256,
        type='ChannelMapperWithGN'),
    roi_head=dict(
        bbox_head=[
            dict(
                angle_version='le90',
                bbox_coder=dict(type='DeltaXYWHTRBBoxCoder'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                content_dim=256,
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                loss_bbox=dict(loss_weight=2.0, type='mmdet.L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='mmdet.FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(
                    loss_weight=5.0, mode='linear', type='RotatedIoULoss'),
                num_classes=1,
                num_cls_fcs=1,
                num_reg_fcs=1,
                o3d_attn_cfg=dict(
                    embed_dims=256,
                    n_heads=64,
                    n_points=32,
                    reduction=4,
                    type='OrientedAttention'),
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
                target_means=(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                target_stds=(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
                type='OrientedFormerDecoderLayer'),
            dict(
                angle_version='le90',
                bbox_coder=dict(type='DeltaXYWHTRBBoxCoder'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                content_dim=256,
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                loss_bbox=dict(loss_weight=2.0, type='mmdet.L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='mmdet.FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(
                    loss_weight=5.0, mode='linear', type='RotatedIoULoss'),
                num_classes=1,
                num_cls_fcs=1,
                num_reg_fcs=1,
                o3d_attn_cfg=dict(
                    embed_dims=256,
                    n_heads=64,
                    n_points=32,
                    reduction=4,
                    type='OrientedAttention'),
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
                target_means=(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                target_stds=(
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
                type='OrientedFormerDecoderLayer'),
        ],
        content_dim=256,
        featmap_strides=[
            4,
            8,
            16,
            32,
            64,
        ],
        num_stages=2,
        stage_loss_weights=[
            1,
            1,
        ],
        type='OrientedAdaMixerDecoder'),
    rpn_head=dict(
        angle_version='le90',
        aux_loss=dict(
            loss_bbox=dict(
                loss_weight=5.0, mode='linear', type='RotatedIoULoss'),
            loss_cls=dict(
                activated=True,
                beta=2.0,
                loss_weight=1.0,
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True),
            train_cfg=dict(
                alpha=1,
                assigner=dict(
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    iou_cost=dict(
                        iou_mode='iou', type='RotatedIoUCost', weight=5.0),
                    reg_cost=dict(
                        angle_version='le90',
                        box_format='xywht',
                        type='RBBoxL1Cost',
                        weight=2.0),
                    topk=8,
                    type='TopkHungarianAssigner'),
                beta=6)),
        ddq_num_classes=1,
        dqs_cfg=dict(iou_threshold=0.7, nms_pre=1000, type='nms_rotated'),
        feat_channels=256,
        in_channels=256,
        main_loss=dict(
            loss_bbox=dict(
                loss_weight=5.0, mode='linear', type='RotatedIoULoss'),
            loss_cls=dict(
                activated=True,
                beta=2.0,
                loss_weight=1.0,
                type='mmdet.QualityFocalLoss',
                use_sigmoid=True),
            train_cfg=dict(
                alpha=1,
                assigner=dict(
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    iou_cost=dict(
                        iou_mode='iou', type='RotatedIoUCost', weight=5.0),
                    reg_cost=dict(
                        angle_version='le90',
                        box_format='xywht',
                        type='RBBoxL1Cost',
                        weight=2.0),
                    topk=8,
                    type='TopkHungarianAssigner'),
                beta=6)),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_proposals=300,
        offset=0.5,
        strides=[
            4,
            8,
            16,
            32,
            64,
        ],
        type='OrientedAdaMixerDDQ'),
    test_cfg=dict(rcnn=dict(max_per_img=300), rpn=None),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=2.0),
                        dict(
                            angle_version='le90',
                            box_format='xywht',
                            type='RBBoxL1Cost',
                            weight=2.0),
                        dict(
                            iou_mode='iou', type='RotatedIoUCost', weight=5.0),
                    ],
                    type='mmdet.HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='mmdet.PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=2.0),
                        dict(
                            angle_version='le90',
                            box_format='xywht',
                            type='RBBoxL1Cost',
                            weight=2.0),
                        dict(
                            iou_mode='iou', type='RotatedIoUCost', weight=5.0),
                    ],
                    type='mmdet.HungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='mmdet.PseudoSampler')),
        ],
        rpn=None),
    type='OrientedDDQRCNN')
num_classes = 1
num_proposals = 300
num_stages = 2
num_workers = 2
optim_wrapper = dict(
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(lr=5e-05, type='AdamW', weight_decay=1e-06),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='ImageSets/test.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='FullDataSet/'),
        data_root='data/hrsc/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                512,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='HRSCDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(metric='mAP', type='DOTAMetric')
test_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        512,
    ), type='mmdet.Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=4,
    dataset=dict(
        ann_file='ImageSets/trainval.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='FullDataSet/'),
        data_root='data/hrsc/',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(keep_ratio=True, scale=(
                800,
                512,
            ), type='mmdet.Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='HRSCDataset'),
    drop_last=True,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True, scale=(
        800,
        512,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='ImageSets/test.txt',
        backend_args=None,
        data_prefix=dict(sub_data_root='FullDataSet/'),
        data_root='data/hrsc/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                512,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='HRSCDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metric='mAP', type='DOTAMetric')
val_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        800,
        512,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_hrsc'
