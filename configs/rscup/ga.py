# model settings
fp16 = dict(loss_scale=512.)
model = dict(
    type='RPN',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        dcn=dict(
            modulated=False, deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='GARPNHead',
        in_channels=256,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        octave_ratios=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 7.0],
        anchor_strides=[4, 8, 16, 32, 64],
        anchor_base_sizes=None,
        anchoring_means=[.0, .0, .0, .0],
        anchoring_stds=[0.07, 0.07, 0.14, 0.14],
        target_means=(.0, .0, .0, .0),
        target_stds=[0.07, 0.07, 0.11, 0.11],
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        ga_assigner=dict(
            type='ApproxMaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        ga_sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        center_ratio=0.3,
        ignore_ratio=0.5,
        debug=False),
)
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=500,
        nms_thr=0.7,
        min_bbox_size=0),
)
# dataset settings
dataset_type = 'CocoDataset'
data_root = './data/rscup/'
aug_root = "./data/rscup/aug/"
other_aug_root = "./data/rscup/otheraug/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=(data_root + 'annotation/annos_rscup_train.json',
                  aug_root + 'annos_rscup_airport.json',
                  other_aug_root + "annos_rscup_baseball-diamond.json",
                  other_aug_root + "annos_rscup_basketball-court.json",
                  other_aug_root + "annos_rscup_container-crane.json",
                  other_aug_root + "annos_rscup_helicopter.json",
                  other_aug_root + "annos_rscup_helipad.json",
                  other_aug_root + "annos_rscup_helipad_ship.json",
                  other_aug_root + "annos_rscup_roundabout.json",
                  other_aug_root + "annos_rscup_soccer-ball-field_ground-track-field.json",
        ),
        img_prefix=(data_root + 'train/',
                    aug_root + "airport/",
                    other_aug_root + "baseball-diamond",
                    other_aug_root + "basketball-court",
                    other_aug_root + "container-crane",
                    other_aug_root + "helicopter",
                    other_aug_root + "helipad",
                    other_aug_root + "helipad_ship",
                    other_aug_root + "roundabout",
                    other_aug_root + "soccer-ball-field_ground-track-field"),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/annos_rscup_val.json',
        img_prefix=data_root + 'val/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file='./data/rscup/debug.json',
        img_prefix='./data/rscup/debug/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=3e-3, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[2, 4, 6])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 8
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ga'
load_from = None
resume_from = None
workflow = [('train', 1)]
