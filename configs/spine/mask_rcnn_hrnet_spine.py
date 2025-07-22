# configs/spine/mask_rcnn_swin_spine.py

_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

backend_args = None


# HRNet 백본 설정
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

'''
albu_train_transforms = [
    dict(type='ShiftScaleRotate',
         shift_limit=0.05, scale_limit=0.0, rotate_limit=5, p=0.5),
    dict(type='RandomBrightnessContrast',
         brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    dict(type='RandomGamma', gamma_limit=(80, 120), p=0.2),
    dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.2),
    dict(type='GaussianBlur', blur_limit=2, p=0.2),
    dict(type='ElasticTransform',
         alpha=1, sigma=50, alpha_affine=50, p=0.2),
    dict(type='ImageCompression',
         quality_lower=90, quality_upper=100, p=0.1),
]


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),

    dict(
      type='FilterAnnotations',
      min_gt_bbox_wh=(1, 1),
      by_mask=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
'''
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
                   
model = dict(
    backbone=dict(
        _delete_=True,
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w32')),
    neck=dict(
        _delete_=True,
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256),
    roi_head=dict(
        bbox_head=dict(num_classes=29),  # spine instance 수에 맞게 수정
        mask_head=dict(num_classes=29)
    )
        )
# 데이터셋 설정
dataset_type = 'CocoDataset'
data_root = '/content/data/processed/coco_annotations/'  # json이 존재하는 경로
raw_data_root= '/content/full_2500_xrays/'  # 원본 이미지가 존재하는 경로
metainfo = {
    'classes': (
        'Femur Head_A', 'Femur Head_B', 'S1', 'L5', 'L4', 'L3', 'L2', 'L1',
        'L1~L2 disc', 'L2~L3 disc', 'L3~L4 disc', 'L4~L5 disc', 'L5~S1 disc',
        'Lt L1~L2 foramen', 'Rt L1~L2 foramen', 'Lt L2~L3 foramen', 'Rt L2~L3 foramen',
        'Lt L3~L4 foramen', 'Rt L3~L4 foramen', 'Lt L4~L5 foramen', 'Rt L4~L5 foramen',
        'Lt L5~S1 foramen', 'Rt L5~S1 foramen', 'L1 pedicle', 'L2 pedicle',
        'L3 pedicle', 'L4 pedicle', 'L5 pedicle', 'S1 pedicle'
    )
}

train_dataloader = dict(
        batch_size=8,
        num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        data_prefix=dict(img = raw_data_root),
        metainfo=metainfo,
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        data_prefix=dict(img = raw_data_root),
        metainfo=metainfo,
        pipeline=test_pipeline
    )
)
test_dataloader = dict(
        batch_size=4,
        num_workers=4,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        data_prefix=dict(img = raw_data_root),
        metainfo=metainfo,
        pipeline=test_pipeline
    )
)

# 학습 설정
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs, val_interval=3)

# Optimizer & Scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[24, 33], gamma=0.1)
]

# optimizer
'''
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
'''
auto_scale_lr = dict(enable=False, base_batch_size=16)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05)
)

# ✅ 대신 evaluator 명시
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric=['bbox', 'segm']
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric=['bbox', 'segm']
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend')  # 필요 시 이미지 저장용
    ],
    alpha = 0.3,
    name='visualizer'
)