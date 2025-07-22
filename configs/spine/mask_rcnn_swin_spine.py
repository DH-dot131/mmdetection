# configs/spine/mask_rcnn_swin_spine.py

_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
optim_wrapper = dict(type='AmpOptimWrapper')

# Swin-T 백본 설정
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(num_classes=29),  # spine instance 수에 맞게 수정
        mask_head=dict(num_classes=29)
    )
)

# 데이터셋 설정
dataset_type = 'CocoDataset'
data_root = '/content/data/processed/coco_annotations/'  # 실제 이미지가 존재하는 경로
raw_data_root= '/content/full_2500/'  # 원본 이미지가 존재하는 경로
metainfo = {
    'classes': (
        'Hip Joint', 'Femur Head', 'S1', 'L5', 'L4', 'L3', 'L2', 'L1',
        'L1~L2 disc', 'L2~L3 disc', 'L3~L4 disc', 'L4~L5 disc', 'L5~S1 disc',
        'Lt L1~L2 foramen', 'Rt L1~L2 foramen', 'Lt L2~L3 foramen', 'Rt L2~L3 foramen',
        'Lt L3~L4 foramen', 'Rt L3~L4 foramen', 'Lt L4~L5 foramen', 'Rt L4~L5 foramen',
        'Lt L5~S1 foramen', 'Rt L5~S1 foramen', 'L1 pedicle', 'L2 pedicle',
        'L3 pedicle', 'L4 pedicle', 'L5 pedicle', 'S1 pedicle'
    )
}

train_dataloader = dict(
        batch_size=8,
        num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        data_prefix=dict(img = raw_data_root),
        metainfo=metainfo,
    )
)
val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        data_prefix=dict(img = raw_data_root),
        metainfo=metainfo,
    )
)
test_dataloader = dict(
        batch_size=2,
        num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        data_prefix=dict(img = raw_data_root),
        metainfo=metainfo,
    )
)

# 학습 설정
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

# Optimizer & Scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[27, 33], gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
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

# ✅ 아예 default_hooks에서 evaluation 제거
default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50,  # 몇 iteration마다 로그를 기록할지
        log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', interval=3)
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='TensorboardVisBackend', save_dir='/content/drive/MyDrive/Spine_LAT_segmentation/work_dirs/tb_logs'),
        # dict(type='LocalVisBackend')  # 필요 시 이미지 저장용
    ],
    alpha = 0.3,
    name='visualizer'
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

work_dir = '/content/drive/MyDrive/Spine_LAT_segmentation/work_dirs/mask_rcnn_swin_spine'
