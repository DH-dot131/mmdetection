# configs/spine/mask_rcnn_swin_spine.py

_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'


# Swin-T 백본 설정

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
data_root = '/content/data/processed/coco_annotations/'  # 실제 이미지가 존재하는 경로
raw_data_root= '/content/full_2500/'  # 원본 이미지가 존재하는 경로
HeungK_raw_data_root = '/content/drive/MyDrive/Spine/Spine_LAT_segmentation/data_HeungK/'
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
        batch_size=2,
        num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        data_prefix=dict(img = raw_data_root),
        metainfo=metainfo,
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'val.json',
        data_prefix=dict(img = raw_data_root),
        metainfo=metainfo,
    )
)

'''
test_dataloader = dict(
        batch_size=1,
        num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        data_prefix=dict(img = raw_data_root),
        metainfo=metainfo,
    )
)
'''

test_dataloader = dict(
        batch_size=1,
        num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'HeungK_external_coco_annotations.json',
        data_prefix=dict(img = HeungK_raw_data_root),
        metainfo=metainfo,
    )
)

# 학습 설정
max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

# Optimizer & Scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[27, 33], gamma=0.1)
]
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

# ✅ 아예 default_hooks에서 evaluation 제거
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5)
)

# ✅ 대신 evaluator 명시
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val.json',
    metric=['bbox', 'segm']
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'HeungK_external_coco_annotations.json',
    metric=['bbox', 'segm']
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        #dict(type='TensorboardVisBackend', save_dir='/content/drive/MyDrive/Spine_LAT_segmentation/work_dirs/mask_rcnn_hrnet_spine'),
        dict(type='LocalVisBackend')  # 필요 시 이미지 저장용
    ],
    alpha = 0.3,
    name='visualizer'
)