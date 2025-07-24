# configs/spine/mask_rcnn_swin_spine.py

_base_ = 'mask_rcnn_hrnet_spine.py'


test_dataloader = dict(
        batch_size=4,
        num_workers=4,
    dataset=dict(
        data_root = None,
        ann_file= '/content/drive/MyDrive/Spine/Spine_LAT_segmentation/data_LSTV_cls/inference_LSTV.json',
        data_prefix=dict(img = "./"),
    )
)



test_evaluator = dict(
    type='CocoMetric',
    ann_file= '/content/drive/MyDrive/Spine/Spine_LAT_segmentation/data_LSTV_cls/inference_LSTV.json',
)

work_dir = '/content/drive/MyDrive/Spine/Spine_LAT_segmentation/work_dirs/mask_rcnn_hrnet_spine_for_LSTV_inference'