_base_ = [
    '../../configs/_base_/models/fpn_poolformer_s12.py',
    '../../configs/_base_/default_runtime.py',
]

norm_cfg               = dict(type='BN', requires_grad=True)
dataset_type           = 'LandcoverAI_CSV'
data_root              = '/misc/home1/m_imm_freedata/Segmentation/Projects/mmseg_water/landcover.ai_512'
   
num_classes            = 2
crop_size              = (512, 512)
batch_size             = 16

num_workers            = 12
max_epochs             = 25

gradient_accumulation_steps = 8
actual_batch_size = batch_size * gradient_accumulation_steps


loss = dict(type='FocalLoss', class_weight=[0.9, 1.1])

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_preprocessor = dict(size=crop_size)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/poolformer/poolformer-m48_3rdparty_32xb128_in1k_20220414-9378f3eb.pth' 

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        arch='m48',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,
            prefix='backbone.'
        )
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    decode_head=dict(num_classes=num_classes),
)
log_processor = dict(by_epoch=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=batch_size, num_workers=num_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root='',
        ann_file='',   
        data_prefix=dict(img_path='', seg_map_path=''),   
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=batch_size, num_workers=num_workers,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
                    img_path='val/images',
                    seg_map_path='val/gt'),        
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader
val_evaluator  = dict(type='IoUMetric', iou_metrics=['mIoU'])
val_cfg = dict(type='ValLoop')
test_evaluator = val_evaluator
test_cfg = dict(type='TestLoop')  
optimizer = dict(type='AdamW', lr=3e-4, weight_decay=0.001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None,  accumulative_counts=gradient_accumulation_steps)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=3e-2,
        begin=0,
        end=45,
        by_epoch=True,
    ),
    dict(
        type='PolyLRRatio',
        eta_min_ratio=3e-2,
        power=0.9,
        begin=45,
        end=90,
        by_epoch=True,
    ),
    dict(
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=90,
        end=100,
    )
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

default_hooks = dict(
    logger=dict(                    
        type='LoggerHook',
        interval=1,
        log_metric_by_epoch=False
    ),

    checkpoint=dict(                
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_last=True
    ),

    visualization=dict(
        type='SegVisualizationHook',
        interval=1
    )
)
custom_hooks = [
    dict(
        type='LoggerHook',
        interval=50,               
        log_metric_by_epoch=True,
    ),
]

vis_backends = [
    dict(                       
        type='LocalVisBackend',
        scalar_save_file='scalars.json'  
    ),
    dict(                      
        type='TensorboardVisBackend',
    )
]

visualizer = dict(
    _delete_=True,                    
    type='SegLocalVisualizer',
    vis_backends=[
        dict(
            type='LocalVisBackend',
            scalar_save_file='scalars.json'
        ),
        dict(
            type='TensorboardVisBackend',
        )
    ],
    name='visualizer'
)


