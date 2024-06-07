# Model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ConvNeXt', arch='base', drop_path_rate=0.8),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.)
    # train_cfg=dict(augments=[
    #     dict(type='Mixup', alpha=0),
    #     dict(type='CutMix', alpha=0),
    # ]),
)
