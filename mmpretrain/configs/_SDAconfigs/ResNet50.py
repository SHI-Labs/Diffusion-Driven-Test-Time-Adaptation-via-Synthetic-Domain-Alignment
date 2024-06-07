_base_ = [
    '../_base_/models/resnet50.py',           # model settings
    '../_base_/datasets/imagenet_bs32.py',    # data settings
    '../_base_/schedules/imagenet_bs256.py',  # schedule settings
    '../_base_/default_runtime.py',           # runtime settings
]

# model settings
load_from='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth '

# data settings
data_root = 'data/DiT_1_DDA/train'
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='',
    ))
val_dataloader = dict(
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='',
    ))
test_dataloader = val_dataloader

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=1e-4))