_base_ = [
    '../_base_/models/convnext/convnext-tiny.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py',
]

# model settings
load_from = 'https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth'

# data settings
data_root = 'data/DiT-Syn'

train_dataloader = dict(
    batch_size=128,
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='',
    ))
val_dataloader = dict(
    batch_size=128,
    dataset=dict(
        type='ImageNet',
        data_root=data_root,
        ann_file='',       # We assume you are using the sub-folder format without ann_file
        data_prefix='',
    ))
test_dataloader = val_dataloader

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))