_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model settings
load_from='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth'


# data settings
data_root = 'data/DiT-Syn'
train_dataloader = dict(
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

optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
auto_scale_lr = dict(base_batch_size=256)
