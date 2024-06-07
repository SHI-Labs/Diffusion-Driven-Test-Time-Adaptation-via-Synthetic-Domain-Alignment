#!/bin/bash

# download checkpoints in the 'pretrained_ckpt' folder
mkdir pretrained_ckpt
cd pretrained_ckpt

# conditional diffusion model DiT
wget https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt

# unconditional diffusion model ADM
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

# recognition model
wget https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth  # resnet50
wget https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth  # swinT
wget https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth  # convnextT
wget https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth  # swinB
wget https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_3rdparty_32xb128_in1k_20220124-d0915162.pth  # convnextB