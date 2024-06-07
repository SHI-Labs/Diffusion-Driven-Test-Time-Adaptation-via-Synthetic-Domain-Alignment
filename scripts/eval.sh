#!/bin/bash

# ImageNet-C
CUDA_VISIBLE_DEVICES=0 python eval/test_ensemble.py eval/configs/ensemble/convnextB.py \
                finetuned_ckpt/ConvNeXtB.pth --originckpt pretrained_ckpt/convnext-base_3rdparty_32xb128_in1k_20220124-d0915162.pth \
                --metrics accuracy --datatype C --ensemble sda --corruption gaussian_noise  --data_prefix1 data/ImageNet-C --data_prefix2 data/ImageNet-C-Syn

# ImageNet-W
CUDA_VISIBLE_DEVICES=0 python eval/test_ensemble.py eval/configs/ensemble/convnextB.py \
                finetuned_ckpt/ConvNeXtB.pth --originckpt pretrained_ckpt/convnext-base_3rdparty_32xb128_in1k_20220124-d0915162.pth \
                --metrics accuracy --datatype W --ensemble sda --data_prefix1 data/ImageNet-W --data_prefix2 data/ImageNet-W-Syn
