#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --master_port 21420 --nnodes 1 --nproc_per_node=5 image_gen_DiT/sample_ddp.py \
                            --image-size 256 --sample-dir data --cfg-scale 1.0 --ckpt pretrained_ckpt/DiT-XL-2-256x256.pt \
                            --global-seed 0 --num-fid-samples 50000