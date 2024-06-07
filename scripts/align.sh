#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

# Synthetic Dataset Process

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpiexec -n 8 python image_align_DDA/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 10 --num_samples 50000 --timestep_respacing 100 \
                            --model_path pretrained_ckpt/256x256_diffusion_uncond.pt --base_samples data/DiT-XL-2-DiT-XL-2-256x256-size-256-vae-ema-cfg-1.0-seed-0 \
                            --D 4 --N 50 --scale 6 --datatype D \
                            --save_dir data/

# ImageNet-C Process

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpiexec -n 8 python image_align_DDA/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 10 --num_samples 50000 --timestep_respacing 100 \
                            --model_path pretrained_ckpt/256x256_diffusion_uncond.pt --base_samples data/ImageNet-C \
                            --D 4 --N 50 --scale 6 --datatype C \
                            --corruption gaussian_noise --severity 5 \
                            --save_dir data/

# ImageNet-W Process

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpiexec -n 8 python image_align_DDA/scripts/image_sample.py $MODEL_FLAGS \
                            --batch_size 10 --num_samples 50000 --timestep_respacing 100 \
                            --model_path DDA/ckpt/256x256_diffusion_uncond.pt --base_samples data/ImageNet-W \
                            --D 4 --N 50 --scale 6 --datatype W \
                            --save_dir data/