#!/bin/bash

model=ResNet50 # model=[ResNet50, SwinT, SwinB, ConvNeXtT, ConvNeXtB]

bash mmpretrain/tools/dist_train.sh mmpretrain/configs/_SDAconfigs/$model.py 8
