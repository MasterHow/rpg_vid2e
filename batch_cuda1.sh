#!/bin/bash
python Reorganize_Image.py --output_prefix images_reorganize_cuda1 --seq 04 05 06 07 08 09 10

cd /workspace/mnt/storage/shihao/EventSSC/TempCode/rpg_vid2e/upsampling/

CUDA_VISIBLE_DEVICES=1 python upsample.py --input_dir=/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/images_reorganize_cuda1 --output_dir=/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/imageFiles_Upsample_cuda1

cd /workspace/mnt/storage/shihao/EventSSC/TempCode/rpg_vid2e

CUDA_VISIBLE_DEVICES=1 python ./esim_torch/scripts/generate_events.py --input_dir=/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/imageFiles_Upsample_cuda1 --output_dir=/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/events
