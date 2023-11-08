#!/bin/bash
#python Reorganize_Image.py --output_prefix images_reorganize_cuda1 --seq 01 02 03

#cd /workspace/mnt/storage/shihao/EventSSC/TempCode/rpg_vid2e/upsampling/
#
#CUDA_VISIBLE_DEVICES=0 python upsample.py --input_dir=/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/images_reorganize_cuda0 --output_dir=/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/imageFiles_Upsample_cuda0
#
#cd /workspace/mnt/storage/shihao/EventSSC/TempCode/rpg_vid2e
#
#CUDA_VISIBLE_DEVICES=0 python ./esim_torch/scripts/generate_events.py --input_dir=/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/imageFiles_Upsample_cuda0 --output_dir=/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/events

#CUDA_VISIBLE_DEVICES=0 python generate_timestamp.py --seq 00 01 02
#
#CUDA_VISIBLE_DEVICES=0 python merge_events.py --seq 00 01 02 03

CUDA_VISIBLE_DEVICES=0 python generate_event_frame.py --seq 00 01 02 03



