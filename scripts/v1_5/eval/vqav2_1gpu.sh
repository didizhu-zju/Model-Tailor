#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

for gpu in "${GPULIST[@]}"; do
    echo "GPU: $gpu"
done


CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-7b"
CKPT="llava-v1.5-7b-tune12layers-flickr30k"
SPLIT="llava_vqav2_mscoco_test-dev2015"

IDX=1

CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
    --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-tune12layers-flickr30k \
    --question-file /your_data_path/data/vqav2/$SPLIT.jsonl \
    --image-folder /your_data_path/data/coco/images/test2015 \
    --answers-file /your_data_path/data/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode vicuna_v1 &
