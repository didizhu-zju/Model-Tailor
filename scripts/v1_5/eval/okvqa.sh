# python -m llava.eval.model_vqa_loader \
#     --model-path /your_data_path/ft_local/models/llava-v1.5-7b \
#     --question-file /your_data_path/data/okvqa/llava_okvqa_question.jsonl \
#     --image-folder /your_data_path/data/coco/images \
#     --answers-file /your_data_path/data/okvqa/answers/llava-v1.5-7b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1


#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

for gpu in "${GPULIST[@]}"; do
    echo "GPU: $gpu"
done

CHUNKS=${#GPULIST[@]}

# MASK="grafted_model_params_sparsity10_v1"
# CKPT="llava-v1.5-7b"
SPLIT="llava_okvqa_question"
CKPT="llava-v1.5-7b-tune12layers-okvqa-v4-1e-4-v2-ours55-onlymask"
# CKPT="llava-v1.5-7b-tune12layers-flickr30k"

# LOAD="-tops-10_v1"
LOAD=""
# CKPT="llava-v1.5-7b-lora-okvqa-1e-4"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v3-2e-5"
# LOAD="-tops-5"
CKPTLOAD="${CKPT}${LOAD}"

# --masked-param-path /your_data_path//code/LLaVA/checkpoints/$CKPT/$MASK.pth \

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_origin \
        --model-path /your_data_path//code/LLaVA/checkpoints/$CKPT \
        --question-file /your_data_path//data/llava_okvqa_question.jsonl \
        --image-folder /mnt/workspace/workgroup/gongyuan.yjl/data/multimodel/lavis/coco/images \
        --answers-file /your_data_path//data/okvqa/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path /your_data_path//code/LLaVA/checkpoints/$CKPT \
#         --model-base /your_data_path//models/llava_v1_5_7b \
#         --masked-param-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT/$MASK.pth \
#         --question-file /your_data_path/data/okvqa/llava_okvqa_question.jsonl \
#         --image-folder /your_data_path/data/coco/images \
#         --answers-file /your_data_path/data/okvqa/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

wait

output_file=/your_data_path//data/okvqa/answers/$SPLIT/$CKPTLOAD/merge.jsonl

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /your_data_path//data/okvqa/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done