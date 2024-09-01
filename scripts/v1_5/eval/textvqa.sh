#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-lora-coco \
#     --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
#     --question-file /your_data_path/data/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder /your_data_path/data/textvqa/train_images \
#     --answers-file /your_data_path/data/textvqa/answers/llava-v1.5-7b-lora-coco.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa --annotation-file /your_data_path/data/textvqa/TextVQA_0.5.1_val.json --result-file /your_data_path/data/textvqa/answers/llava-v1.5-7b-lora-coco.jsonl


#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

for gpu in "${GPULIST[@]}"; do
    echo "GPU: $gpu"
done

CHUNKS=${#GPULIST[@]}

# MASK="flickr-sqa-avg_sparse_lora_params_sparsity20"
# MASK="flickr-sqa-sparse_avg_lora_params_sparsity20_v5"
MASK="filckr-sqa-sparse_lora_fusion_sparsity50"
# MASK="sparse_lora_params_sparsity20"
LOAD="-flickr-sqa-sparse-fusion-50"
# LOAD=""
# CKPT=""
# CKPT="llava-v1.5-7b-lora-flickr30k"
# CKPT="llava-v1.5-7b-lora-sqa-instruction2"
CKPT="fused_lora"
# LOAD="llava_v1_5_7b"
CKPTLOAD="${CKPT}${LOAD}"
SPLIT="llava_textvqa_val_v051_ocr"

# --masked-param-path /your_data_path//code/LLaVA/checkpoints/$CKPT/$MASK.pth \

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path /your_data_path//code/LLaVA/checkpoints/$CKPT \
#         --model-base /your_data_path//models/llava_v1_5_7b \
#         --question-file /mnt/workspace/workgroup/honghaochen/datasets/llava_eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#         --image-folder /mnt/workspace/workgroup/honghaochen/datasets/llava_eval/textvqa/train_images \
#         --answers-file /your_data_path//data/textvqa/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /your_data_path//models/llava_v1_5_7b \
        --masked-param-path /your_data_path//code/LLaVA/checkpoints/$CKPT/$MASK.pth \
        --question-file /mnt/workspace/workgroup/honghaochen/datasets/llava_eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
        --image-folder /mnt/workspace/workgroup/honghaochen/datasets/llava_eval/textvqa/train_images \
        --answers-file /your_data_path//data/textvqa/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/your_data_path//data/textvqa/answers/$SPLIT/$CKPTLOAD/merge.jsonl

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /your_data_path//data/textvqa/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m llava.eval.eval_textvqa --annotation-file /mnt/workspace/workgroup/honghaochen/datasets/llava_eval/textvqa/TextVQA_0.5.1_val.json --result-file $output_file
