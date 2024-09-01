# #!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-0,2,3,4,5,6,7}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# for gpu in "${GPULIST[@]}"; do
#     echo "GPU: $gpu"
# done


# CHUNKS=${#GPULIST[@]}

# # CKPT="llava-v1.5-7b"
# CKPT="llava-v1.5-7b-lora-coco"
# SPLIT="llava_vqav2_mscoco_test-dev2015"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-lora-coco \
#         --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
#         --question-file /your_data_path/data/vqav2/$SPLIT.jsonl \
#         --image-folder /your_data_path/data/coco/images/test2015 \
#         --answers-file /your_data_path/data/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=/your_data_path/data/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /your_data_path/data/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_vqav2_for_submission.py --dir "/your_data_path/data/vqav2" --split $SPLIT --ckpt $CKPT





#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

for gpu in "${GPULIST[@]}"; do
    echo "GPU: $gpu"
done


CHUNKS=${#GPULIST[@]}

MASK="grafted_model_params_sparsity5"
# CKPT="llava-v1.5-7b-lora-flickr30k2"
LOAD="-tops-5"
# MASK="grafted_model_params_sparsity10_v1"
CKPT="llava-v1.5-7b-lora-okvqa-1e-4"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v3-2e-5"
# CKPT="llava-v1.5-7b-tune12layers-flickr30k"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v4-1e-4"
# CKPT="llava-v1.5-7b"
# LOAD="-tops-10-v1"
# LOAD="-tops-10"
SPLIT="llava_vqav2_mscoco_test-dev2015"
CKPTLOAD="${CKPT}${LOAD}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT \
        --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
        --masked-param-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT/$MASK.pth \
        --question-file /your_data_path/data/vqav2/$SPLIT.jsonl \
        --image-folder /your_data_path/data/coco/images/test2015 \
        --answers-file /your_data_path/data/vqav2/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/your_data_path/data/vqav2/answers/$SPLIT/$CKPTLOAD/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /your_data_path/data/vqav2/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --dir "/your_data_path/data/vqav2" --split $SPLIT --ckpt $CKPTLOAD

