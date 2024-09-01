# #!/bin/bash
# gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# # gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-7b-lora-coco"
# SPLIT="llava_gqa_testdev_balanced"
# GQADIR="/your_data_path/data/gqa"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-lora-coco \
#         --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
#         --question-file /your_data_path/data/gqa/$SPLIT.jsonl \
#         --image-folder /your_data_path/data/gqa/images \
#         --answers-file /your_data_path/data/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=/your_data_path/data/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /your_data_path/data/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

# cd $GQADIR
# python eval/eval.py --tier testdev_balanced




#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# MASK="grafted_model_params_sparsity10_v1"
# CKPT="llava-v1.5-7b-lora-flickr30k2"
# LOAD="-tops-10"
LOAD=""
CKPT="llava-v1.5-7b-tune12layers-okvqa-v4-1e-4-v2-ours55-onlymask"
# CKPT="llava-v1.5-7b-tune12layers-flickr30k"
# CKPT="llava-v1.5-7b-lora-okvqa-1e-4"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v3-2e-5"
# LOAD="-tops-10-v1"
CKPTLOAD="${CKPT}${LOAD}"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/your_data_path//data/gqa"

# --masked-param-path /your_data_path//code/LLaVA/checkpoints/$CKPT/$MASK.pth \

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_origin \
        --model-path /your_data_path//code/LLaVA/checkpoints/$CKPT \
        --question-file /mnt/workspace/workgroup/honghaochen/datasets/llava_eval/gqa/$SPLIT.jsonl \
        --image-folder /mnt/workspace/workgroup/honghaochen/datasets/llava_instruct_tuning/gqa/images \
        --answers-file /your_data_path//data/gqa/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT \
#         --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
#         --question-file /your_data_path/data/gqa/$SPLIT.jsonl \
#         --image-folder /your_data_path/data/gqa/images \
#         --answers-file /your_data_path/data/gqa/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

wait

output_file=/your_data_path//data/gqa/answers/$SPLIT/$CKPTLOAD/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /your_data_path//data/gqa/answers/$SPLIT/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced