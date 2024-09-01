#!/bin/bash

MASK="filckr-sqa-sparse_lora_fusion_sparsity50"
# MASK="flickr-sqa-fused_sparse_lora_params_sparsity20"
# MASK="flickr-sqa-avg_sparse_lora_params_sparsity20"
# MASK="avg_sparse_lora_params_sparsity20"
# MASK="fused_sparse_lora_params_sparsity20_v2"
# MASK="sparse_lora_params_sparsity20"
# CKPT="llava-v1.5-7b-lora-flickr30k"
# LOAD="-fs-fused-sparse-20"
# LOAD="-fs-avg-sparse-20"
LOAD="-flickr-sqa-sparse-avg-50"
# LOAD=""
# CKPT="llava-v1.5-7b"
CKPT="fused_lora"
# CKPT="llava-v1.5-7b-lora-textvqa-instruction"
# CKPT="llava-v1.5-7b-lora-flickr30k"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v4-1e-4"
CKPTLOAD="${CKPT}${LOAD}"

# python -m llava.eval.model_vqa_science \
#     --model-path /your_data_path//code/LLaVA/checkpoints/$CKPT \
#     --model-base /your_data_path//models/llava_v1_5_7b \
#     --question-file /your_data_path//data/scienceqa/llava_test_CQM-A.json \
#     --image-folder /your_data_path//data/scienceqa/images/test \
#     --answers-file /your_data_path//data/scienceqa/answers/$CKPTLOAD.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# --masked-param-path /your_data_path//code/LLaVA/checkpoints/$CKPT/$MASK.pth \

python -m llava.eval.model_vqa_science \
    --model-path /your_data_path//models/llava_v1_5_7b \
    --question-file /your_data_path//data/scienceqa/llava_test_CQM-A.json \
    --masked-param-path /your_data_path//code/LLaVA/checkpoints/$CKPT/$MASK.pth \
    --image-folder /your_data_path//data/scienceqa/images/test \
    --answers-file /your_data_path//data/scienceqa/answers/$CKPTLOAD.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /your_data_path//data/scienceqa \
    --result-file /your_data_path//data/scienceqa/answers/$CKPTLOAD.jsonl \
    --output-file /your_data_path//data/scienceqa/answers/$CKPTLOAD_output.jsonl \
    --output-result /your_data_path//data/scienceqa/answers/$CKPTLOAD_result.json





# #!/bin/bash

# gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# for gpu in "${GPULIST[@]}"; do
#     echo "GPU: $gpu"
# done

# CHUNKS=${#GPULIST[@]}

# MASK="flickr-sqa-sparse_avg_lora_params_sparsity20_v2"
# # MASK="flickr-sqa-fused_sparse_lora_params_sparsity20"
# # MASK="flickr-sqa-avg_sparse_lora_params_sparsity20"
# # MASK="avg_sparse_lora_params_sparsity20"
# # MASK="fused_sparse_lora_params_sparsity20_v2"
# # MASK="sparse_lora_params_sparsity20"
# # CKPT="llava-v1.5-7b-lora-flickr30k"
# # LOAD="-fs-fused-sparse-20"
# # LOAD="-fs-avg-sparse-20"
# LOAD="-flickr-sqa-sparse-avg-20"
# # LOAD=""
# # CKPT="llava-v1.5-7b"
# CKPT="fused_lora"
# # CKPT="llava-v1.5-7b-lora-textvqa-instruction"
# # CKPT="llava-v1.5-7b-lora-flickr30k"
# # CKPT="llava-v1.5-7b-tune12layers-okvqa-v4-1e-4"
# CKPTLOAD="${CKPT}${LOAD}"

# # python -m llava.eval.model_vqa_science \
# #     --model-path /your_data_path//code/LLaVA/checkpoints/$CKPT \
# #     --model-base /your_data_path//models/llava_v1_5_7b \
# #     --question-file /your_data_path//data/scienceqa/llava_test_CQM-A.json \
# #     --image-folder /your_data_path//data/scienceqa/images/test \
# #     --answers-file /your_data_path//data/scienceqa/answers/$CKPTLOAD.jsonl \
# #     --single-pred-prompt \
# #     --temperature 0 \
# #     --conv-mode vicuna_v1

# # --masked-param-path /your_data_path//code/LLaVA/checkpoints/$CKPT/$MASK.pth \

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_science \
#     --model-path /your_data_path//models/llava_v1_5_7b \
#     --question-file /your_data_path//data/scienceqa/llava_test_CQM-A.json \
#     --masked-param-path /your_data_path//code/LLaVA/checkpoints/$CKPT/$MASK.pth \
#     --image-folder /your_data_path//data/scienceqa/images/test \
#     --answers-file /your_data_path//data/scienceqa/answers/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl \
#     --num-chunks $CHUNKS \
#     --chunk-idx $IDX \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1 &
# done

# wait

# output_file=/your_data_path//data/textvqa/answers/$CKPTLOAD/merge.jsonl

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat /your_data_path//data/sqa/answers/$CKPTLOAD/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# python llava/eval/eval_science_qa.py \
#     --base-dir /your_data_path//data/scienceqa \
#     --result-file $output_file \
#     --output-file /your_data_path//data/scienceqa/answers/$CKPTLOAD_output.jsonl \
#     --output-result /your_data_path//data/scienceqa/answers/$CKPTLOAD_result.json

