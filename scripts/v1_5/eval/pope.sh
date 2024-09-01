#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-lora-coco \
#     --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
#     --question-file /your_data_path/data/pope/llava_pope_test.jsonl \
#     --image-folder /your_data_path/data/coco/images/val2014 \
#     --answers-file /your_data_path/data/pope/answers/llava-v1.5-7b-lora-coco.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir /your_data_path/data/pope/coco \
#     --question-file /your_data_path/data/pope/llava_pope_test.jsonl \
#     --result-file /your_data_path/data/pope/answers/llava-v1.5-7b-lora-coco.jsonl


MASK="grafted_model_params_sparsity5"
# CKPT="llava-v1.5-7b-lora-flickr30k2"
LOAD="-tops-5"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v4-1e-4"
# MASK="grafted_model_params_sparsity10_v1"
# CKPT="llava-v1.5-7b-tune12layers-flickr30k"
# LOAD="-tops-10"
CKPT="llava-v1.5-7b-lora-okvqa-1e-4"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v3-2e-5"
# LOAD=""
# LOAD="-tops-10-v1"
CKPTLOAD="${CKPT}${LOAD}"

python -m llava.eval.model_vqa_loader \
    --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT \
    --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
    --masked-param-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT/$MASK.pth \
    --question-file /your_data_path/data/pope/llava_pope_test.jsonl \
    --image-folder /your_data_path/data/coco/images/val2014 \
    --answers-file /your_data_path/data/pope/answers/$CKPTLOAD.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# python -m llava.eval.model_vqa_loader \
#     --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT \
#     --question-file /your_data_path/data/pope/llava_pope_test.jsonl \
#     --image-folder /your_data_path/data/coco/images/val2014 \
#     --answers-file /your_data_path/data/pope/answers/$CKPTLOAD.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /your_data_path/data/pope/coco \
    --question-file /your_data_path/data/pope/llava_pope_test.jsonl \
    --result-file /your_data_path/data/pope/answers/$CKPTLOAD.jsonl