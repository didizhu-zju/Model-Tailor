#!/bin/bash

# SPLIT="mmbench_dev_cn_20231003"

# python -m llava.eval.model_vqa_mmbench \
#     --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-lora-coco \
#     --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
#     --question-file /your_data_path/data/mmbench/$SPLIT.tsv \
#     --answers-file /your_data_path/data/mmbench_cn/answers/$SPLIT/llava-v1.5-7b-lora-coco.jsonl \
#     --lang cn \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p /your_data_path/data/mmbench_cn/answers_upload/$SPLIT

# python scripts/convert_mmbench_for_submission.py \
#     --annotation-file /your_data_path/data/mmbench_cn/$SPLIT.tsv \
#     --result-dir /your_data_path/data/mmbench_cn/answers/$SPLIT \
#     --upload-dir /your_data_path/data/mmbench_cn/answers_upload/$SPLIT \
#     --experiment llava-v1.5-7b-lora-coco



#!/bin/bash

SPLIT="mmbench_dev_cn_20231003"

MASK="grafted_model_params_sparsity5"
# CKPT="llava-v1.5-7b-lora-flickr30k2"
LOAD="-tops-5"

# MASK="grafted_model_params_sparsity10_v1"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v4-1e-4"
# CKPT="llava-v1.5-7b-tune12layers-flickr30k"
# LOAD="-tops-10"
CKPT="llava-v1.5-7b-lora-okvqa-1e-4"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v3-2e-5"
# LOAD=""
# LOAD="-tops-10-v1"
CKPTLOAD="${CKPT}${LOAD}"

python -m llava.eval.model_vqa_mmbench \
    --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT \
    --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
    --masked-param-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT/$MASK.pth \
    --question-file /your_data_path/data/mmbench/$SPLIT.tsv \
    --answers-file /your_data_path/data/mmbench_cn/answers/$SPLIT/$CKPTLOAD.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

# python -m llava.eval.model_vqa_mmbench \
#     --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT \
#     --question-file /your_data_path/data/mmbench/$SPLIT.tsv \
#     --answers-file /your_data_path/data/mmbench_cn/answers/$SPLIT/$CKPTLOAD.jsonl \
#     --lang cn \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

mkdir -p /your_data_path/data/mmbench_cn/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file /your_data_path/data/mmbench_cn/$SPLIT.tsv \
    --result-dir /your_data_path/data/mmbench_cn/answers/$SPLIT \
    --upload-dir /your_data_path/data/mmbench_cn/answers_upload/$SPLIT \
    --experiment $CKPTLOAD
