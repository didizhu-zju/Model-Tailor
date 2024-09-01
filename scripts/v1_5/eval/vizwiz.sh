#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-lora-coco \
#     --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
#     --question-file /your_data_path/data/vizwiz/llava_test.jsonl \
#     --image-folder /your_data_path/data/vizwiz/test \
#     --answers-file /your_data_path/data/vizwiz/answers/llava-v1.5-7b-lora-coco.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python scripts/convert_vizwiz_for_submission.py \
#     --annotation-file /your_data_path/data/vizwiz/llava_test.jsonl \
#     --result-file /your_data_path/data/vizwiz/answers/llava-v1.5-7b-lora-coco.jsonl \
#     --result-upload-file /your_data_path/data/vizwiz/answers_upload/llava-v1.5-7b-lora-coco.json


MASK="grafted_model_params_sparsity5"
# CKPT="llava-v1.5-7b-lora-flickr30k2"
LOAD="-tops-5"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v4-1e-4"
# MASK="grafted_model_params_sparsity10_v1"
# CKPT="llava-v1.5-7b-tune12layers-flickr30k"
# LOAD="-tops-10"
# LOAD="-tops-10-v1"
CKPT="llava-v1.5-7b-lora-okvqa-1e-4"
# CKPT="llava-v1.5-7b-tune12layers-okvqa-v3-2e-5"
# LOAD=""
CKPTLOAD="${CKPT}${LOAD}"

python -m llava.eval.model_vqa_loader \
    --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT \
    --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
    --masked-param-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT/$MASK.pth \
    --question-file /your_data_path/data/vizwiz/llava_test.jsonl \
    --image-folder /your_data_path/data/vizwiz/test \
    --answers-file /your_data_path/data/vizwiz/answers/$CKPTLOAD.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# python -m llava.eval.model_vqa_loader \
#     --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/$CKPT \
#     --question-file /your_data_path/data/vizwiz/llava_test.jsonl \
#     --image-folder /your_data_path/data/vizwiz/test \
#     --answers-file /your_data_path/data/vizwiz/answers/$CKPTLOAD.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file /your_data_path/data/vizwiz/llava_test.jsonl \
    --result-file /your_data_path/data/vizwiz/answers/$CKPTLOAD.jsonl \
    --result-upload-file /your_data_path/data/vizwiz/answers_upload/$CKPTLOAD.json
