#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-lora-flickr30k2 \
    --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
    --question-file /your_data_path/data/MME/llava_mme.jsonl \
    --image-folder /your_data_path/data/MME/MME_Benchmark_release_version \
    --answers-file /your_data_path/data/MME/answers/llava-v1.5-7b-lora-flickr30k2.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /your_data_path/data/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b-lora-flickr30k2

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b-lora-flickr30k2
