#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-tune12layers-flickr30k \
    --question-file /your_data_path/data/mm-vet/llava-mm-vet.jsonl \
    --image-folder /your_data_path/data/mm-vet/images \
    --answers-file /your_data_path/data/mm-vet/answers/llava-v1.5-7b-tune12layers-flickr30k.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /your_data_path/data/mm-vet/answers/llava-v1.5-7b-tune12layers-flickr30k.jsonl \
    --dst /your_data_path/data/mm-vet/results/llava-v1.5-7b-tune12layers-flickr30k.json

