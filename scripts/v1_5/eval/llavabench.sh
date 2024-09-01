#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path /your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-lora-flickr30k2 \
    --model-base /your_data_path/ft_local/models/llava-v1.5-7b \
    --question-file /your_data_path/data/llava-bench-in-the-wild/questions.jsonl \
    --image-folder /your_data_path/data/llava-bench-in-the-wild/images \
    --answers-file /your_data_path/data/llava-bench-in-the-wild/answers/llava-v1.5-7b-lora-flickr30k2.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /your_data_path/data/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question /your_data_path/data/llava-bench-in-the-wild/questions.jsonl \
    --context /your_data_path/data/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        /your_data_path/data/llava-bench-in-the-wild/answers_gpt4.jsonl \
       /your_data_path/data/llava-bench-in-the-wild/answers/llava-v1.5-7b-lora-flickr30k2.jsonl \
    --output \
        /your_data_path/data/llava-bench-in-the-wild/reviews/llava-v1.5-7b-lora-flickr30k2.jsonl

python llava/eval/summarize_gpt_review.py -f /your_data_path/data/llava-bench-in-the-wild/reviews/llava-v1.5-7b-lora-flickr30k2.jsonl
