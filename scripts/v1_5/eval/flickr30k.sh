#!/bin/bash

python -m llava.eval.model_caption_loader \
    --model-path /your_data_path/ft_local/models/llava-v1.5-7b \
    --question-file /dockerdata/data/flickr30k/question.json \
    --image-folder /dockerdata/data/flickr30k/images/flickr30k-images \
    --answers-file /dockerdata/data/flickr30k/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /dockerdata/data/textvqa/TextVQA_0.5.1_val.json \
    --result-file /dockerdata/data/textvqa/answers/llava-v1.5-13b.jsonl
