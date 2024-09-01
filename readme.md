
# Model Tailor

This repository contains code to reproduce the key results of the paper [Model Tailor: Mitigating Catastrophic Forgetting in
Multi-modal Large Language Models](https://arxiv.org/pdf/2402.12048) which was accepted by ICML 2024.


## Prerequisites

Before you begin, ensure you have completed the following steps:

1. Configure the LLaVA environment: Follow the instructions in the [LLaVA GitHub repository](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file).
2. Prepare relevant datasets: Make sure you have downloaded and properly set up the required datasets.

## Usage

### Step 0: Configure Paths

Before running any scripts, it's crucial to update the model and dataset paths in the relevant code files to match your local setup:

1. Open the script files located in `/llava-model-tailor/scripts/v1_5/`.
2. In `finetune_flickr30k.sh` and `finetune_okvqa.sh`, locate the paths for the model and dataset.
3. Replace these paths with the correct locations on your system.
4. Similarly, update the paths in `main_model_tailor.py` if necessary.

### Step 1: Fine-tune Models

First, we need to fine-tune models separately on the Flickr30k and OKVQA datasets to obtain two specialized models.

Run the following scripts:

```bash
sh /llava-model-tailor/scripts/v1_5/finetune_flickr30k.sh
sh /llava-model-tailor/scripts/v1_5/finetune_okvqa.sh
```

This will generate two fine-tuned models.

### Step 2: Model Concatenation

After obtaining the two fine-tuned models, we use the model tailoring tool to concatenate them into a comprehensive model.

Run the following command:

```bash
python /llava-model-tailor/main_model_tailor.py
```

This script will process the fine-tuned models and generate a new concatenated model.

### Step 3: Evaluation
To evaluate the performance of your tailored model, run the following script:

```bash
sh llava-model-tailor/scripts/v1_5/eval/eval_all.sh
```


### Adapting to Other Models

To use this method with InstructBLIP or other large language models:

Copy ``main_model_tailor.py`` and ``sparsegpt.py`` to your target model's project directory.
Adjust the import statements and model-specific code in these files as necessary to match your target model's structure.


