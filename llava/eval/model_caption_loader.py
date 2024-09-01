import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

# import sys
# desired_path ='/your_data_path//code/LLaVA/peft'
# sys.path.insert(0, desired_path)
# import llava.peft

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader


from PIL import Image
import math

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def coco_caption_eval(results_file, split, dataset_name):

    print("dataset_name:", dataset_name)

    if dataset_name == "coco_caption":
        filenames = {
            "val": "coco_karpathy_val_gt.json",
            "test": "coco_karpathy_test_gt.json",
        }
        annotation_file = os.path.join("/your_data_path/data/coco_gt/", filenames[split])

    elif dataset_name == "nocaps":
        # nocaps dataset
        annotation_file = '/your_data_path/data/nocaps/annotations/nocaps_val_4500_captions.json'

    elif dataset_name == "flickr30k":
        # flickr30k dataset
        annotation_file = "/your_data_path//data/flickr30k/annotations/test_coco_format.json"

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    if dataset_name == "nocaps":
        # Evaluate for each domain for nocaps
        domains = ['in-domain', 'near-domain', 'out-domain']
        for domain in domains:
            # Filter IDs by domain
            domain_img_ids = [img_id for img_id, info in coco.imgs.items() if info['domain'] == domain]
            
            # Create a COCOeval object by filtering for image IDs in the specific domain
            coco_eval = COCOEvalCap(coco, coco_result)
            coco_eval.params['image_id'] = domain_img_ids  # Set the image IDs for this domain
            
            # Evaluate on the filtered image IDs
            coco_eval.evaluate()

            # print output evaluation scores
            for metric, score in coco_eval.eval.items():
                print(f"{metric}: {score:.3f}")
            print(f'--- Evaluation scores for {domain} ---')

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def report_metrics(eval_result_file, split_name, dataset_name):

    # TODO better way to define this
    coco_val = coco_caption_eval(eval_result_file, split_name, dataset_name)

    agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
    log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

    # with open(
    #     os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
    # ) as f:
    #     f.write(json.dumps(log_stats) + "\n")

    # coco_res = {k: v for k, v in coco_val.eval.items()}
    # coco_res["agg_metrics"] = agg_metrics

    # return coco_res


def eval_model(args):
    # Model
    disable_torch_init()
    # device='cuda:1'
    device = torch.device("cuda:1")
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if "lora" in model_path:
        tokenizer, model, image_processor, context_len, _ = load_pretrained_model(model_path, args.model_base, model_name)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    

    # dare_model_params = torch.load("/your_data_path//code/LLaVA/checkpoints/llava-v1.5-7b-lora-flickr30k/sparse_lora_params_sparsity20_pretrain_projector.pth")
    # # dare_model_params = torch.load(args.model_path + "/grafted_model_params_sparsity10_v1.pth")
    # model.load_state_dict(dare_model_params, strict=False)
    if args.masked_param_path:
        print("1")
        masked_param_dict = torch.load(args.masked_param_path,  map_location=lambda storage, loc: storage.cuda())
        
        # print("2")
        # # param_grad_dic = {k: v.requires_grad for (k, v) in model.named_parameters()}
        # pretrained_model = model.state_dict()
        # for k in list(pretrained_model.keys()):
        #     if "vision_tower" in k: #lora
        #         del pretrained_model[k]
        # print("3")
        # # dare_model_params = {}
        # for key in pretrained_model:
        #     if key in masked_param_dict:
        #         masked_param_dict[key] = pretrained_model[key].to(device) + masked_param_dict[key]
        #     else:
        #         # print("Current key: " + key + "is not in masked_param_dict")
        #         masked_param_dict[key] = pretrained_model[key].to(device)
        # print("4")        
        # model.load_state_dict(masked_param_dict, strict=False)
        print("3")
        with torch.no_grad():  # 确保不会对梯度进行计算
            for name, param in model.named_parameters():
                if name in masked_param_dict:
                    # 将 CPU 上的参数转移到 GPU 上，并与模型的现有参数相加
                    temp_tensor = masked_param_dict[name].to(param.device)
                    param.add_(temp_tensor)
                    del temp_tensor  # 释放 GPU 内存

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)


    
    results = []
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda:1', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda:1', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        results.append({
        "caption": outputs,
        "image_id": int(idx)  # Generates a random UUID and converts to int
        })

    # Write the list of dictionaries as a JSON array to the file
    with open(answers_file, 'w') as ans_file:
        json.dump(results, ans_file, indent=4)
    ans_file.close()



def eval_results(args):
    answers_file = os.path.expanduser(args.answers_file)
    eval_result_file = answers_file
    metrics = report_metrics(eval_result_file=eval_result_file, split_name="test", dataset_name = "flickr30k")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/your_data_path//code/LLaVA/checkpoints/llava-v1.5-7b-lora-textvqa-instruction")
    # parser.add_argument("--model-base", type=str, default="/your_data_path//models/llava_v1_5_7b")
    # parser.add_argument("--model-path", type=str, default="/your_data_path//code/LLaVA/checkpoints/llava-v1.5-7b-tune12layers-okvqa-v4-1e-4")
    parser.add_argument("--model-path", type=str, default="/your_data_path//models/llava_v1_5_7b")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--masked-param-path", type=str, default="/your_data_path//code/LLaVA/checkpoints/llava-v1.5-7b-lora-textvqa-instruction/sparse_lora_params_sparsity20.pth")
    parser.add_argument("--masked-param-path", type=str, default="/your_data_path//code/LLaVA/checkpoints/fused_lora/filckr-sqa-sparse_lora_fusion_sparsity50.pth")
    parser.add_argument("--image-folder", type=str, default="/your_data_path//data/flickr30k/images/flickr30k-images")
    parser.add_argument("--question-file", type=str, default="/your_data_path//data/flickr30k/llava1.5_flickr30k_question5.jsonl")
    # parser.add_argument("--answers-file", type=str, default="/your_data_path//data/flickr30k/answers/llava-v1.5-7b-fused-flickr-sqa-sparse-lora-20.json")
    parser.add_argument("--answers-file", type=str, default="/your_data_path//data/flickr30k/answers/llava-v1.5-7b-avg-filckr-sqa-sparse_lora_fusion_sparsity50.json")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
    eval_results(args)

