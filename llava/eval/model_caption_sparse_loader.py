import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

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
import torch.nn as nn

from llava.sparsegpt import SparseGPT

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
        annotation_file = "/your_data_path/data/flickr30k/annotations/test_coco_format.json"

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
    device="cuda:1"
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    premodel_path = os.path.expanduser(args.premodel_path)
    premodel_name = get_model_name_from_path(premodel_path)
    _, pretrained_model, _,_ = load_pretrained_model(premodel_path, args.premodel_base, premodel_name)

    model.eval()
    pretrained_model.eval()

    for param in model.parameters():
        param.requires_grad = False

    for param in pretrained_model.parameters():
        param.requires_grad = False

    model.to("cpu")
    pretrained_model.to("cpu")

    # dare_model_params = torch.load(args.model_path + "/dare_model_params_sparsity10_v3.pth")
    # model.load_state_dict(dare_model_params, strict=False)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for i in range(20):
        model.model.layers[i].requires_grad_(False)

    # layers = model.Qformer.bert.encoder.layer
    layers = model.model.layers[20:]
    # layers.append(model.llm_proj)
    dtype = next(iter(model.model.layers[20].parameters())).dtype
    # seqlen = model.Qformer.bert.config.max_position_embeddings
    # inps = torch.zeros(
    #     (nsamples, 37, model.Qformer.bert.config.hidden_size), dtype=dtype, device=device
    # )
    # inps = torch.zeros(
    #     (nsamples, 41, model.Qformer.bert.config.hidden_size), dtype=dtype, device=device
    # )
    inps = {}
    attention_mask = {}
    pretrain_layers = pretrained_model.model.layers[20:]

    encoder_inps = torch.zeros(
            (128, 257, 1408), dtype=dtype, device=device
        )
    cache = {'i': 0, 'attention_mask': None, 'encoder_attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, *args, **kwargs):
            inp = args[0]
            print("begin...")
            print(inp.shape)
            inps[cache['i']] = inp.squeeze()
            # encoder_inps[cache['i']] = args[3]
            # attention_mask[cache['i']] = args[1]
            # print(args[1].shape)
            # print(args[3].shape)
            # print(args[4].shape)
            cache['i'] += 1
            # cache['attention_mask'] = args[1]
            # cache['encoder_attention_mask'] = args[4]
            raise ValueError


    # model = model.to('cpu')
    model.model.layers[20] = Catcher(model.model.layers[20])
    model = model.to(device)

    
    results = []
    first_batch = 0
    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        try:
            idx = line["question_id"]
            cur_prompt = line["text"]

            first_batch += 1 
            # #restrict the number of loops
            if first_batch > 128: 
                break

            input_ids = input_ids.to(device=device, non_blocking=True)

            model(input_ids, images=image_tensor.to(dtype=torch.float16, device=device, non_blocking=True))
        except ValueError:
                pass
    
    outs = {}
    for i in inps:
        outs[i] = torch.zeros_like(inps[i])

    print('Ready.')

    def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    tops_trainable_name_mask = {}
    for subname, submask in zip(trainable_name, basepatch):
        tops_trainable_name_mask[subname] = submask
    
    delta_ft = {}
    delta_sparsepre = {} #sparsegpt后的参数-pretrain参数
    delta_ftpre= {} #finetune后的参数-pretrain参数
    total_mask = {}

    for i in range(len(layers)):
        layer = layers[i].to(device)
        pretrain_layer = pretrain_layers[i].to(device)

        subset = find_layers(layer)
        pretrain_subset = find_layers(pretrain_layer)
        
        gpts = {}
        for name in subset:
            # if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
            #     continue
            gpts[name] = SparseGPT(subset[name],pretrain_subset[name])
            # if args.wbits < 16:
            #     gpts[name].quantizer = Quantizer()
            #     gpts[name].quantizer.configure(
            #         args.wbits, perchannel=True, sym=False, mse=False
            #     )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            if len(name)!=0:
                outs[j] = layer(inps[j].unsqueeze(0))[0]
            else:
                # outs[j] = layer(inps[j].unsqueeze(0))[0]
                outs_final = layer(inps[j].unsqueeze(0))[0]

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            delta, delta2, delta3, param_total_mask = gpts[name].fasterprune(
                sparsity_level, prunen=prunen, prunem=prunem, percdamp=percdamp, blocksize=blocksize
            )

            layer_name = "Qformer.bert.encoder.layer." + str(i)
            complete_name = layer_name + "." + name + ".weight"
            
            # delta = gpts[name].fasterprune(
            #     sparsity_level, prunen=prunen, prunem=prunem, percdamp=percdamp, blocksize=blocksize, tops_mask=(1-tops_trainable_name_mask[complete_name]).bool()
            # )

            delta_ft[complete_name] = delta
            delta_ftpre[complete_name] = delta2
            delta_sparsepre[complete_name] = delta3
            total_mask[complete_name] = param_total_mask

            gpts[name].free()

        for j in range(nsamples):
            if len(name) != 0:
                outs[j] = layer(inps[j].unsqueeze(0))[0]
                outs[j] = outs[j].squeeze()
            else:
                # outs[j] = layer(inps[j].unsqueeze(0))[0]
                outs_final = layer(inps[j].unsqueeze(0))[0]


        # layers[i] = layer.cpu()
        layers[i] = layer
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    import pickle
    # with open('./mylogs/delta_weightmask_plus_sparsegpt10_onlymask.pkl', 'wb') as f:
    #     pickle.dump([delta_sparsepre, delta_ftpre, total_mask], f)
    with open(mask_dir, 'wb') as f:
        pickle.dump([delta_sparsepre, delta_ftpre, total_mask], f)

    return delta_ft
    





def eval_results(args):
    answers_file = os.path.expanduser(args.answers_file)
    # eval_result_file = "/your_data_path/ft_local/LAVIS-main/lavis/output/BLIP2/Caption_coco/202401101648/result/test_epochbest.json"
    eval_result_file = answers_file
    metrics = report_metrics(eval_result_file=eval_result_file, split_name="test", dataset_name = "flickr30k")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-lora-flickr30k2")
    # parser.add_argument("--model-base", type=str, default="/your_data_path/ft_local/models/llava-v1.5-7b")
    parser.add_argument("--model-path", type=str, default="/your_data_path/ft_local/LLaVA-main/checkpoints/llava-v1.5-7b-tune12layers-flickr30k")
    # parser.add_argument("--model-path", type=str, default="/your_data_path/ft_local/models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--premodel-path", type=str, default="/your_data_path/ft_local/models/llava-v1.5-7b")
    parser.add_argument("--premodel-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/your_data_path/data/flickr30k/images/flickr30k-images")
    parser.add_argument("--question-file", type=str, default="/your_data_path/data/flickr30k/questions.jsonl")
    parser.add_argument("--answers-file", type=str, default="/your_data_path/data/flickr30k/answers/llava-v1.5-7b-v2.json")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
    # eval_results(args)
