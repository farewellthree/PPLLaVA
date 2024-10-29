
import logging
from tqdm import tqdm
import os
from ppllava.common.dist_utils import get_rank, get_world_size, init_distributed_mode
import argparse
import numpy as np

import torch

import torch.distributed as dist
import transformers

from ppllava.common.config import Config
from ppllava.test.video_utils import LLaVA_Processer, STLLM_Processer
from ppllava.conversation.conv import conv_templates
from ppllava.test.mvbench.utils import MVBench_dataset, infer_mvbench_stllm, \
      infer_mvbench_llava, check_ans, save_results
from ppllava.common.registry import registry

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt-path", help="path to checkpoint file.", default="")
    parser.add_argument("--num-frames", type=int, required=False, default=100)
    parser.add_argument("--specified_item", type=str, required=False, default=None)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--all_token", action='store_true')
    parser.add_argument("--system_llm", action='store_false')
    parser.add_argument("--ask_simple", action='store_true')
    return parser.parse_args()

def load_model_and_dataset(rank, world_size, args):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    #model_config.ckpt = args.ckpt_path
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    model.load_pretrained_weight(args.ckpt_path)
    model = model.to('cuda:{}'.format(rank))
    for name, para in model.named_parameters():
        para.requires_grad = False
    model.eval()

    dataset = MVBench_dataset(num_segments=args.num_frames, specified_item = args.specified_item)
    dataset.set_rank_and_world_size(rank, world_size)

    if 'qwen' in model_config.llama_model:
        phrase = 'llava_qwen'
        conv = conv_templates['plain_qwen']
    elif 'llava' in model_config.llama_model:
        phrase = 'llava_vicuna'
        conv = conv_templates['plain_v1']
    else:
        phrase = 'stllm'
        conv = None

    if 'llava' in phrase:
        processor = LLaVA_Processer(model_config)
    else:
        processor = STLLM_Processer(model_config)

    return model, processor, dataset, conv, phrase

def run(rank, args, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, dataset, conv, phrase = load_model_and_dataset(rank,
                                                       world_size,
                                                       args)

    if rank == 0:
        tbar = tqdm(total=len(dataset))

    correct = 0
    total = 0
    result_list = []
    acc_dict = {}
    done_count = 0

    infer_mvbench = infer_mvbench_llava if 'llava' in phrase else infer_mvbench_stllm
    for example in dataset:
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        
        pred = infer_mvbench(
            model, processor, example, conv,
            system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
            question_prompt="\nOnly give the best option.",
            answer_prompt="Best option:(",
            return_prompt='(',
            system_llm=args.system_llm,
            all_token=args.all_token,
            ask_simple=args.ask_simple,
        )
        gt = example['answer']
        result_list.append({
            'pred': pred,
            'gt': gt,
            'task_type': task_type,
            'video_path': example['video_path'],
            'question': example['question'],

        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1
        if rank == 0:
            tbar.update(len(result_list) - done_count, )
            tbar.set_description_str(
                f"One Chunk--Task Type: {task_type}, Chunk Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%;" 
                f" Chunk Total Acc: {correct / total * 100 :.2f}%"
            )
            done_count = len(result_list)
    return result_list

def main():
    args = parse_args()

    args.distributed = True
    args.dist_url = "env://"
    init_distributed_mode(args)
    rank, world_size = get_rank(), get_world_size()
    if not os.path.exists(args.output_dir) and rank==0:
        os.makedirs(args.output_dir)

    local_result = run(rank, args, world_size)
    gather_list = [None for _ in range(world_size)]
    # Gather results at all ranks
    dist.barrier()
    dist.all_gather_object(gather_list, local_result)
    result_list = []
    for res in gather_list:
        result_list.extend(res)
    if rank == 0:
        save_results(result_list, args.output_dir, args.output_name)
    
if __name__ == "__main__":
    main()