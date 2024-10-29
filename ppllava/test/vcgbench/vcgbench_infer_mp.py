
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
from ppllava.conversation.conv import conv_templates
from ppllava.test.video_utils import LLaVA_Processer, STLLM_Processer
from ppllava.test.vcgbench.utils import VideoChatGPTBenchDataset, infer_vcgbench_stllm, \
    infer_vcgbench_llava, save_results
from ppllava.common.registry import registry


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt-path", help="path to checkpoint file.", default="")
    parser.add_argument("--num-frames", type=int, required=False, default=100)
    parser.add_argument("--system_vcg", action='store_true')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
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

    dataset = VideoChatGPTBenchDataset(num_segments=args.num_frames)
    dataset.set_rank_and_world_size(rank, world_size)

    if 'qwen' in model_config.llama_model:
        phrase = 'llava_qwen'
        conv = conv_templates['conv_vcg_qwen'] if args.system_vcg else conv_templates['plain_qwen']
    elif 'llava' in model_config.llama_model:
        phrase = 'llava_vicuna'
        conv = conv_templates['conv_vcg_v1'] if args.system_vcg else conv_templates['plain_v1']
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

    result_list = []
    done_count = 0
    infer_vcgbench = infer_vcgbench_llava if 'llava' in phrase else infer_vcgbench_stllm
    for example in dataset:
        task_type = example['task_type']
        gt = example['answer']

        if task_type == 'consistency_qa':
            assert 'question' in example and 'question1' in example, 'two questions'
            pred = infer_vcgbench(
                model, processor, example, conv
            )

            example['question'], example['question1'] = example['question1'], example['question']
            pred1 = infer_vcgbench(
                model, processor, example, conv
            )
            res = {
                'pred1': pred,
                'pred2': pred1,
                'video_name': example['video_name'],
                'A': gt,
                'task_type': task_type,
                'Q1': example['question'],
                'Q2': example['question1'],
            }
        else:
            pred = infer_vcgbench(
                model, processor, example, conv
            )

            res = {
                'pred': pred,
                'A': gt,
                'Q': example['question'],
                'task_type': task_type,
                'video_name': example['video_name'],
            }

        result_list.append(res)
        if rank == 0:
            tbar.update(len(result_list) - done_count, )
            tbar.set_description_str(
                f"One Chunk--Task Type: {task_type}-"
                f"gt: {gt[:min(15, len(gt))]}......--pred: {pred[:min(15, len(gt))]}......"
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
        save_results(result_list, args.output_dir)
    
if __name__ == "__main__":
    main()