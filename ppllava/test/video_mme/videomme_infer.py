
import logging
from tqdm import tqdm
import os
from ppllava.common.dist_utils import get_rank, get_world_size, init_distributed_mode
import argparse
import json

import torch

import torch.distributed as dist
import transformers

from ppllava.common.config import Config
from ppllava.conversation.conv import conv_templates
from ppllava.test.video_utils import LLaVA_Processer, STLLM_Processer
from ppllava.test.video_mme.utils import VideoMME_dataset, infer_videomme_llava, \
    infer_videomme_stllm
from ppllava.test.video_mme.eval_result import eval_your_results
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
    parser.add_argument("--use_subtitles", action='store_true')
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

    dataset = VideoMME_dataset(num_segments=args.num_frames, use_subtitles = args.use_subtitles)
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

    result_list = []

    infer_videomme = infer_videomme_llava if 'llava' in phrase else infer_videomme_stllm
    done_count = 0
    
    for i, example in enumerate(dataset):
        pred = [infer_videomme(
            model, processor, example, question, conv,
            system="Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.",
            answer_prompt="The best answer is:",
            return_prompt='',
            all_token=args.all_token,
            ask_simple=args.ask_simple,
        ) for question in example['questions']]

        answer_dicts = [{
            'question_id': question_meta['question_id'],
            'task_type': question_meta['task_type'],
            'question': question_meta['question'],
            'options': list(question_meta['options']),
            'answer': question_meta['answer'],
            'response': pred[i],
        } for i, question_meta in enumerate(example['question_metas'])]
        video_id = example['video_id']

        result_list.append(
            {
                "video_id": example['video_id'],
                "duration": example['duration'],
                "domain": example['domain'],
                "sub_category": example['sub_category'],
                "questions": answer_dicts
            }
        )
   
        if rank == 0:
            tbar.update(1, )
            tbar.set_description_str(
                f"gt: {example['question_metas'][0]['answer']}......--pred: {pred[0][:min(15, len(pred[0]))]}......"
            )

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
 
    result_list = {}
    for local_results in gather_list:
        for res in local_results:
            if res['video_id'] not in result_list:
                result_list[res['video_id']] = res
            else:
                result_list[res['video_id']]['questions'].extend(res['questions'])
    result_list = list(result_list.values())

    if rank == 0:
        eval_your_results(result_list, video_types=["short","medium","long"], return_categories_accuracy=False,
                           return_sub_categories_accuracy=False, return_task_types_accuracy=False)
        with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as f:
            json.dump(result_list, f)
if __name__ == "__main__":
    main()