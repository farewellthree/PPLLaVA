
import logging
from tqdm import tqdm
import os
from ppllava.common.dist_utils import get_rank, get_world_size, init_distributed_mode
import argparse
import cv2

import torch

import einops
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import transformers

from ppllava.common.config import Config
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
    parser.add_argument("", action='store_true')
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

    dataset = VideoMME_dataset('Video-MME/videomme/test-00000-of-00001.parquet', 
                               num_segments=args.num_frames, use_subtitles = args.use_subtitles)
    dataset.set_rank_and_world_size(rank, world_size)

    if args.llava:
        processor = LLaVA_Processer(model_config)
    else:
        processor = STLLM_Processer(model_config)

    return model, processor, dataset

def visualize_attention(video, output, save_dir='./visual'):
    """
    Args:
    - video (list of PIL.Image): 视频的每一帧存储为 PIL.Image 对象的列表。
    - output (torch.Tensor): 大小为 (1, T, w, h) 的 attention score。
    - save_dir (str): 保存可视化结果的目录。
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 从 video 的第一帧中获取宽度和高度
    W, H = video[0].size
    
    # 取出 attention 的 tensor 并去掉 batch 维度
    attention = output[0]  # shape: (T, w, h)
    T, w, h = attention.shape
    
    # 对 attention score 进行插值，从 (w, h) 扩展到 (W, H)
    attention_resized = F.interpolate(attention.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
    
    # 遍历每一帧视频
    for t in range(T):
        # 将 PIL.Image 转换为 NumPy 数组
        frame = np.array(video[t])
        
        # 获取第 t 帧的 attention 并归一化
        attention_map = attention_resized[t].cpu().numpy()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())  # 归一化到 [0, 1]
        
        # 将 attention map 映射为颜色热图
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        
        # 如果视频帧是灰度图像（即 shape 为 (H, W)），则先将其转换为 RGB 格式
        if len(frame.shape) == 2:  # 灰度图像
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        
        # 将 heatmap 与原始视频帧叠加
        combined_frame = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        
        # 保存当前帧
        save_path = os.path.join(save_dir, f'frame_{t:04d}.png')
        cv2.imwrite(save_path, cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))  # OpenCV 使用 BGR 格式，因此需要转换
    
    print(f"Visualization saved in {save_dir}")

def visual_videomme_llava(
        q_i, model, processor, data_sample,
        question, system="", 
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        all_token=False,
        ask_simple=False,
    ):
    video = data_sample["video"]
    role = ("USER", "ASSISTANT")

    #prompt =  "USER:"+ " <image>\n" + system + "\n" + question + "ASSISTANT: " + answer_prompt 
    if data_sample['subtitles']:
        prompt = ("USER:"+ " <image>\n" + "This video's subtitles are listed below: " + data_sample['subtitles'] + ' \n' + 
            system + "\n" + question + " ASSISTANT: " + answer_prompt)
    else:
        prompt =  "USER:"+ " <image>\n" + system + "\n" + question + " ASSISTANT: " + answer_prompt


    inputs = processor(prompt, question, video)
    inputs = inputs.to(model.device)
    output = model(**inputs, output_visual_attention=True)
    
    B,T,W,H = output.size()

    output = einops.rearrange(output, 'B T W H -> B T (W H)')
    output = torch.softmax(output*0.2, dim=-1) 
    output = einops.rearrange(output, 'B T (W H) -> B T W H', W=W)


    visualize_attention(video, output, save_dir=f'./visual/{data_sample["video_id"]}/{q_i}')
    with open(f'./visual/{data_sample["video_id"]}/{q_i}/question.txt','w') as f:
        f.write(question)
    return output

def run(rank, args, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, dataset = load_model_and_dataset(rank,
                                                       world_size,
                                                       args)

    if rank == 0:
        tbar = tqdm(total=len(dataset))

    result_list = []
    done_count = 0
    
    for i, example in enumerate(dataset):
        pred = [visual_videomme_llava(
            q_i, model, processor, example, question,
            system="Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.",
            answer_prompt="The best answer is:",
            return_prompt='',
            all_token=args.all_token,
            ask_simple=args.ask_simple,
        ) for q_i, question in enumerate(example['questions'])]

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
        pass
if __name__ == "__main__":
    main()