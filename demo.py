import argparse
import torch

from ppllava.common.config import Config
from ppllava.common.registry import registry
from ppllava.conversation.conv import conv_templates
from ppllava.models import *
from ppllava.runners import *
from ppllava.tasks import *

from ppllava.test.video_utils import LLaVA_Processer

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='config/instructblipbase_stllm_conversation.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--ckpt-path", required=True, help="path to STLLM_conversation_weight.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
llama_model = model_config.llama_model 
model_config.llama_model = args.ckpt_path
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_pretrained(args.ckpt_path, torch_dtype=torch.float16)
model = model.to('cuda:{}'.format(args.gpu_id))

processor = LLaVA_Processer(model_config)
processor.processor.image_processor.set_reader(video_reader_type='decord', num_frames=32)
if 'qwen' in llama_model:
    conv = conv_templates['plain_qwen']
elif 'llava' in llama_model:
    conv = conv_templates['plain_v1']

video = 'example/wukong_yugao.mp4'
prompt = "Describe this video in detail."

local_conv = conv.copy()
local_conv.user_query(prompt, is_mm=True)
full_question = local_conv.get_prompt()

inputs = processor(full_question, prompt, video)
inputs = inputs.to(model.device)

if conv.sep[0]=='<|im_end|>\n': #qwen
    split_str = 'assistant\n'
    target_dtype = model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
    inputs['pixel_values'] = inputs.pixel_values.to(f'cuda:{target_dtype}' if isinstance(target_dtype, int) else target_dtype)
else:
    split_str = conv.roles[1]
output = model.generate(**inputs, max_new_tokens=200)
llm_message = processor.processor.decode(output[0], skip_special_tokens=True)
llm_message = llm_message.split(split_str)[1].strip()

print (llm_message)
    