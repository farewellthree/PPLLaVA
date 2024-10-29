# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Any, Tuple

import sys
import argparse

import torch
import deepspeed

import transformers
from torch.utils.data.dataloader import default_collate
from ppllava.train.ppllava_trainer import PPLLaVATrainer

import ppllava.tasks as tasks
from ppllava.common.config import Config
from ppllava.common.logger import setup_logger
# imports modules for registration
from ppllava.datasets.builders import *
from ppllava.models import *
from ppllava.processors import *
from ppllava.runners import *
from ppllava.tasks import *

from transformers.data.data_collator import DataCollatorWithPadding, pad_without_fast_tokenizer_warning

local_rank = None
import os
os.environ["WANDB_DISABLED"]="true"


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--local_rank", required=False, default=0)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
    
@dataclass
class ModelArguments:
    freeze_backbone: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=1024,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    custom_lr: Optional[float] = None
    custom_names: str = None
    group_by_modality_length: bool = field(default=False)
    per_device_train_batch_sizes: str = None


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['multi_modal_projector', 'btadapter']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return
        
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    if trainer.args.should_save:
        model_no_ddp = trainer.model
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict     
        trainer._save(output_dir, state_dict=cpu_state_dict)

def merge_dict_to_argv(input_dict):
    i = 0
    while i < len(sys.argv):
        if sys.argv[i].startswith('--cfg-path'):
            sys.argv.pop(i)
            sys.argv.pop(i)
            break
        else:
            i += 1
    sys.argv.extend([f'--{key}={value}' for key, value in input_dict.items()])

@dataclass
class DefaultDataCollator(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return default_collate(instances)
    
@dataclass
class InterleaveDataCollator(DataCollatorWithPadding):
    def __init__(self, video_batch=False, **kwargs):
        super().__init__(**kwargs)
        self.video_batch = video_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.video_batch:
            pixel_values = [feat.pop('pixel_values') for feat in features]
            clip_ids = [feat.pop('clip_ids') for feat in features] if 'clip_ids' in features[0] else None
        media_types = [feat.pop('media_type') for feat in features]
        
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if not self.video_batch:
            batch['pixel_values'] = pixel_values
            batch['clip_ids'] = clip_ids
        batch['media_types'] = media_types
        
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch
    
def train():
    global local_rank

    cfg = Config(parse_args())
    model_cfg = cfg.model_cfg

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    task_name =  cfg.run_cfg.pop('task')
    merge_dict_to_argv(cfg.run_cfg)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    training_args.lora_enable = model_cfg.get('use_lora', False)
    training_args.tune_mm_mlp_adapter = (not model_cfg.get('use_lora', False)) and model_cfg.get('freeze_LLM', True)
    # set after init_distributed_mode() to only log on master.
    setup_logger()

    #cfg.pretty_print()

    data_module = {}
    
    data_module['train_dataset'] = datasets
    data_module['eval_dataset'] = None

    if task_name=='interleave_sft':
        tokenizer = datasets[list(datasets.keys())[0]]['train'].processor.tokenizer
        video_batch = (cfg.model_cfg.arch=='llava_vid_nogrid')
        data_module['data_collator'] = InterleaveDataCollator(video_batch=video_batch, tokenizer=tokenizer)
    elif 'llava_vid' in cfg.model_cfg.arch:
        data_module['tokenizer'] = datasets[list(datasets.keys())[0]]['train'].processor.tokenizer
    else:
        data_module['data_collator'] = DefaultDataCollator()

    trainer = PPLLaVATrainer(model=model,
                    args=training_args,
                    **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), bias="none"
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
