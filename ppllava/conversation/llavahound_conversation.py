import argparse
import time
import numpy as np
from PIL import Image
import torch_npu

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any

from stllm.common.registry import registry
from stllm.test.video_utils import load_video
from stllm.datasets.datasets.llavavid_processor import LlavaNextViDTextProcessor

from transformers import LlavaNextProcessor, CLIPProcessor
from transformers.feature_extraction_utils import BatchFeature

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    instruction: bool
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            instruction=self.instruction,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


CONV_llava_Vicuna0 = Conversation(
    system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, give your answer that best addresses the question.\n",
    roles=("USER: ", "ASSISTANT: "),
    messages=[],
    instruction=False,
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_llava_Vicuna = Conversation(
    system="",
    roles=("USER: ", "ASSISTANT:"),
    messages=[],
    instruction=False,
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

class llava_Chat:
    def __init__(self, model, model_config, device='npu:0'):
        self.device = device
        self.model = model

        self.clip_length =  248 if model_config.get('long_clip', False) else 77
        llama_model = model_config.llama_model
        if model_config.arch=='llava_vid_nogrid':
            self.processor = LlavaNextViDTextProcessor.from_pretrained(llama_model)
        elif model_config.arch=='llava_vid':
            self.processor = LlavaNextProcessor.from_pretrained(llama_model)

        if model_config.get('clip_weight',None) is not None:
            clip_processor = CLIPProcessor.from_pretrained(model_config.clip_weight)
            self.clip_tokenizer = clip_processor.tokenizer
            del clip_processor

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and (conv.messages[-1][1][-6:] == '</Img>' or conv.messages[-1][1][-8:] == '</Video>' 
                    or conv.messages[-1][1][-8:] == '</Frame>'):  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=100, num_beams=1, min_length=1, top_p=0.9, system=True,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000, do_sample=True):
        question = conv.messages[0][1]
        question = question.split('</Video> ')[1]
        system = conv.system if system else ""
        full_question = system + conv.roles[0] + "<image>\n"+ question + " " + conv.roles[1]
        text_inputs = self.processor(full_question)

        inputs = BatchFeature(data={**text_inputs, **img_list[0]})
        if hasattr(self,'clip_tokenizer'):
            clip_input = self.clip_tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=self.clip_length)
            clip_ids, clip_mask = clip_input.input_ids, clip_input.attention_mask
            inputs.update(
                {
                    "clip_ids": clip_ids,
                    "clip_mask": clip_mask,
                }
            )
        inputs = inputs.to(f'npu:{self.device}' if isinstance(self.device, int) else self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_text = self.processor.decode(output[0], skip_special_tokens=True)
        output_text = output_text.split(conv.roles[1])[1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output[0].cpu().numpy()

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            raw_image = image
    
        img_list.append(self.processor.image_processor(raw_image,return_tensors='pt'))
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def upload_video(self, video, conv, img_list, num_frame=64, text=None):
        raw_frames = load_video(video, num_frm=num_frame) if isinstance(video,str) else video

        video_input = self.processor.image_processor(raw_frames,return_tensors='pt')
        video_input['pixel_values'] = video_input['pixel_values'].unsqueeze(0)
        video_input['image_sizes'] = video_input['image_sizes'][0].unsqueeze(0)
        img_list.append(video_input)
        sign='<Video><ImageHere></Video>'
        conv.append_message(conv.roles[0], sign)
        msg = "Received."
        return msg
    

