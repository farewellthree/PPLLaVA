import copy
import os
import json
from enum import auto, Enum
import dataclasses
from typing import Any, List

import torch
from .utils import EasyDict

IMAGE_TOKEN = "<image>"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()

def dump_json(obj_serializable ,save_dir_path, json_file_name):
    os.makedirs(save_dir_path, exist_ok=True)
    save_path = os.path.join(save_dir_path, json_file_name)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(obj_serializable, f, indent=4, ensure_ascii=False, )

def load_json(load_dir_path, json_file_name):
    
    load_path = os.path.join(load_dir_path, json_file_name)
    if not os.path.exists(load_path):
        return None
    with open(load_path, 'r', encoding='utf-8') as f:
        obj_serializable = json.load(f)
    return obj_serializable



@dataclasses.dataclass
class Conversation(EasyDict):
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    sep: List[str]
    mm_token: str
    
    pre_query_prompt: str=None
    post_query_prompt: str=None
    answer_prompt: str=None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.sep, str):
            self.sep = [self.sep for _ in self.roles]

    def get_prompt(self):
        sep = [self.sep for _ in self.roles] if isinstance(self.sep, str) else self.sep  # if only one sep given, then both sep are the sames
        sep = dict(zip(self.roles, sep))
        ret = self.system + sep[self.roles[0]] if self.system != "" else ""
        for i, (role, message) in enumerate(self.messages):
            # if is last msg(the prompt for assistant), if answer prompt exists, no sep added
            if i+1 == len(self.messages):
                if role != self.roles[-1]: # last role is not the model
                    ret += role + message + sep[role] + self.roles[-1]
                else:
                    ret += role + message
            else:
                ret += role + message + sep[role]
        return ret
    # def get_prompt_multichoice(self):
    #     pass
    def user_query(self, query=None, pre_query_prompt=None, post_query_prompt=None, is_mm=False, num_mm_token=1):
        if post_query_prompt is not None:
            query = f"{query} {post_query_prompt}"

        if pre_query_prompt is not None:
            query = f"{pre_query_prompt} {query}"
        role = self.roles[0]
        # TODO: remove the num_mm_token and hack the self.mm_token outside
        if is_mm:
            mm_str = num_mm_token*self.mm_token[:-1] + self.mm_token[-1]
            if self.mm_token not in query:
                if self.mm_token[:-1] in query:
                    query = query.replace(self.mm_token[:-1], self.mm_token)
                else:
                    query = f'{mm_str}{query}'
        self._append_message(role, query)
    
    def assistant_response(self, response, pre_query_prompt=None, post_query_prompt=None):
        if post_query_prompt is not None:
            response = f"{response} {post_query_prompt}"

        if pre_query_prompt is not None:
            response = f"{post_query_prompt} {response}"

        role = self.roles[1]
        self._append_message(role, response)
    
    def _append_message(self, role, message):
        message = '' if message is None else message
        self.messages.append([role, message])

    def copy(self):
        return copy.deepcopy(self)

conv_plain_v1 = Conversation(
    system="",
    roles=("USER: ", "ASSISTANT: "),
    messages=[],
    sep=(" ", "</s>"),
    mm_token='<image>\n'
)

conv_vcg_v1 = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    roles=("USER: ", "ASSISTANT: "),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
)

SYSTEM_MVBENCH="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
conv_mvbench_v1 = Conversation(
    system=SYSTEM_MVBENCH,
    roles=("USER: ", "ASSISTANT: "),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
)


conv_plain_qwen = Conversation(
    system="",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
)

conv_vcg_qwen = Conversation(
    system="You are Video-ChatGPT, a large vision-language assistant. "
           "You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language."
           "Follow the instructions carefully and explain your answers in detail based on the provided video.",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
)

conv_videoqa_v1 = Conversation(
    system="",
    roles=("USER: ", "ASSISTANT: "),
    messages=[],
    sep=[" ","</s>"],
    mm_token='<image>\n',
    pre_query_prompt="The input consists of a sequence of key frames from a video. Answer the question concisely first and followed by significant events, characters, or objects that appear throughout the frames. Question:",
    post_query_prompt="\n",
    answer_prompt='Answer: In the video,'
)

conv_videoqa_qwen = Conversation(
    system="",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    messages=[],
    sep=["<|im_end|>\n","<|im_end|>\n"],
    mm_token='<image>\n',
    pre_query_prompt="The input consists of a sequence of key frames from a video. Answer the question concisely first and followed by significant events, characters, or objects that appear throughout the frames. Question:",
    post_query_prompt="\n",
    answer_prompt='Answer: In the video,'
)


conv_templates = {
    "plain_v1": conv_plain_v1,
    "plain_qwen": conv_plain_qwen,
    "conv_vcg_v1": conv_vcg_v1,
    "conv_vcg_qwen": conv_vcg_qwen,
    "conv_mvbench_v1": conv_mvbench_v1,
    "conv_videoqa_v1": conv_videoqa_v1,
    "conv_videoqa_qwen": conv_videoqa_qwen,
}

