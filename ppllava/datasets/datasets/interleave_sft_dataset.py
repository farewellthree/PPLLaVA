import logging
import os
import torch

from torch.utils.data import Dataset

import torch.nn.functional as F
from ppllava.datasets.datasets.llavavid_processor import LlavaNextViDTextProcessor, LlavaOnevisionViDTextProcessor
from ppllava.conversation.conv import conv_templates

import json
import numpy as np

from transformers import LlavaNextProcessor, LlavaOnevisionProcessor, CLIPModel, CLIPProcessor, SiglipProcessor

from .utils import load_anno, pre_text, VIDEO_READER_FUNCS, load_image_from_path
try:
    from mmengine import fileio 
    has_client = True
except ImportError:
    has_client = False

logger = logging.getLogger(__name__)

class InterleaveBaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""

    media_type = "video"

    def __init__(self, ann_files):
        self.client = None
        if has_client:
            self.client = fileio
        self.all_anno = self.read_anno(ann_files)
        self.conv = None
        
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def read_anno(self, ann_files):
        all_anno = []
        for ann_file in ann_files:
            media_type = ann_file["media_type"]
            label_file = ann_file.pop("label_file")
            data_root = ann_file.pop("data_root")
            repeat = ann_file["repeat"] if "repeat" in ann_file else False
            current_anno = []
            with open(label_file, 'r') as f:
                annos = json.load(f)
                for anno in annos:
                    anno.update(ann_file)
                    if isinstance(anno[media_type], str):
                        anno[media_type] = os.path.join(data_root, anno[media_type])
                    elif isinstance(anno[media_type], list):
                        anno[media_type] = [os.path.join(data_root, img_path) for img_path in anno[media_type]]
                    current_anno.append(anno)
            if repeat:
                current_anno = current_anno * 2
            all_anno.extend(current_anno)
        self.num_examples = len(all_anno)
        #random.shuffle(all_anno)
        return all_anno
    
    def process_qa(self, qa, msg=""):
        conv = self.conv.copy()
        
        if "i" in qa[0].keys() and qa[0]["i"] != "":
            conv.system = qa[0]["i"] 

        conv.user_query(qa[0]["q"], is_mm=True)
        conv.assistant_response(qa[0]["a"])
        if len(qa) > 1:
            for sentence in qa[1:]:
                q = sentence["q"]
                a = sentence["a"]
                conv.user_query(q)
                conv.assistant_response(a)
        return conv.get_prompt() + self.conv.sep[1]
    
    def load_and_transform_media_data(self, index, data_path, media_type="image"):
        if media_type == "image":
            return self.load_and_transform_media_data_image(index, data_path)
        else:
            return self.load_and_transform_media_data_video(index, data_path)

    def load_and_transform_media_data_image(self, index, data_path):
        image = load_image_from_path(data_path, client=self.client)
        return image, index

    def load_and_transform_media_data_video(self, index, data_path, video_reader, num_frames=-1,
                                            return_fps=False, clip=None, max_num_frames=-1):
        
        frames, frame_indices, sec = video_reader(
            data_path, num_frames, sample='rand', 
            max_num_frames=max_num_frames, client=self.client, clip=clip
        )
        if return_fps:
            #sec = [str(round(f / fps, 1)) for f in frame_indices]
            return frames, index, sec
        else:
            return frames, index
        

class Interleave_sft_dataset(InterleaveBaseDataset):
    def __init__(self, ann_files, model_cfg, **kwargs):
        super().__init__(ann_files=ann_files, **kwargs)
        llama_model = model_cfg.get('llama_model')
        qwen = ('qwen' in llama_model)
        self.qwen = qwen
        processor_cls = LlavaOnevisionProcessor if qwen else LlavaNextProcessor
        vid_processor_cls = LlavaOnevisionViDTextProcessor if qwen else LlavaNextViDTextProcessor

        self.processor = processor_cls.from_pretrained(llama_model)
        self.vid_processor = vid_processor_cls.from_pretrained(llama_model)

        self.arch = model_cfg.get('arch')
        if qwen:
            self.conv = conv_templates["plain_qwen"]
        else:
            self.conv = conv_templates["plain_v1"]

        if model_cfg.get('clip_weight', None):
            clip_processor_cls = SiglipProcessor if qwen else CLIPProcessor
            clip_processor = clip_processor_cls.from_pretrained(model_cfg.get('clip_weight'))
            self.clip_tokenizer = clip_processor.tokenizer
            if qwen:
                self.clip_length = 196 if model_cfg.get('extend_clip', False) else 64
            else:
                self.clip_length = 248 if model_cfg.get('extend_clip', False) else 77
            del clip_processor

        self.video_readers = VIDEO_READER_FUNCS

    def __len__(self):
        return self.num_examples

    def transform(self, images, media_type):
        processor = self.processor if media_type=='image' else self.vid_processor
        return processor.image_processor(images, return_tensors='pt')
    
    def _get_text_len(self, tokenizer, text):
        return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.shape[1]

    def __getitem__(self, index):
        try:
            ann = self.all_anno[index]
            media_type = ann['media_type']
            data_path = ann[media_type]
            
            if media_type=="image":
                embed, index = self.load_and_transform_media_data_image(index, data_path)
                embed = embed.squeeze()
            elif media_type=="multi-image":
                video_reader = self.video_readers['rawframe']
                max_num_frames = num_frames = -1
                embed, index = self.load_and_transform_media_data_video(index, data_path, video_reader,
                                        num_frames=num_frames, max_num_frames=max_num_frames)
            else:
                video_reader_type = ann["video_reader_type"] if "video_reader_type" in ann else "decord"
                video_reader = self.video_readers[video_reader_type]
                max_num_frames = ann["total_frames"] if "total_frames" in ann else -1
                num_frames = ann["num_frames"] if "num_frames" in ann else -1
                embed, index = self.load_and_transform_media_data_video(index, data_path, video_reader,
                                        num_frames=num_frames, max_num_frames=max_num_frames)
            
            processor = self.processor if media_type=='image' else self.vid_processor
            embed = processor.image_processor(embed, return_tensors='pt')
            
            instruction = self.process_qa(ann["QA"])
            instruction_full = processor.tokenizer.bos_token + instruction if processor.tokenizer.bos_token is not None else instruction
            sep1 = self.conv.roles[0] 
            sep2 = self.conv.roles[1] 
            raw_text = instruction_full.split(sep2)
            for idx in range(0, len(raw_text)-1):
                raw_text[idx] = raw_text[idx] + sep2
        
            instruction_embed = [processor.tokenizer(i, return_tensors='pt', add_special_tokens=False) for i in raw_text]
            input_ids = torch.cat([i.input_ids for i in instruction_embed], dim=1)
            input_att = torch.cat([i.attention_mask for i in instruction_embed], dim=1)

            answer_targets = input_ids.clone()
            cur_len = self._get_text_len(processor.tokenizer, raw_text[0])
            answer_targets[:, :cur_len] = -100
            for text in raw_text[1:-1]: 
                total_len = self._get_text_len(processor.tokenizer, text)
                ans_len = self._get_text_len(processor.tokenizer, text.split(sep1)[0].rstrip(' '))
                answer_targets[:, (cur_len+ans_len):(cur_len+total_len)] = -100
                cur_len += total_len
            cur_len += self._get_text_len(processor.tokenizer, raw_text[-1].rstrip(' '))
            assert cur_len == answer_targets.shape[1], f"The final length ({cur_len}) is not equal to the original prompt ({answer_targets.shape[1]}): {instruction}"
            answer_targets = answer_targets.squeeze()
            targets = F.pad(answer_targets, (0, 3000 - len(answer_targets)), value=-100)

            if hasattr(self,'clip_tokenizer'):
                if self.qwen:
                    if self.arch=="llava_interleave":
                        clip_input = self.clip_tokenizer(instruction, return_tensors='pt', max_length=self.clip_length)
                    else:
                        clip_input = self.clip_tokenizer(instruction, return_tensors='pt', padding='max_length', truncation=True, max_length=self.clip_length)
                    clip_ids, clip_mask = clip_input.input_ids.squeeze(), None
                else:
                    clip_input = self.clip_tokenizer(instruction, return_tensors='pt', padding='max_length', truncation=True, max_length=self.clip_length)
                    clip_ids, clip_mask = clip_input.input_ids.squeeze(), clip_input.attention_mask.squeeze()
            else:
                clip_ids, clip_mask = None, None

            return_dict = {
                "input_ids": input_ids.squeeze(),
                "pixel_values": embed["pixel_values"],
                "image_sizes": embed["image_sizes"][0],
                "attention_mask": input_att.squeeze(),
                "labels": targets,
                "dict": False,
                "media_type": media_type
            }    
            if clip_ids is not None:
                return_dict.update({"clip_ids": clip_ids,})  
            if clip_mask is not None:
                return_dict.update({"clip_mask": clip_mask,}) 
            return return_dict 

        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {data_path}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)


