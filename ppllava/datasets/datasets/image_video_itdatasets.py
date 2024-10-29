import logging
import os
import random
from tqdm import tqdm
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from ppllava.datasets.datasets.instruction_data import available_corpus, train_transform
from ppllava.datasets.datasets.llavavid_processor import LlavaNextViDTextProcessor

import json
from os.path import basename
import numpy as np

from transformers import LlavaNextProcessor, CLIPModel, CLIPProcessor
from transformers.feature_extraction_utils import BatchFeature

from .utils import load_anno, pre_text, VIDEO_READER_FUNCS, load_image_from_path

try:
    from mmengine import fileio 
    has_client = True
except ImportError:
    has_client = False

logger = logging.getLogger(__name__)


class ImageVideoBaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""

    media_type = "video"

    def __init__(self):
        assert self.media_type in ["image", "video", "only_video"]
        self.data_root = None
        self.anno_list = (
            None  # list(dict), each dict contains {"image": str, # image or video path}
        )
        self.transform = None
        self.video_reader = None
        self.num_tries = None

        self.client = None
        if has_client:
            self.client = fileio

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_anno(self, index):
        """obtain the annotation for one media (video or image)

        Args:
            index (int): The media index.

        Returns: dict.
            - "image": the filename, video also use "image".
            - "caption": The caption for this file.

        """
        anno = self.anno_list[index]
        if self.data_root is not None:
            anno["image"] = os.path.join(self.data_root, anno["image"])
        return anno

    def load_and_transform_media_data(self, index, data_path, transform=True):
        if self.media_type == "image":
            return self.load_and_transform_media_data_image(index, data_path, transform=transform)
        else:
            return self.load_and_transform_media_data_video(index, data_path, transform=transform)

    def load_and_transform_media_data_image(self, index, data_path, transform=True):
        image = load_image_from_path(data_path, client=self.client)
        if transform:
            image = self.transform(image)
        return image, index

    def load_and_transform_media_data_video(self, index, data_path, return_fps=False, clip=None, transform=True):
        for _ in range(self.num_tries):
            try:
                max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
                frames, frame_indices, sec = self.video_reader(
                    data_path, self.num_frames, self.sample_type, 
                    max_num_frames=max_num_frames, client=self.client, clip=clip
                )
            except Exception as e:
                #logger.warning(
                #    f"Caught exception {e} when loading video {data_path}, "
                #    f"randomly sample a new video as replacement"
                #)
                index = random.randint(0, len(self) - 1)
                ann = self.get_anno(index)
                data_path = ann["image"]
                continue
            # shared aug for video frames
            if transform:
                frames = self.transform(frames)
            if return_fps:
                #sec = [str(round(f / fps, 1)) for f in frame_indices]
                return frames, index, sec
            else:
                return frames, index
        else:
            raise RuntimeError(
                f"Failed to fetch video after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted videos."
            )

class PTImgTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(self, ann_file, transform, model_config, pre_text=True, max_length=77, use_instruction=None):
        super().__init__()

        if len(ann_file) == 3 and ann_file[2] == "video":
            self.media_type = "video"  
        else:
            self.media_type = "image"
        self.label_file, self.data_root = ann_file[:2]
        self.use_instruction = use_instruction
        if use_instruction:
            self.instruction = json.load(open(use_instruction, 'r'))
            

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)

        self.transform = transform
        self.pre_text = pre_text
        self.max_length = max_length

        self.clip_processor = CLIPProcessor.from_pretrained(model_config['clip_model'])
        logger.info(f"Pre-process text: {pre_text}")

    def get_anno(self, index):
        filename = self.anno[index][self.media_type]
        caption = self.anno[index]["caption"]
        anno = {"image": os.path.join(self.data_root, filename), "caption": caption}
        return anno

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data(index, ann["image"], transform=False)
            caption = pre_text(ann["caption"], pre_text=self.pre_text)
            if self.use_instruction:
                caption = "USER: " + self.instruction[random.randint(0,len(self.instruction))] + "\n ASSISTANT: " + caption
            inputs = self.clip_processor(text=[caption], images=image, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
            return_dict = {
                'input_ids':inputs['input_ids'].squeeze(),
                'pixel_values':inputs['pixel_values'],
                'attention_mask':inputs['attention_mask'].squeeze(),
                'dict': False,
            }
            return return_dict
        except Exception as e:
            #logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)

class PTVidTrainDataset(PTImgTrainDataset):
    media_type = "video"

    def __init__(
        self,
        ann_file,
        transform,
        model_config,
        num_frames=4,
        video_reader_type="decord",
        sample_type="rand",
        num_tries=1000,
        pre_text=True,
        max_length=77,
        use_instruction=False,
    ):
        super().__init__(ann_file, transform, model_config=model_config, pre_text=pre_text, max_length=max_length, use_instruction=use_instruction)
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries

class ITImgTrainDataset(ImageVideoBaseDataset):
    media_type = "image"

    def __init__(
        self, ann_file, transform, simple=False,
        system="", role=("Human", "Assistant"),
        start_token="<Image>", end_token="</Image>",
        begin_signal="###", place_holder='<ImageHere>',
        random_shuffle=True, # if True, shuffle the QA list
    ):
        super().__init__()

        if len(ann_file) == 3 and ann_file[2] == "video":
            self.media_type = "video"  
        else:
            self.media_type = "image"
        self.label_file, self.data_root = ann_file[:2]

        logger.info('Load json file')
        with open(self.label_file, 'r') as f:
            self.anno = json.load(f)
        self.num_examples = len(self.anno)
        self.transform = transform
        self.place_holder = place_holder

        # prompt parameters
        if system:
            assert system[-1] == " ", "' ' should be add in the end of system, thus '###' will be tokenized into one token."
        # currently not support add start_token and end_token in the system, since the msg should be added properly
        self.begin_signal = begin_signal
        self.end_signal = " "
        self.start_token = start_token
        self.end_token = end_token
        self.system = system
        self.role = role
        self.random_shuffle = random_shuffle
        self.simple = simple
        # instruction location and number
        logger.info(f"Random shuffle: {self.random_shuffle}")

    def get_anno(self, index):
        filename = self.anno[index][self.media_type]
        qa = self.anno[index]["QA"]
        if "num_frames" in self.anno[index]:
            self.max_num_frames = self.anno[index]["num_frames"]
        if "start" in self.anno[index] and "end" in self.anno[index]:
            anno = {
                "image": os.path.join(self.data_root, filename), "qa": qa,
                "start": self.anno[index]["start"], "end": self.anno[index]["end"],
            }
        else:
            anno = {"image": os.path.join(self.data_root, filename), "qa": qa}
        return anno

    def __len__(self):
        return self.num_examples
    
    def process_qa(self, qa, msg=""):
        cur_instruction = ""
        # randomly shuffle qa for conversation
        if self.random_shuffle and len(qa) > 1:
            random.shuffle(qa)
        if "i" in qa[0].keys() and qa[0]["i"] != "":
            cur_instruction = qa[0]["i"] + self.end_signal

        conversation = self.system
        # add instruction as system message

        # rstrip() for the extra " " in msg
        place_holder = '' if self.place_holder.rstrip() in qa[0]["q"] else self.place_holder
        if not self.simple:
            if cur_instruction:
                conversation += cur_instruction
            conversation += (
                self.begin_signal + self.role[0] + ": " + 
                self.start_token + place_holder + self.end_token + msg.rstrip() + 
                qa[0]["q"] + self.end_signal + self.begin_signal + self.role[1] + ": "
            )
        else:
            conversation += place_holder
            conversation += (
                self.begin_signal + self.role[0] + ": " + cur_instruction + msg.rstrip() + 
                qa[0]["q"] + self.end_signal + self.begin_signal + self.role[1] + ": "
            )
        
        return conversation, qa[0]["a"]

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            image, index = self.load_and_transform_media_data_image(index, ann["image"])
            instruction, answer = self.process_qa(ann["qa"])
            return {
                "image": image,
                "answer": answer,
                "image_id": index,
                "instruction_input": instruction
            }
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading image {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)

class ITVidTrainDataset(ITImgTrainDataset):
    media_type = "video"

    def __init__(
        self, ann_file, transform, simple=False,
        num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=3,
        system="", role=("Human", "Assistant"),
        start_token="<Video>", end_token="</Video>",
        add_second_msg=False,
        random_shuffle=True,
    ):
        super().__init__(
            ann_file, transform, 
            system=system, role=role,
            start_token=start_token, end_token=end_token,
            random_shuffle=random_shuffle,
            simple=simple,
        )
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg

        logger.info(f"Use {video_reader_type} for data in {ann_file}")
        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            msg = ""
            clip = None
            if "start" in ann and "end" in ann:
                clip = [ann["start"], ann["end"]]
            video, index, sec = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True, clip=clip)
            if self.add_second_msg:
                # " " should be added in the start and end
                msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
            instruction, answer = self.process_qa(ann["qa"], msg)
            return {
                "image": video,
                "answer": answer,
                "image_id": index,
                "instruction_input": instruction,
                "video_len": sec
            }
        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)

class LLaVAITVidTrainDataset(ITImgTrainDataset):
    media_type = "video"

    def __init__(
        self, ann_file, transform, llama_model, simple=False,
        num_frames=4, video_reader_type="decord", sample_type="rand", num_tries=3,
        system="", role=("USER", "ASSISTANT"),
        start_token="", end_token="", place_holder="<image>\n",
        add_second_msg=False, begin_signal="", max_length=77, cut_clip=False, both_clip=False,
        random_shuffle=True, max_txt_l=1000, no_grid=False, clip_model=None
    ):
        super().__init__(
            ann_file, transform, 
            system=system, role=role,
            start_token=start_token, end_token=end_token,
            random_shuffle=random_shuffle,
            simple=simple, begin_signal=begin_signal,
            place_holder=place_holder
        )
        self.num_frames = num_frames
        self.video_reader_type = video_reader_type
        self.video_reader = VIDEO_READER_FUNCS[video_reader_type]
        self.sample_type = sample_type
        self.num_tries = num_tries
        self.add_second_msg = add_second_msg
        self.max_txt_l = max_txt_l
        self.max_length = max_length
        self.cut_clip = cut_clip
        self.both_clip = both_clip

        del self.transform
        if no_grid:
            self.processor = LlavaNextViDTextProcessor.from_pretrained(llama_model)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(llama_model)

        if clip_model:
            clip_processor = CLIPProcessor.from_pretrained(clip_model)
            self.clip_tokenizer = clip_processor.tokenizer
            del clip_processor
            
        logger.info(f"Use {video_reader_type} for data in {ann_file}")
        if add_second_msg:
            logger.info(f"Add second message: The video contains X frames sampled at T seconds.")
    
    def transform(self, vid):
        return self.processor.image_processor(vid, return_tensors='pt')

    def __getitem__(self, index):
        try:
            ann = self.get_anno(index)
            msg = ""
            clip = None
            if "start" in ann and "end" in ann:
                clip = [ann["start"], ann["end"]]
            video_embed, index, sec = self.load_and_transform_media_data_video(index, ann["image"], return_fps=True, clip=clip)
            if self.add_second_msg:
                # " " should be added in the start and end
                msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "

            instruction, answer = self.process_qa(ann["qa"], msg)

            #instruction_embed = self.processor.tokenizer(instruction, return_tensors='pt', 
            #                                             padding='max_length', max_length=self.max_txt_l)
            instruction_embed = self.processor.tokenizer(instruction, return_tensors='pt')
            q_ids, q_att = instruction_embed['input_ids'], instruction_embed['attention_mask']
            answer = answer + self.processor.tokenizer.eos_token
            answer_embed = self.processor.tokenizer(answer, return_tensors='pt', add_special_tokens=False)
            a_ids, a_att = answer_embed['input_ids'], answer_embed['attention_mask']

            q_len = sum(q_att[0])
            a_len = a_ids.size(1)

            input_ids = torch.cat([q_ids, a_ids], dim=1)
            input_att = torch.cat([q_att, a_att], dim=1)

            targets = torch.ones(input_ids.size(),dtype=torch.long).fill_(-100)
            targets[:, q_len:q_len+a_len] = a_ids
            targets = targets.squeeze()
            targets = F.pad(targets, (0, 1000 - len(targets)), value=-100)

            if hasattr(self,'clip_tokenizer'):
                if self.both_clip:
                    instruction = instruction + answer
                if self.cut_clip:
                    cut_answer = answer.split('.')
                    answer = cut_answer[0]
                clip_input = self.clip_tokenizer(instruction, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
                clip_ids, clip_mask = clip_input.input_ids, clip_input.attention_mask
                clip_answer = self.clip_tokenizer(answer, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
                clip_answer_ids, clip_answer_mask = clip_answer.input_ids, clip_answer.attention_mask

            else:
                clip_ids, clip_mask = None, None
            
            return_dict = {
                "input_ids": input_ids.squeeze(),
                "pixel_values": video_embed["pixel_values"],
                "image_sizes": video_embed["image_sizes"][0],
                "attention_mask": input_att.squeeze(),
                "labels": targets,
                "dict": False,
            }
            if clip_ids is not None:
                return_dict.update(
                    {   
                        "clip_answer_ids": clip_answer_ids.squeeze(),
                        "clip_answer_mask": clip_answer_mask.squeeze(),
                        "clip_ids": clip_ids.squeeze(),
                        "clip_mask": clip_mask.squeeze(),
                    }
                )
            return return_dict

        except Exception as e:
            logger.warning(f"Caught exception {e} when loading video {ann['image']}")
            index = np.random.randint(0, len(self))
            return self.__getitem__(index)

