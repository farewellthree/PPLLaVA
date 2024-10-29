import os
import copy
import cv2
import math
import numpy as np
import io
from PIL import Image
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from ppllava.test.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)

import torch

from mmengine.fileio import FileClient
from torch.utils.data import Dataset
from typing import Any, List
from mmengine.fileio import FileClient
import imageio
client = FileClient('disk')

from ppllava.datasets.datasets.llavavid_processor import LlavaNextViDTextProcessor, LlavaOnevisionViDTextProcessor
from transformers import LlavaNextProcessor, CLIPProcessor, LlavaOnevisionProcessor, SiglipProcessor
from transformers.feature_extraction_utils import BatchFeature

def load_video(vis_path, n_clips=1, num_frm=100):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    assert n_clips == 1

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).numpy()
    
    img_array = img_array.reshape(
        (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs

def load_video_rawframes(vis_path, total_frame_num, n_clips=1, num_frm=100):
    # Currently, this function supports only 1 clip
    assert n_clips == 1
    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = get_frames_from_raw(vis_path, frame_idx)
    # Set target image height and width
    target_h, target_w = 224, 224
    # If image shape is not as target, resize it
    if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(f'cuda:{torch.uint8}' if isinstance(torch.uint8, int) else torch.uint8).numpy()

    # Reshape array to match number of clips and frames
    img_array = img_array.reshape(
        (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs

def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq

def get_frames_from_raw(directory, frame_idx, filename_tmpl="{:0>6}.jpg", offset=1):
    import mmcv
    mmcv.use_backend('cv2')
    file_client = FileClient('disk')
    imgs = list()
    cache = {}
    for i, frame_idx in enumerate(frame_idx):
        if frame_idx in cache:
            imgs.append(copy.deepcopy(imgs[cache[frame_idx]]))
            continue
        else:
            cache[frame_idx] = i
        frame_idx += offset
        filepath = os.path.join(directory, filename_tmpl.format(frame_idx))
        try:
            img_bytes = file_client.get(filepath)
        except:
            filepath = os.path.join(directory, filename_tmpl.format(frame_idx+1))
            img_bytes = file_client.get(filepath)
        cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        imgs.append(cur_frame)    
    return np.stack(imgs, axis=0)

class EvalDataset(Dataset):

    def __init__(self, num_segments, test_ratio=None):
        super().__init__()
        self.num_segments = num_segments
        self.test_ratio = test_ratio
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
    def __getitem__(self, index) -> Any:
        raise NotImplementedError('')
        
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0, assigned_frame=16):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)

        if bound:
            video_len = bound[1] - bound[0]
        else:
            video_len = max_frame / fps

        if self.num_segments > 0:
            num_segments = self.num_segments  
        elif self.num_segments == 0:
            num_segments = assigned_frame
        else:  #fps 1
            cur_len = video_len / abs(self.num_segments)
            if cur_len <= 32:
                num_segments = 32
            elif cur_len > 32:
                num_segments = 64
            #else:
            #    num_segments = math.floor(cur_len)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None, return_frame_idx=False, assigned_frame=16):
        video_bytes = client.get(video_path)
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, assigned_frame=assigned_frame) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        if return_frame_idx:
            return images_group, frame_indices
        return images_group
    
    def read_gif(self, video_path, bound=None, fps=25, assigned_frame=16):
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, assigned_frame=assigned_frame) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)

        return images_group
    
    def read_frame(self, video_path, bound=None, fps=3, assigned_frame=16, filename_tmpl="{:0>6}.jpg", offset=0):
        if os.path.exists(video_path):
            max_frame = len(os.listdir(video_path))
        else:
            max_frame = len([k for k in client.list(video_path)])
            
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1, assigned_frame=assigned_frame) # frame_idx starts from 1
        
        for frame_index in frame_indices:
            img_bytes = client.get(os.path.join(video_path, filename_tmpl.format(frame_index+offset)))
            img = Image.open(io.BytesIO(img_bytes))
            images_group.append(img)

        return images_group

    def set_rank_and_world_size(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        # self.data_list = self.data_list[::200] # debug
        if self.test_ratio is None:
            self.data_list = self.data_list[rank::world_size]
        else:
            np.random.RandomState(42).shuffle(self.data_list)
            if isinstance(self.test_ratio, float):
                num_samples = int(len(self.data_list) * self.test_ratio)
            else:
                num_samples = int(self.test_ratio)
            self.data_list = self.data_list[rank:num_samples:world_size]

class LLaVA_Processer:
    def __init__(self, model_config):
        llama_model = model_config.llama_model
        qwen = ('qwen' in llama_model)
        if model_config.arch=='llava_vid':
            process_cls = LlavaOnevisionProcessor if qwen else LlavaNextProcessor
            self.processor = process_cls.from_pretrained(llama_model)
        else:
            process_cls = LlavaOnevisionViDTextProcessor if qwen else LlavaNextViDTextProcessor
            self.processor = process_cls.from_pretrained(llama_model)
        
        if qwen:
            self.clip_length = 196 if model_config.get('extend_clip', False) else 64
        else:
            self.clip_length = 248 if model_config.get('extend_clip', False) else 77
  
        if model_config.get('clip_weight',None) is not None:
            clip_processor_cls = SiglipProcessor if qwen else CLIPProcessor
            clip_processor = clip_processor_cls.from_pretrained(model_config.clip_weight)
            self.clip_tokenizer = clip_processor.tokenizer
            del clip_processor
    
    def __call__(self, prompt, question, raw_frames):
        text_inputs = self.processor(text=prompt,return_tensors='pt')
        video_input = self.processor.image_processor(raw_frames,return_tensors='pt')
        video_input['pixel_values'] = video_input['pixel_values'].unsqueeze(0)
        video_input['image_sizes'] = video_input['image_sizes'][0].unsqueeze(0)
        inputs = BatchFeature(data={**text_inputs, **video_input})

        if hasattr(self,'clip_tokenizer'):
            clip_input = self.clip_tokenizer(question, return_tensors='pt', padding='max_length', truncation=True, max_length=self.clip_length)
            clip_ids = clip_input.input_ids
            clip_mask = clip_input.attention_mask if 'attention_mask' in clip_input else None
            inputs.update({"clip_ids": clip_ids,})
            if clip_mask is not None:
                inputs.update({"clip_mask": clip_mask,})                
        return inputs
    
class STLLM_Processer:
    def __init__(self, model_config, resolution=224):
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])

    def __call__(self, video):
        return self.transform(video)
