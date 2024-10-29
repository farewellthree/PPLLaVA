import os
import re
import pandas as pd
import cv2

from  ppllava.test.video_utils import EvalDataset


video_root = '/Path/to/Video-MME/data'
subtitle_root = '/Path/to/Video-MME/subtitle'
data_dir = '/Path/to/Video-MME/videomme/test-00000-of-00001.parquet'
class VideoMME_dataset(EvalDataset):
    def __init__(self, num_segments=8, use_subtitles=False):
        super().__init__(num_segments=num_segments)
        
        id2q = {}
        data = pd.read_parquet(data_dir)
        for index, row in data.iterrows():
            video_id = row['videoID']
            question = {
                'question': row['question'],
                'options': row['options'],
                'answer': row['answer'],
                'question_id': row['question_id'],
                'task_type': row['task_type'],
            }
            if video_id not in id2q:
                id2q[video_id] = {
                "video_id": video_id,
                "duration": row['duration'],
                "domain": row['domain'],
                "sub_category": row['sub_category'],
                'data_type': "video",
                "question_metas": [question]
                }
            else:
                id2q[video_id]["question_metas"].append(question)
        self.data_list = list(id2q.values())

        #self.data_list = self.data_list[:10]
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        self.use_subtitles = use_subtitles

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        
        video_path = os.path.join(video_root, self.data_list[idx]['video_id']+'.mp4')
        assigned_frame = 32 if self.data_list[idx]['duration']=='short' else 64
        torch_imgs, frame_idxs = decord_method(video_path, bound=None, return_frame_idx=True, assigned_frame=assigned_frame)
        #torch_imgs, frame_idxs = decord_method(video_path, bound=None, return_frame_idx=True)
        subtitle_path = os.path.join(subtitle_root, self.data_list[idx]['video_id']+'.srt')
        if self.use_subtitles and os.path.exists(subtitle_path):
            subtitle_by_frame, total_frame = extract_subtitles(video_path, subtitle_path)
            subtitle_by_frame_idx = []
            for frame_idx in frame_idxs:
                for subtitle_idx, title in enumerate(subtitle_by_frame):
                    if frame_idx < title[1] and frame_idx >= title[0]:
                        subtitle_by_frame_idx.append(subtitle_idx)
            subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

            textlist = []
            for subtitle_idx in subtitle_by_frame_idx:
                pattern = r'<font color="white" size=".72c">(.*?)</font>'
                raw_text = re.findall(pattern, subtitle_by_frame[subtitle_idx][2])
                try:
                    textlist.append(raw_text[0])
                except:
                    continue
            #subtitles = "\n".join(textlist)
            subtitles = " ".join(textlist)
        else:
            subtitles = None
        
        questions = []
        for q in self.data_list[idx]['question_metas']:
            question = q['question']
            options = str(q['options'])
        #options = '\n'.join(self.data_list[idx]['options'])
            question = question + "\n" + options
            questions.append(question)

        return {
            'video': torch_imgs, 
            'video_id': self.data_list[idx]['video_id'],
            'questions': questions, 
            'subtitles': subtitles,
            'duration': self.data_list[idx]['duration'],
            'domain': self.data_list[idx]['domain'],
            'sub_category': self.data_list[idx]['sub_category'],
            'question_metas': self.data_list[idx]['question_metas'], 
        }

def infer_videomme_llava(
        model, processor, data_sample, 
        question, conv, system="", 
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        all_token=False,
        ask_simple=False,
    ):
    video = data_sample["video"]
    role = ("USER", "ASSISTANT")

    local_conv = conv.copy()

    if data_sample['subtitles']:
        local_conv.user_query("This video's subtitles are listed below: " + data_sample['subtitles'] + ' \n' + system + "\n" + question, is_mm=True)
    else:
        local_conv.user_query(system + "\n" + question, is_mm=True)
    local_conv.assistant_response(answer_prompt)
    prompt = local_conv.get_prompt()

    inputs = processor(prompt, question, video)
    inputs = inputs.to(model.device)
    if conv.sep[0]=='<|im_end|>\n': #qwen
        target_dtype = model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        inputs['pixel_values'] = inputs.pixel_values.to(f'cuda:{target_dtype}' if isinstance(target_dtype, int) else target_dtype)
    output = model.generate(**inputs, max_new_tokens=200)
    llm_message = processor.processor.decode(output[0], skip_special_tokens=True).split(answer_prompt)[1]
    #llm_message = llm_message.split(role[1])[1]

    # remove potential explanation
    llm_message = return_prompt + llm_message.strip()
    return llm_message

def infer_videomme_stllm(
        model, processor, data_sample,
        question, system="", 
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        all_token=False,
        ask_simple=False,
    ):
    pass

def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles

def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)

def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame

