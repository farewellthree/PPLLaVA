import os
import json
from typing import OrderedDict
from  ppllava.test.video_utils import EvalDataset

class VideoQABenchDataset(EvalDataset):
    data_list_info = OrderedDict({
        "MSVD": OrderedDict(
            data_dir='/Path/to/MSVD/YouTubeClips',
            gt_file='/Path/to/MSVD/ft_local/MSVD-QA/new_test_qa.json',
            data_type="video", 
            bound=False,
            question_key='question',
            answer_key='answer',
            name_key='video_name',
            postfix=('avi',),
        ),
        "MSRVTT": OrderedDict(
            data_dir='Path/to/MSRVTT-QA/video/',
            gt_file='Path/to/MSRVTT-QA/test_qa.json',
            data_type="video", 
            bound=False,
            question_key='question',
            answer_key='answer',
            name_key='video_name',
            postfix=('mp4', ),
        ), # don't has start & end 
        "ActivityNet": OrderedDict(
            data_dir='/Path/to/ActivityNet/activitynet_frames',
            q_json_relpath="/Path/to/video_chatgpt/test_q.json", 
            a_json_relpath="/Path/to/video_chatgpt/test_a.json", 
            data_type="frame", 
            bound=False,
            question_key='question',
            answer_key='answer',
            name_key='video_name',
            postfix=('mp4', 'mkv', 'webm'),
        ), # don't has start & end

    })

    def __init__(self, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_list_info = self.data_list_info[dataset]
        self.data_list = []
        
        if dataset=='ActivityNet':
            with open(data_list_info.pop('q_json_relpath'), 'r') as f:
                gt_questions = json.load(f)
            with open(data_list_info.pop('a_json_relpath'), 'r') as f:
                gt_answers = json.load(f)
            
            for i, sample in enumerate(gt_questions):
                video_name = 'v_' + sample['video_name']
                path = os.path.join(self.data_list_info[dataset]['data_dir'], video_name)
                video_name = video_name if os.path.exists(path) else video_name + '.webm'

                question = sample['question']
                id = sample['question_id']
                answer = gt_answers[i]['answer']

                self.data_list.append({'video_name':video_name, 'id': id, 'question': question, 'answer': answer, **data_list_info})
        
        else:
            with open(data_list_info.pop('gt_file'), 'r') as f:
                gt_file = json.load(f)
            
            for sample in gt_file:
                if 'video_name' in sample:
                    video_name = sample['video_name']
                elif 'video' in sample:
                    video_name = sample['video']
                else:
                    video_name = str(sample['video_id']) + '.mp4'
                question = sample['question']
                answer = sample['answer']
                id = sample['question_id'] if 'question_id' in sample else sample['id']
                self.data_list.append({'video_name':video_name, 'id': id, 'question': question, 'answer': answer, **data_list_info})

        print(len(self.data_list))
        
    def __len__(self):
        return len(self.data_list)
     
    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        
        video_name_key = self.data_list[idx]['name_key']
        video_name = self.data_list[idx][video_name_key]

        video_path = os.path.join(self.data_list[idx]['data_dir'], video_name)
        images_group = decord_method(video_path, None)

        question_key = self.data_list[idx]['question_key']
        answer_key = self.data_list[idx]['answer_key']
        question = self.data_list[idx][question_key]
        answer = self.data_list[idx][answer_key]
        
        return {
            'video': images_group, # some might use the original pils and do their own transforms
            'question': question,
            'video_path': video_path,
            'answer': answer,
            'id': self.data_list[idx]['id'],
        }

def infer_qabench_llava(
        model, processor,
        data_sample, conv,
    ):
    video = data_sample["video"]
    question = data_sample['question']

    local_conv = conv.copy()
    local_conv.user_query(question, pre_query_prompt=conv.pre_query_prompt, is_mm=True)
    local_conv.assistant_response(conv.answer_prompt)
    full_question = local_conv.get_prompt()

    inputs = processor(full_question, question, video)
    inputs = inputs.to(model.device)

    if conv.sep[0]=='<|im_end|>\n': #qwen
        target_dtype = model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        inputs['pixel_values'] = inputs.pixel_values.to(f'cuda:{target_dtype}' if isinstance(target_dtype, int) else target_dtype)

    output = model.generate(**inputs, max_new_tokens=200)
    llm_message = processor.processor.decode(output[0], skip_special_tokens=True)

    llm_message = llm_message.split(conv.answer_prompt)[1].strip()
    return llm_message

def infer_qabench_stllm(
        model, processor,
        data_sample, system="", 
    ):
    pass
