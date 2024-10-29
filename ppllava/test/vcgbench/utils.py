import os
import json
from typing import OrderedDict
from  ppllava.test.video_utils import EvalDataset

class VideoChatGPTBenchDataset(EvalDataset):
    data_dir = "/Path/to/video_chatgpt"
    data_list_info = OrderedDict({
        "generic_qa": OrderedDict(
            json_relpath="Benchmarking_QA/generic_qa.json", 
            prefix="/Path/to/video_chatgpt/Test_Videos", 
            data_type="video", 
            bound=False,
            question_key='Q',
            answer_key='A',
            name_key='video_name',
            postfix=('mp4', 'mkv', 'mov', 'avi'),
        ),
        "temporal_qa": OrderedDict(
            json_relpath="Benchmarking_QA/temporal_qa.json", 
            prefix="/Path/to/video_chatgpt/Test_Videos", 
            data_type="video", 
            bound=False,
            question_key='Q',
            answer_key='A',
            name_key='video_name',
            postfix=('mp4', 'mkv', 'mov', 'avi')
        ), # don't has start & end
        "consistency_qa":  OrderedDict( 
            # consistency is quite different in evaluating, and also awkward, hold to later.
            json_relpath="Benchmarking_QA/consistency_qa.json", 
            prefix="/Path/to/video_chatgpt/Test_Videos", 
            data_type="video", 
            bound=False,
            question_key=('Q1', 'Q2'),
            answer_key='A',
            name_key='video_name',
            postfix=('mp4', 'mkv', 'mov', 'avi'),
        ),
    })

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_list_info = self.data_list_info
        data_dir = self.data_dir

        self.data_list = []
        for k, v in data_list_info.items():
            with open(os.path.join(data_dir, v['json_relpath']), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'data': data,
                    **v, # all the infos
                })
                
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
     
    def __getitem__(self, idx):
        task_type = self.data_list[idx]['task_type']
        video_name_key = self.data_list[idx]['name_key']
        video_name = self.data_list[idx]['data'][video_name_key]
        video_postfixs = self.data_list[idx]['postfix']
        
        if self.num_segments != 0:
            for p in video_postfixs:
                temp_path = os.path.join(self.data_list[idx]['prefix'], video_name + '.' + p)
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break

            # video_filename = self.data_list[idx]['data'][video_name_key] + video_postfix
            decord_method = self.decord_method[self.data_list[idx]['data_type']]
            bound = None
            if self.data_list[idx]['bound']:
                bound = (
                    self.data_list[idx]['data']['start'],
                    self.data_list[idx]['data']['end'],
                )
            images_group = decord_method(video_path, bound)
        else:
            # zero frame, no image
            images_group = None

        data = {
            'video_path': video_path,
            'video_name': video_name,
            'video': images_group, # some might use the original pils and do their own transforms
            'task_type': task_type,
        }


        answer_key = self.data_list[idx]['answer_key']
        question_key = self.data_list[idx]['question_key']
        
        if task_type == 'consistency_qa' and isinstance(question_key, tuple):
            question=self.data_list[idx]['data'][question_key[0]]
            question1=self.data_list[idx]['data'][question_key[1]]
            answer=self.data_list[idx]['data'][answer_key]    

            data.update({
                'question': question, 
                'question1': question1, 
                'answer': answer,
            })
        elif isinstance(question_key, str):
            question=self.data_list[idx]['data'][question_key]
            answer=self.data_list[idx]['data'][answer_key]
            data.update({
                'question': question, 
                'answer': answer,
            })
        else:
            raise ValueError('')

        return data

def infer_vcgbench_llava(
        model, processor,
        data_sample, conv
    ):
    video = data_sample["video"]
    question = data_sample['question']

    local_conv = conv.copy()
    local_conv.user_query(question, is_mm=True)
    full_question = local_conv.get_prompt()

    inputs = processor(full_question, question, video)
    inputs = inputs.to(model.device)

    if conv.sep[0]=='<|im_end|>\n': #qwen
        split_str = 'assistant\n'
        target_dtype = model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        inputs['pixel_values'] = inputs.pixel_values.to(f'cuda:{target_dtype}' if isinstance(target_dtype, int) else target_dtype)
    else:
        split_str = conv.roles[1]
    output = model.generate(**inputs, max_new_tokens=200)
    llm_message = processor.processor.decode(output[0], skip_special_tokens=True)
    #llm_message = processor.processor.decode(output[0])
    llm_message = llm_message.split(split_str)[1].strip()
    return llm_message



def infer_vcgbench_stllm(
        model, processor,
        data_sample, system="", 
    ):
    pass


def save_results(result_list, save_path):
    general = []
    temporal = []
    consist = []

    for res in result_list:
        task_type = res.pop('task_type')
        if task_type=="generic_qa":
            general.append(res)
        elif task_type=="temporal_qa":
            temporal.append(res)
        elif task_type=="consistency_qa":
            consist.append(res)
        
    
    with open(os.path.join(save_path, f"general.json"), 'w') as f:
        json.dump(general, f)
    with open(os.path.join(save_path, f"temporal.json"), 'w') as f:
        json.dump(temporal, f)
    with open(os.path.join(save_path, f"consist.json"), 'w') as f:
        json.dump(consist, f)