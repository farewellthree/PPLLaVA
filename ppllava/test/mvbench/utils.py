import os
import json

from  ppllava.test.video_utils import EvalDataset


mvbench_path = '/Path/to/MVBench'
data_list = {
    "Action Sequence": ("action_sequence.json", f"{mvbench_path}/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", f"{mvbench_path}/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", f"{mvbench_path}/video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", f"{mvbench_path}/video/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", f"{mvbench_path}/video/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", f"{mvbench_path}/video/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", f"{mvbench_path}/video/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", f"{mvbench_path}/video/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", f"{mvbench_path}/video/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", f"{mvbench_path}/video/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", f"{mvbench_path}/video/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", f"{mvbench_path}/video/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", f"{mvbench_path}/video/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", f"{mvbench_path}/video/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", f"{mvbench_path}/video/perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", f"{mvbench_path}/video/nturgbd/", "video", False),
    "Character Order": ("character_order.json", f"{mvbench_path}/video/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", f"{mvbench_path}/video/vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", f"{mvbench_path}/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", f"{mvbench_path}/video/clevrer/video_validation/", "video", False),
}

class MVBench_dataset(EvalDataset):
    def __init__(self, data_dir=f"{mvbench_path}/json", data_list=data_list, num_segments=8, resolution=224, specified_item=None):
        super().__init__(num_segments=num_segments)
        self.data_list = []
        if specified_item:
            data_list = {specified_item: data_list[specified_item]}
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })

        #self.data_list = self.data_list[:10]
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
    
    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        if self.data_list[idx]['data_type']=='frame':
            torch_imgs = decord_method(video_path, bound, filename_tmpl="{:0>5}.jpg")
        else:
            torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'video_path': video_path,
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type'],
        }

def infer_mvbench_llava(
        model, processor,
        data_sample, conv, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_llm=False,
        all_token=False,
        ask_simple=False,
    ):
    video = data_sample["video"]
    
    if system_llm:
        question = system + data_sample['question'] + question_prompt
    else:
        question = data_sample['question'] + question_prompt

    local_conv = conv.copy()
    local_conv.user_query(question, is_mm=True)
    local_conv.assistant_response(answer_prompt)
    prompt = local_conv.get_prompt()
    
    inputs = processor(prompt, data_sample['question'], video)
    inputs = inputs.to(model.device)

    if conv.sep[0]=='<|im_end|>\n': #qwen
        target_dtype = model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        inputs['pixel_values'] = inputs.pixel_values.to(f'cuda:{target_dtype}' if isinstance(target_dtype, int) else target_dtype)
    output = model.generate(**inputs, max_new_tokens=200)
    llm_message = processor.processor.decode(output[0], skip_special_tokens=True).split(answer_prompt)[1]

    # remove potential explanation
    llm_message = return_prompt + llm_message.strip()
    return llm_message

def infer_mvbench_stllm(
        model, processor,
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_llm=False,
        all_token=False,
        ask_simple=False,
    ):
    pass

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

def save_results(result_list, save_path, save_name):
    final_res, acc_dict = {}, {}
    correct, total = 0, 0
    for res in result_list:
        task_type = res['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred = res['pred']
        gt = res['gt']
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1

    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]    
    final_res['Avg'] = correct / total * 100

    acc_dict['Total Acc'] = correct / total * 100
    all_results = {
        "acc_dict": acc_dict,
        "result_list": result_list
    }
    
    print ('Total Acc:', correct / total * 100)
    with open(os.path.join(save_path, f"{save_name}.json"), 'w') as f:
        json.dump(all_results, f)