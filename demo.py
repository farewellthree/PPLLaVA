import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

import argparse
import torch
from mmengine.fileio import FileClient
client = FileClient('disk')
from decord import VideoReader, cpu
import io
from PIL import Image
import numpy as np

# imports modules for registration
from ppllava.datasets.builders import *
from ppllava.models import *
from ppllava.processors import *
from ppllava.runners import *
from ppllava.tasks import *

from ppllava.common.config import Config
from ppllava.conversation.conv import conv_templates
from ppllava.test.video_utils import LLaVA_Processer, STLLM_Processer
from ppllava.common.registry import registry
from ppllava.test.vcgbench.utils import VideoChatGPTBenchDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.", default="config/vicuna/ppllava_vicuna7b_image_video.yaml")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--ckpt-path", help="path to ckpt file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def read_video(video_path, bound=None, num_segments=32):
    video_bytes = client.get(video_path)
    vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    images_group = list()
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments) 
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)

    return images_group

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')

args = parse_args()
cfg = Config(args)
model_config = cfg.model_cfg
llama_model = model_config.llama_model 
model_config.llama_model = args.ckpt_path
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_pretrained(args.ckpt_path, torch_dtype=torch.float16)
model = model.to('cuda:{}'.format(args.gpu_id))

processor = LLaVA_Processer(model_config)
processor.processor.image_processor.set_reader(video_reader_type='decord', num_frames=32)
if 'qwen' in llama_model:
    conv = conv_templates['plain_qwen']
elif 'llava' in llama_model:
    conv = conv_templates['plain_v1']
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list


def upload_video(gr_video, chat_state, num_segments, text_prompt='Watch the video and answer the question.'):
    # print('gr_video: ', gr_video)
    img_list = []
    # if gr_video: 
    #     chat_state = CONV_VISION.copy()
    #     chat.upload_video(gr_video, chat_state, img_list, num_segments, text=text_prompt)
    return gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state, gr_video, num_segments):
    chat_state = conv.copy()
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    img_list = read_video(gr_video, num_segments=num_segments)
    #img_list = gr_video
    chatbot = chatbot + [[user_message, None]]
    chat_state.user_query(user_message, is_mm=True)
    return '', chatbot, chat_state, img_list


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    full_question = chat_state.get_prompt()
    inputs = processor(full_question, chatbot[-1][0], img_list)

    inputs = inputs.to(model.device)
    if conv.sep[0]=='<|im_end|>\n': #qwen
        split_str = 'assistant\n'
        target_dtype = model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        inputs['pixel_values'] = inputs.pixel_values.to(f'cuda:{target_dtype}' if isinstance(target_dtype, int) else target_dtype)
    else:
        split_str = conv.roles[1]
    output = model.generate(**inputs, num_beams=num_beams, temperature=temperature, max_new_tokens=200)
    llm_message = processor.processor.decode(output[0], skip_special_tokens=True)
    llm_message = llm_message.split(split_str)[1].strip()
    chatbot[-1][1] = llm_message

    print(chat_state)
    print(f"Answer: {llm_message}")
    # print(chatbot)
    return chatbot, chat_state, img_list

class PPLLAVA(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = PPLLAVA(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )

title = """<h1 align="center"><a href="https://github.com/farewellthree/PPLLaVA"><img src="https://s21.ax1x.com/2024/11/04/pArvHxg.png" border="0" style="margin: 0 auto; height: 150px;" /></a> </h1>"""
description ="""
        CLICK FOR SOURCE CODE!<br><p><a href='https://github.com/farewellthree/PPLLaVA'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p>
        """

with gr.Blocks(title="PPLLaVA Chatbot!",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            with gr.Column(elem_id="image", scale=0.5) as img_part:
                with gr.Tab("Video", elem_id='video_tab'):
                    up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload").style(height=360)
            # text_prompt_input = gr.Textbox(value="Watch the video and answer the question.",show_label=False, placeholder='Input your text prompt, example: "Watch the video and answer the question."', interactive=True).style(container=False)           
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                interactive=True,
                label="beam search numbers",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            
            num_segments = gr.Slider(
                minimum=16,
                maximum=96,
                value=32,
                step=1,
                interactive=True,
                label="Video Segments",
            )
        
        with gr.Column(visible=True)  as input_raws:
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(elem_id="chatbot",label='PPLLaVA')
            with gr.Row():
                with gr.Column(scale=0.7):
                    text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first', interactive=False).style(container=False)
                with gr.Column(scale=0.15, min_width=0):
                    run = gr.Button("💭Send")
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("🔄Clear️")     
    
    upload_button.click(upload_video, [up_video, chat_state, num_segments], [up_video, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state, up_video, num_segments], [text_input, chatbot, chat_state, img_list]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    run.click(gradio_ask, [text_input, chatbot, chat_state, up_video, num_segments], [text_input, chatbot, chat_state, img_list]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    run.click(lambda: "", None, text_input)  
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, up_video, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(share=True, enable_queue=True)
