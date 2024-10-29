
import torch.nn as nn
import torch
import einops
from transformers import CLIPModel, CLIPConfig, SiglipModel, SiglipConfig
from transformers.models.clip.modeling_clip import clip_loss
from transformers.activations import ACT2FN
from transformers.models.llava_next.modeling_llava_next import  LlavaNextConfig

from ppllava.models.clip import GatherLayer, CLIPModelwithVideo, CLIPVideoConfig
from ppllava.models.siglip import SiglipModelwithVideo, SiglipVideoConfig, SiglipPoolingHead_PatchCLS
from ppllava.models.utils import weighted_adaptive_avg_pool3d_loop, weighted_adaptive_avg_pool3d_unfold, get_state_dict

class PllavaMultiModalProjector(nn.Module):
    supported_highres = ['pad_crop_four', 'slide', ]
    def __init__(self, config: LlavaNextConfig):
        super().__init__()  
        self.frame_shape=config.frame_shape
        self.pooling_shape = config.pllava_pooling_shape
        
        self.pooling = nn.AdaptiveAvgPool3d(self.pooling_shape)
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def convert_Fembeddings2video(self, input, num_videos, frame_shape):
        input = einops.rearrange(input, 
                                '(num_videos num_frames) (h w) embed_dims -> num_videos embed_dims num_frames h w', 
                                num_videos=num_videos, h=frame_shape[0])
        return input
    
    def convert_video2Fembeddings(self, input):
        input = einops.rearrange(input, 'num_videos embed_dims num_frames h w -> (num_videos num_frames) (h w) embed_dims ', )
        return input

    def convert_video2MMembeddings(self, input):
        input = einops.rearrange(input, 'num_videos embed_dims num_frames h w -> num_videos (num_frames h w) embed_dims ', )
        return input

    def forward(self, image_features, batch_size, num_frames, media_type='video'):
        frame_shape = self.frame_shape
        assert media_type in ( 'video', 'image'), f'only image or video, but got media_type {media_type}'
        hidden_states = image_features

        if media_type == 'image':
            hidden_states = hidden_states.repeat(num_frames, 1, 1)

        total_frames, spatial_seqlen, embed_dims = hidden_states.shape
        #TODO: temporal code, should ensure num_frames == total frames in data loading later
        if total_frames < num_frames: # 
            multiplier = int(num_frames/total_frames)+1
            hidden_states= hidden_states.repeat_interleave(multiplier, dim=0)[:num_frames]
            total_frames, spatial_seqlen, embed_dims = hidden_states.shape

        assert total_frames % num_frames == 0
        assert frame_shape[0] * frame_shape[1] == spatial_seqlen
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states_videos = self.convert_Fembeddings2video(hidden_states, batch_size, frame_shape)
        hidden_states_videos = self.pooling(hidden_states_videos)
        hidden_states = einops.rearrange(hidden_states_videos, 'batch_size_num_videos embed_dims num_frames h w -> batch_size_num_videos num_frames (h w) embed_dims', )
        hidden_states = einops.rearrange(hidden_states, 'batch_size_num_videos num_frames hw embed_dims -> batch_size_num_videos (num_frames hw) embed_dims ')
        return hidden_states

#npu has not supported AdaptiveAvgPool3d yet
class PllavaNPU3DMultiModalProjector(nn.Module):
    supported_highres = ['pad_crop_four', 'slide', ]
    def __init__(self, config: LlavaNextConfig):
        super().__init__()  
        self.frame_shape = config.frame_shape
        self.pooling_shape = config.pllava_pooling_shape
        
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def convert_Fembeddings2video(self, input, num_videos, frame_shape):
        input = einops.rearrange(input, 
                                '(num_videos num_frames) (h w) embed_dims -> num_videos embed_dims num_frames h w', 
                                num_videos=num_videos, h=frame_shape[0])
        return input
    
    def convert_video2Fembeddings(self, input):
        input = einops.rearrange(input, 'num_videos embed_dims num_frames h w -> (num_videos num_frames) (h w) embed_dims ', )
        return input

    def convert_video2MMembeddings(self, input):
        input = einops.rearrange(input, 'num_videos embed_dims num_frames h w -> num_videos (num_frames h w) embed_dims ', )
        return input

    def forward(self, image_features, batch_size, num_frames, media_type='video'):
        frame_shape = self.frame_shape
        assert media_type in ( 'video', 'image'), f'only image or video, but got media_type {media_type}'
        hidden_states = image_features

        if media_type == 'image':
            hidden_states = hidden_states.repeat(num_frames, 1, 1)

        total_frames, spatial_seqlen, embed_dims = hidden_states.shape
        #TODO: temporal code, should ensure num_frames == total frames in data loading later
        if total_frames < num_frames: # 
            multiplier = int(num_frames/total_frames)+1
            hidden_states= hidden_states.repeat_interleave(multiplier, dim=0)[:num_frames]
            total_frames, spatial_seqlen, embed_dims = hidden_states.shape

        assert total_frames % num_frames == 0
        assert frame_shape[0] * frame_shape[1] == spatial_seqlen
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states_videos = self.convert_Fembeddings2video(hidden_states, batch_size, frame_shape)
        hidden_states_videos = weighted_adaptive_avg_pool3d_unfold(hidden_states_videos, self.pooling_shape)
        hidden_states = einops.rearrange(hidden_states_videos, 'batch_size_num_videos embed_dims num_frames h w -> batch_size_num_videos num_frames (h w) embed_dims', )
        hidden_states = einops.rearrange(hidden_states, 'batch_size_num_videos num_frames hw embed_dims -> batch_size_num_videos (num_frames hw) embed_dims ')
        return hidden_states
      
class LLaVACLIP3DMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()  

        self.frame_shape = config.frame_shape
        self.pooling_shape = config.pllava_pooling_shape
        self.pooling_kernel = config.pooling_kernel
        self.pooling_stride = config.pooling_stride
        self.image_pooling_kernel = config.image_pooling_kernel if config.image_pooling_kernel is not None else config.pooling_kernel
        self.image_pooling_stride = config.image_pooling_stride if config.image_pooling_stride is not None else config.pooling_stride
        self.pooling_temp = config.pooling_temp

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

        if config.extend_clip:
            clipvideoconfig = CLIPVideoConfig.from_dict(CLIPConfig._get_config_dict(config.clip_weight)[0])
            clip_model = CLIPModelwithVideo(clipvideoconfig)
            clip_model2 = CLIPModel.from_pretrained(config.clip_weight)
            clip_model.load_state_dict(clip_model2.state_dict())
            clip_model.convert_text_position()
            del clip_model2
            if config.clip_post_pretrain:
                postpretrain_weight = get_state_dict(config.clip_post_pretrain)
                clip_model.load_state_dict(postpretrain_weight)
        else:
            clipconfig = config.clip_post_pretrain if config.clip_post_pretrain is not None else config.clip_weight
            clip_model = CLIPModel.from_pretrained(clipconfig)
        
        self.clip_text_model = clip_model.text_model
        self.clip_visual_projection = clip_model.visual_projection
        self.clip_visual_ln = clip_model.vision_model.post_layernorm
        self.clip_text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        self.logit_scale.requires_grad = False
        del clip_model
    
    def forward(self, image_features, clip_ids, clip_mask, num_frames,  video=True, output_visual_attention=False):
        hidden_states = image_features

        image_embeds = self.clip_visual_projection(self.clip_visual_ln(image_features))
        
        image_embeds = einops.rearrange(image_embeds, '(B T) L D -> B (T L) D', T=num_frames)
        
        with torch.no_grad():
            text_embeds = self.clip_text_model(input_ids=clip_ids, attention_mask=clip_mask)[1]
        #text_embeds = self.clip_text_model(input_ids=clip_ids, attention_mask=clip_mask)[1]

        text_embeds = self.clip_text_projection(text_embeds)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        retrieve_logits = torch.einsum('ad,bvd->abv', [text_embeds, image_embeds])
        retrieve_logits = retrieve_logits.diagonal(dim1=0, dim2=1).transpose(0, 1).contiguous()
        if output_visual_attention:
            retrieve_logits = einops.rearrange(retrieve_logits, 'B (T W H) -> B T W H', T=num_frames, W=self.frame_shape[0])
            return retrieve_logits
        
        tv_softmax = torch.softmax(retrieve_logits*logit_scale, dim=-1) 
        tv_softmax = einops.rearrange(tv_softmax, 'B (T W H) -> B T W H', T=num_frames, W=self.frame_shape[0])
        
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = einops.rearrange(hidden_states, '(B T) (W H) D -> B D T W H', T=num_frames, W=self.frame_shape[0])
        if video:
            hidden_states = weighted_adaptive_avg_pool3d_unfold(hidden_states, self.pooling_shape, self.pooling_kernel, self.pooling_stride, tv_softmax, self.pooling_temp)
        else:
            hidden_states = weighted_adaptive_avg_pool3d_unfold(hidden_states, self.pooling_shape, self.image_pooling_kernel, self.image_pooling_stride, tv_softmax, self.pooling_temp)
        hidden_states = einops.rearrange(hidden_states, 'B D T W H -> B T (W H) D', )
        hidden_states = einops.rearrange(hidden_states, 'B T WH D -> B (T WH) D ')

        return hidden_states

class LLaVASiglip3DMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()  

        self.frame_shape = config.frame_shape
        self.pooling_shape = config.pllava_pooling_shape
        self.pooling_kernel = config.pooling_kernel
        self.pooling_stride = config.pooling_stride
        self.image_pooling_kernel = config.image_pooling_kernel if config.image_pooling_kernel is not None else config.pooling_kernel
        self.image_pooling_stride = config.image_pooling_stride if config.image_pooling_stride is not None else config.pooling_stride
        self.pooling_temp = config.pooling_temp
        self.valuew_plus = config.valuew_plus

        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

        if config.extend_clip:
            clipvideoconfig = SiglipVideoConfig.from_dict(SiglipConfig._get_config_dict(config.clip_weight)[0])
            clip_model = SiglipModelwithVideo(clipvideoconfig)
            clip_model2 = SiglipModel.from_pretrained(config.clip_weight)
            clip_model.load_state_dict(clip_model2.state_dict())
            clip_model.convert_text_position()
            del clip_model2
            if config.clip_post_pretrain:
                postpretrain_weight = get_state_dict(config.clip_post_pretrain)
                clip_model.load_state_dict(postpretrain_weight)
        else:
            clipconfig = config.clip_post_pretrain if config.clip_post_pretrain is not None else config.clip_weight
            clip_model = CLIPModel.from_pretrained(clipconfig)
        
        self.clip_text_model = clip_model.text_model
        #self.clip_visual_head = clip_model.vision_model.head
        self.clip_visual_head = SiglipPoolingHead_PatchCLS(config.vision_config)
        self.clip_visual_head.load_state_dict(clip_model.vision_model.head.state_dict())
        self.clip_visual_ln = clip_model.vision_model.post_layernorm
        self.logit_scale = clip_model.logit_scale
        self.logit_bias = clip_model.logit_bias
        self.logit_scale.requires_grad = False
        self.logit_bias.requires_grad = False
        del clip_model
    
    def forward(self, image_features, clip_ids, clip_mask, num_frames, after=True, video=True, output_visual_attention=False):
        hidden_states = image_features

        image_embeds, attn_weights = self.clip_visual_head(self.clip_visual_ln(image_features))
        
        image_embeds = einops.rearrange(image_embeds, '(B T) L D -> B (T L) D', T=num_frames)
        
        text_embeds = self.clip_text_model(input_ids=clip_ids, attention_mask=clip_mask)[1]

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        retrieve_logits = torch.einsum('ad,bvd->abv', [text_embeds, image_embeds])
        retrieve_logits = retrieve_logits.diagonal(dim1=0, dim2=1).transpose(0, 1).contiguous()
        if output_visual_attention:
            retrieve_logits = einops.rearrange(retrieve_logits, 'B (T W H) -> B T W H', T=num_frames, W=self.frame_shape[0])
            return retrieve_logits
        
        tv_softmax = torch.softmax(retrieve_logits*logit_scale + self.logit_bias, dim=-1) 
        tv_softmax = einops.rearrange(tv_softmax, 'B (T W H) -> B T W H', T=num_frames, W=self.frame_shape[0])
        
        if self.valuew_plus:
            tv_softmax = attn_weights.mean(1).view(1,num_frames,self.frame_shape[0],self.frame_shape[0]) * tv_softmax * 1000

        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = einops.rearrange(hidden_states, '(B T) (W H) D -> B D T W H', T=num_frames, W=self.frame_shape[0])
        if video:
            hidden_states = weighted_adaptive_avg_pool3d_unfold(hidden_states, self.pooling_shape, self.pooling_kernel, self.pooling_stride, tv_softmax, self.pooling_temp)
        else:
            hidden_states = weighted_adaptive_avg_pool3d_unfold(hidden_states, self.pooling_shape, self.image_pooling_kernel, self.image_pooling_stride, tv_softmax, self.pooling_temp)

        hidden_states = einops.rearrange(hidden_states, 'B D T W H -> B T (W H) D', )
        hidden_states = einops.rearrange(hidden_states, 'B T WH D -> B (T WH) D ')

        return hidden_states


