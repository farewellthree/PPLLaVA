from webbrowser import get

import torch
import torch.nn as nn
from torch.utils import checkpoint
import numpy as np

from einops import rearrange

from transformers import CLIPModel, CLIPConfig, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPAttention, CLIPMLP, \
    CLIPVisionTransformer, CLIPPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from mmengine.model import constant_init
from mmengine.runner.checkpoint import _load_checkpoint

class CLIPVisionModel_BTAdapter(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer_BTAdapter(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(self, pixel_values, **kwargs): 
        return self.vision_model(
            pixel_values=pixel_values,
        )

class CLIPVisionTransformer_BTAdapter(nn.Module):
    def __init__(self, configuration: CLIPVisionConfig): 

        super().__init__()

        self.depth = configuration.depth
        dpr = np.linspace(0, 0.1, self.depth)

        self.num_patches = (configuration.image_size // configuration.patch_size) ** 2
        self.embed_dim = configuration.hidden_size

        clip_vision_model = CLIPVisionTransformer(configuration)

        self.encoder = clip_vision_model.encoder
        self.embeddings = clip_vision_model.embeddings
        self.pre_layrnorm = clip_vision_model.pre_layrnorm
        self.post_layernorm = clip_vision_model.post_layernorm

        self.btadapter_S_layers = nn.ModuleList([CLIPLayer_Spatial(configuration, dpr[i]) for i in range(self.depth)])
        self.btadapter_T_layers = nn.ModuleList([CLIPLayer_AttnTime(configuration, dpr[i]) for i in range(self.depth)])
        
        self.btadapter_time_embed = nn.Embedding(configuration.max_T, self.embed_dim)
        
        self.btadapter_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.drop_after_pos = nn.Dropout(p=0)
        self.drop_after_time = nn.Dropout(p=0)

    def init_weights(self):
        total_depth = len(self.encoder.layers)
        layer_para = self.encoder.layers.state_dict()
        spatial_para = {}
        load_start = total_depth - self.depth
        for k, v in layer_para.items():
            num_layer = int(k.split(".")[0])
            if num_layer >= load_start:
                spatial_para[k.replace(str(num_layer),str(num_layer-load_start),1)] = v.clone()
        self.btadapter_S_layers.load_state_dict(spatial_para)

    def forward_embedding(self, x):
        target_dtype = self.embeddings.patch_embedding.weight.dtype
        batch_size = x.shape[0]
        patch_embeds = self.embeddings.patch_embedding(x.to(f'cuda:{target_dtype}' if isinstance(target_dtype, int) else target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.embeddings.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_ids = torch.arange(self.num_patches + 1, dtype=torch.long, device=f'cuda:{x.device}' if isinstance(x.device, int) else x.device).expand((1, -1))
        embeddings = embeddings + self.embeddings.position_embedding(position_ids)
        return embeddings
    
    def get_combined_embedding0(self,x,x2,T):
        x_patch, x2_patch = x[:,1:,], x2[:,1:,]

        x_cls = rearrange(x[:,:1], '(b t) p m -> b t p m', t=T).mean(dim=1)
        x2_cls = x2[:,:1,]
        x_patch = rearrange(x_patch, '(b t) p m -> b t p m', t=T).mean(dim=1)
        p = x_patch.size(1)
        x2_patch = rearrange(x2_patch, 'b (p t) m -> b t p m', p=p).mean(dim=1)
        
        combine = torch.cat(((x_patch+x2_patch)/2,(x_cls+x2_cls)/2),dim=1)
        return combine
    
    def get_combined_embedding(self,x,x2,T):
        x_patch, x2_patch = x[:,1:,], x2[:,1:,]

        x_cls = x[:,:1]
        p = x_patch.size(1)
        x2_patch = rearrange(x2_patch, 'b (p t) m -> (b t) p m', p=p)
        
        combine = torch.cat((x_cls, (x_patch+x2_patch)/2),dim=1)
        return combine

    def forward_patch(self, x, T):
        if getattr(self, 'pre_layrnorm', None) is not None:
            x = self.pre_layrnorm(x)
        total = len(self.encoder.layers)
        x2 = None
        encoder_states = ()
        for idx, encoder_layer in enumerate(self.encoder.layers):
            if x2 is None:
                encoder_states = encoder_states + (x,)
            else:
                combine = self.get_combined_embedding(x,x2,T)
                encoder_states = encoder_states + (combine,)

            layer_outputs = encoder_layer(x, None, None)
            x = layer_outputs[0]

            if idx >= total-self.depth:
                num_layer = idx + self.depth - total
                x2 = self.forward_BTAdapter(x, x2, num_layer, T)
    
        encoder_states = encoder_states + (self.get_combined_embedding(x,x2,T),)
        
        cls_token = x[:, 0] + x2[:, 0].repeat(1, T).view(x2.size(0) * T, -1)
        cls_token = self.post_layernorm(cls_token)

        return encoder_states, cls_token

    def forward_BTAdapter(self, x1, x2, num_layer, T):
        x1 = rearrange(x1, '(b t) l d -> b t l d', t=T)
        assert T <= self.btadapter_time_embed.num_embeddings

        if x2 is not None:
            cls_token = x1[:, :, 0, :].mean(dim=1).unsqueeze(1)
            x1 = x1[:, :, 1:, :]
            x1 = rearrange(x1, 'b t l d -> b (l t) d')
            x1 = torch.cat((cls_token, x1), dim=1)
            
            x = x2 + x1
            
        else:
            x = x1
        
        if num_layer==0:
            x = self.input_ini(x)

        if self.training:  
            x = checkpoint.checkpoint(self.btadapter_T_layers[num_layer],x, T)
            x = checkpoint.checkpoint(self.btadapter_S_layers[num_layer],x, T)
        else:
            x = self.btadapter_T_layers[num_layer](x, T)
            x = self.btadapter_S_layers[num_layer](x, T)
        return x 

    def input_ini(self, x):
        cls_old = x[:, :, 0, :].mean(dim=1).unsqueeze(1)
        x = x[:,:,1:,:]
        B,T,L,D = x.size()
        x = rearrange(x, 'b t l d -> (b t) l d')
        #cls_tokens = self.class_embedding.expand(x.size(0), 1, -1)
        cls_tokens = self.btadapter_cls_token.expand(x.size(0), 1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=f'cuda:{x.device}' if isinstance(x.device, int) else x.device).unsqueeze(0).expand(x.size(0), -1)
        pos_embed = self.embeddings.position_embedding(position_ids)
        x = x + pos_embed
        x = self.drop_after_pos(x)
        cls = x[:B, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) l d -> (b l) t d', b=B)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=f'cuda:{x.device}' if isinstance(x.device, int) else x.device).unsqueeze(0).expand(x.size(0), -1)
        time_embed = self.btadapter_time_embed(position_ids)
        x = x + time_embed
        x = self.drop_after_time(x)
        x = rearrange(x, '(b l) t d -> b (l t) d', b=B)
        cls = (cls_old + cls) / 2
        x = torch.cat((cls, x), dim=1)
        return x 

    def forward(self, pixel_values, **kwargs):
        x = pixel_values
        if x.ndim == 5:
            # B, 3, num_frames, 224, 224
            if x.shape[1]==3:
                B, D, T, H, W = x.shape             
                x = x.permute(0, 2, 1, 3, 4)
            else:
                B, T, D, H, W = x.shape   
            x = x.reshape((-1,) + x.shape[2:])
        elif x.ndim == 4:
            T = 1
        
        x = self.forward_embedding(x)
        vision_outputs = self.forward_patch(x, T)
        
        x_tokens, x_cls = vision_outputs

        return BaseModelOutputWithPooling(
            pooler_output=x_cls,
            hidden_states=x_tokens,
        )
    
class CLIPLayer_Spatial(nn.Module):
    def __init__(self, config: CLIPConfig, layer_num=0.1):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = DropPath(
            layer_num) if layer_num > 0. else nn.Identity()
       
    def forward(self, hidden_states, T):
        residual = hidden_states

        #init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        #query_s = hidden_states[:, 1:, :]
        
        init_cls_token = hidden_states[:, :1, :]
        query_s = hidden_states[:, 1:, :]

        b, pt, m = query_s.size()
        p, t = pt // T, T
        cls_token = init_cls_token.unsqueeze(1).repeat(1, t, 1, 1).reshape(b * t, 1, m) 
        #cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t, m).unsqueeze(1)
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        hidden_states = torch.cat((cls_token, query_s), 1)
        #hidden_states = self.process.before(hidden_states)

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
        )

        res_spatial = self.dropout_layer(
            self.proj_drop(hidden_states.contiguous()))
        cls_token = res_spatial[:, :1, :].reshape(b, T, 1, m)
        cls_token = torch.mean(cls_token, 1)
        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=T)
        hidden_states = torch.cat((cls_token, res_spatial), 1)
        #hidden_states = self.process.after(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class CLIPLayer_AttnTime(nn.Module):
    def __init__(self, config: CLIPConfig, T, layer_num=0.1):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)

        self.proj_drop = nn.Dropout(0)
        self.dropout_layer = DropPath(
            layer_num) if layer_num > 0. else nn.Identity()
        
        self.temporal_fc = nn.Linear(self.embed_dim, self.embed_dim)
        constant_init(self.temporal_fc, val=0, bias=0)

    def forward(self, hidden_states, T):
        residual = hidden_states[:, 1:, :]

        #init_cls_token = hidden_states[:, 0, :].unsqueeze(1)
        #query_t = hidden_states[:, 1:, :]
        init_cls_token = hidden_states[:, :1, :]
        query_t = hidden_states[:, 1:, :]
        # query_t [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_t.size()
        p, t = pt // T, T
        hidden_states = query_t.reshape(b * p, t, m)

        #init_cls_token, hidden_states = self.process.before(hidden_states)

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
        )

        res_temporal = self.dropout_layer(
            self.proj_drop(hidden_states.contiguous()))
        res_temporal = self.temporal_fc(res_temporal)
        # res_temporal [batch_size, num_patches * num_frames, embed_dims]
        hidden_states = res_temporal.reshape(b, p * T, m)
        #hidden_states = self.process.after(hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = torch.cat((init_cls_token, hidden_states), 1)
        outputs = hidden_states

        return outputs

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501

    Args:
        drop_prob (float): Probability of the path to be zeroed. Default: 0.1
    """

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)
    
def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py  # noqa: E501
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # handle tensors with different dimensions, not just 4D tensors.
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=f'cuda:{x.device}' if isinstance(x.device, int) else x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output
