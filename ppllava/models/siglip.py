
import torch 
import os
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Any, List, Optional, Tuple, Union
from transformers import SiglipModel, SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipMLP

from ppllava.models.base_model import BaseModel
from ppllava.models.clip import ExtendTextEmbeddings
from ppllava.common.registry import registry

class SiglipVideoConfig(SiglipConfig):
    def __init__(self, btadapter=False, btadapter_depth=4, max_T=128, **kwargs):
        super().__init__(**kwargs)
        self.btadapter = btadapter
        self.btadapter_depth = btadapter_depth
        self.max_T = max_T

@registry.register_model("siglip__video")
class SiglipModelwithVideo(SiglipModel, BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {}
    def __init__(self, config: SiglipVideoConfig):
        super().__init__(config)

    def convert_text_position(self):
        position_embedding_pre = self.text_model.embeddings.position_embedding.weight.data.type(self.dtype)
            
        length, dim = position_embedding_pre.shape
        keep_len = 20
        posisitonal_embedding_new = torch.zeros([4*length-3*keep_len, dim], dtype=self.dtype)
        for i in range(keep_len):
            posisitonal_embedding_new[i] = position_embedding_pre[i]
        for i in range(length-1-keep_len):
            posisitonal_embedding_new[4*i + keep_len] = position_embedding_pre[i + keep_len]
            posisitonal_embedding_new[4*i + 1 + keep_len] = 3*position_embedding_pre[i + keep_len]/4 + 1*position_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 2+keep_len] = 2*position_embedding_pre[i+keep_len]/4 + 2*position_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 3+keep_len] = 1*position_embedding_pre[i+keep_len]/4 + 3*position_embedding_pre[i+1+keep_len]/4
    
        posisitonal_embedding_new[4*length -3*keep_len - 4] = position_embedding_pre[length-1] + 0*(position_embedding_pre[length-1] - position_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 3] = position_embedding_pre[length-1] + 1*(position_embedding_pre[length-1] - position_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 2] = position_embedding_pre[length-1] + 2*(position_embedding_pre[length-1] - position_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 1] = position_embedding_pre[length-1] + 3*(position_embedding_pre[length-1] - position_embedding_pre[length-2])/4
                
        position_embedding_res = posisitonal_embedding_new.clone()
        
        extend_text_embedding = ExtendTextEmbeddings(self.config.text_config)
        extend_text_embedding.position_embedding = nn.Parameter(posisitonal_embedding_new, requires_grad=False)
        extend_text_embedding.position_embedding_res = nn.Parameter(position_embedding_res, requires_grad=True)
        self.text_model.embeddings = extend_text_embedding

class SiglipPoolingHead_PatchCLS(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        query, attn_weights, value = self.attention(probe, hidden_state, hidden_state)

        hidden_state = value
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state, attn_weights

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # 将 Q、K、V 的线性变换参数合并
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, bsz, embed_dim = query.size()
        
        assert embed_dim == self.embed_dim

        q, k, value_ = self._in_proj_qkv(query, key, value)

        q = q.reshape((tgt_len, bsz * self.num_heads, self.head_dim)).transpose(0, 1)
        k = k.reshape((-1, bsz * self.num_heads, self.head_dim)).transpose(0, 1)
        v = value_.reshape((-1, bsz * self.num_heads, self.head_dim)).transpose(0, 1)

        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, attn_mask, key_padding_mask)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_weights.contiguous().view(bsz, self.num_heads, -1), value_.transpose(0, 1)

    def _in_proj_qkv(self, query, key, value):
        q = F.linear(query, self.in_proj_weight[:self.embed_dim], self.in_proj_bias[:self.embed_dim])
        k = F.linear(key, self.in_proj_weight[self.embed_dim:2 * self.embed_dim], self.in_proj_bias[self.embed_dim:2 * self.embed_dim])
        v = F.linear(value, self.in_proj_weight[2 * self.embed_dim:], self.in_proj_bias[2 * self.embed_dim:])
        return q, k, v

    def scaled_dot_product_attention(self, q, k, v, attn_mask=None, key_padding_mask=None):
        q_scaled = q / self.head_dim ** 0.5
        attn_weights = torch.bmm(q_scaled, k.transpose(-2, -1)) 

        if attn_mask is not None:
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights
