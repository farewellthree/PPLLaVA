
import torch 
import os
import torch.nn as nn
import einops
from typing import Any, List, Optional, Tuple, Union
from transformers import CLIPModel, CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.modeling_clip import CLIPOutput, clip_loss, CLIP_INPUTS_DOCSTRING,\
    CLIPTextTransformer, CLIPVisionTransformer
from mmengine.dist import all_gather, get_rank

from ppllava.models.clip_btadapter import CLIPVisionTransformer_BTAdapter
from ppllava.models.base_model import BaseModel
from ppllava.common.registry import registry
from safetensors.torch import load_file

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> Tuple[List]:
        ctx.save_for_backward(input)
        output = all_gather(input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank()]
        return grad_out

class CLIPVideoConfig(CLIPConfig):
    def __init__(self, wdim='T', btadapter=False, btadapter_depth=4, max_T=128, **kwargs):
        super().__init__(**kwargs)
        self.wdim = wdim
        self.btadapter = btadapter
        self.btadapter_depth = btadapter_depth
        self.max_T = max_T

@registry.register_model("clip_video")
class CLIPModelwithVideo(CLIPModel, BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {}
    def __init__(self, config: CLIPVideoConfig):
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.wdim = config.wdim
        self.btadapter = config.btadapter
        self.config.hidden_size = self.text_embed_dim

        self.text_model = CLIPTextTransformer(text_config)
        if self.btadapter:
            vision_config.depth = config.btadapter_depth
            vision_config.max_T = config.max_T
            self.vision_model = CLIPVisionTransformer_BTAdapter(vision_config)
        else:
            self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

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

    @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=CLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        B, T, D, W, H = pixel_values.shape
        pixel_values = einops.rearrange(pixel_values, 'B T D W H -> (B T) D W H')
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        image_embeds = vision_outputs.hidden_states[-2][:, 1:]
        image_embeds = self.visual_projection(image_embeds)
        image_cls = self.visual_projection(vision_outputs.pooler_output)
        if self.training and torch.cuda.is_available():
            image_embeds = torch.cat(GatherLayer.apply(image_embeds), dim=0)
            image_cls = torch.cat(GatherLayer.apply(image_cls), dim=0)
            text_embeds = torch.cat(GatherLayer.apply(text_embeds), dim=0)
        if self.wdim == 'T':
            image_embeds = einops.rearrange(image_embeds, '(B T) L D -> B T L D', T=T)
            image_embeds = image_embeds.mean(dim=2)
        elif self.wdim == 'ST':
            image_embeds = einops.rearrange(image_embeds, '(B T) L D -> B (T L) D', T=T)
        elif self.wdim == 'cls':
            image_cls = einops.rearrange(image_cls, '(B T) D -> B T D', T=T)
            image_embeds = image_cls.mean(dim=1)
        elif self.wdim == 'clsT':
            image_embeds = einops.rearrange(image_cls, '(B T) D -> B T D', T=T)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        if self.wdim == 'cls':
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        else:
            retrieve_logits = torch.einsum('ad,bvd->abv', [text_embeds, image_embeds])
            tv_softmax = torch.softmax(retrieve_logits*logit_scale, dim=-1) 
            logits_per_text = torch.sum(retrieve_logits * tv_softmax, dim=-1) * logit_scale

        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
    
    @classmethod
    def from_config(cls, cfg):
        clip_model = cfg.get("clip_model")
        pretrain_model = CLIPModel.from_pretrained(clip_model, torch_dtype=torch.float16) 
        pretrained_cfg = CLIPVideoConfig.from_pretrained(clip_model)

        pretrained_cfg.wdim = cfg.get("wdim","T")
        pretrained_cfg.btadapter = cfg.get("btadapter",False)
        pretrained_cfg.btadapter_depth = cfg.get("btadapter_depth",4)
        pretrained_cfg.max_T = cfg.get("btadapter_depth",64)

        model = cls(pretrained_cfg).to(f'cuda:{torch.float16}' if isinstance(torch.float16, int) else torch.float16)
        model.logit_scale.requires_grad = False 
        model.load_state_dict(pretrain_model.state_dict(), strict=False)

        if cfg.get("btadapter",False):
            model.vision_model.init_weights()

        if cfg.get("freeze_vision",True):
            for n, p in model.vision_model.named_parameters():
                if 'btadapter' not in n:
                    p.requires_grad = False

        if cfg.get("freeze_text",False):
            for p in model.text_model.parameters():
                p.requires_grad = False    
        
        if cfg.get("use_grad_checkpoint",True):
            model.gradient_checkpointing_enable()
        
        if cfg.get("clip_text_extend",False):
            model.convert_text_position()

        ckpt_path = cfg.get("clip_post_pretrain", "")
        if ckpt_path:
            if os.path.isdir(ckpt_path):
                model = model.from_pretrained(ckpt_path)
            else:
                if ckpt_path.endswith('safetensors'):
                    ckpt = load_file(ckpt_path)
                else:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                msg = model.load_state_dict(ckpt, strict=False)

        return model

class ExtendTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)

        self.position_embedding = nn.Parameter(torch.empty(248, embed_dim))
        self.position_embedding_res = nn.Parameter(torch.empty(248, embed_dim))

        self.mask1 = torch.zeros([248, 1])
        self.mask1[:20, :] = 1
        self.mask2 = torch.zeros([248, 1])
        self.mask2[20:, :] = 1
        

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        x = inputs_embeds
        input_len = inputs_embeds.size(1)
        position_embeddings = (self.position_embedding.to(f'cuda:{x.device}' if isinstance(x.device, int) else x.device)[:input_len] * self.mask1.to(f'cuda:{x.device}' if isinstance(x.device, int) else x.device)[:input_len]).type(x.dtype).to(f'cuda:{x.device}' if isinstance(x.device, int) else x.device)\
              + (self.position_embedding_res.to(f'cuda:{x.device}' if isinstance(x.device, int) else x.device)[:input_len] * self.mask2.to(f'cuda:{x.device}' if isinstance(x.device, int) else x.device)[:input_len]).type(x.dtype).to(f'cuda:{x.device}' if isinstance(x.device, int) else x.device)
        embeddings = inputs_embeds + position_embeddings

        return embeddings