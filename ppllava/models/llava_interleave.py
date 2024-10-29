from transformers import CLIPVisionModel, PretrainedConfig, GenerationMixin
from transformers.models.llava_next.modeling_llava_next import LlavaNextCausalLMOutputWithPast, \
    LlavaNextPreTrainedModel, LLAVA_NEXT_INPUTS_DOCSTRING, LlavaNextMultiModalProjector, LlavaNextConfig

from transformers.models.llava_onevision.modeling_llava_onevision import  LLAVA_ONEVISION_START_DOCSTRING, LlavaOnevisionConfig 

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.cache_utils import Cache
_CONFIG_FOR_DOC = "LlavaNextVidConfig"

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ppllava.common.registry import registry
from ppllava.models.base_model import BaseModel

import os
import re
import gc
from ppllava.models.llava_projector import PllavaMultiModalProjector, LLaVASiglip3DMultiModalProjector,\
    PllavaNPU3DMultiModalProjector, LLaVACLIP3DMultiModalProjector
import einops
from ppllava.models.clip_btadapter import CLIPVisionModel_BTAdapter
from ppllava.models.clip import CLIPModelwithVideo
from ppllava.models.llava_interleave_meta import LlavaInterleaveMetaForConditionalGeneration
from safetensors import safe_open

from peft import (
    LoraConfig,
    get_peft_model,
)

class PPLLaVAConfig(PretrainedConfig):
    pooling = 'avp'
    pllava_pooling_shape=None
    pooling_kernel=None
    pooling_stride=None
    image_pooling_kernel=None
    image_pooling_stride=None
    frame_shape=(24,24)
    clip_weight = None
    clip_post_pretrain = None
    btadapter = False
    btadapter_depth = 4
    max_T = 128
    extend_clip = False
    pooling_temp = 0.01

class LlavaNextVidConfig(LlavaNextConfig, PPLLaVAConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pad_token_id = 0

class LlavaOnevisionVidConfig(LlavaOnevisionConfig, PPLLaVAConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pad_token_id = 151643
        self.valuew_plus = False

@registry.register_model("llava_interleave")
class LlavaInterleaveForConditionalGeneration(LlavaNextPreTrainedModel, BaseModel, GenerationMixin, LlavaInterleaveMetaForConditionalGeneration):
    PRETRAINED_MODEL_CONFIG_DICT = {}
    def __init__(self, config: LlavaNextVidConfig):
        super().__init__(config)
        if config.btadapter:
            config.vision_config.depth = config.btadapter_depth
            config.vision_config.max_T = config.max_T
            self.vision_tower = CLIPVisionModel_BTAdapter._from_config(config.vision_config)
        else:
            self.vision_tower = AutoModel.from_config(
                config.vision_config, attn_implementation=config._attn_implementation
            )

        self.pooling = config.pooling 
        if config.pooling == 'avp' or config.pooling == 'none':
            self.multi_modal_projector = LlavaNextMultiModalProjector(config)
        elif config.pooling == 'pllava':
            self.multi_modal_projector = PllavaMultiModalProjector(config)
        elif config.pooling == 'pllava_npu':
            self.multi_modal_projector = PllavaNPU3DMultiModalProjector(config)
        elif config.pooling == 'ppllava':
            if self.config.qwen:
                self.multi_modal_projector = LLaVASiglip3DMultiModalProjector(config)
            else:
                self.multi_modal_projector = LLaVACLIP3DMultiModalProjector(config)
         
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size, dtype=self.dtype))

        self.config.hidden_size = config.text_config.hidden_size
        #self.post_init()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder
    def get_decoder(self):
        return self.language_model.get_decoder()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.tie_weights
    def tie_weights(self):
        return self.language_model.tie_weights()

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.resize_token_embeddings
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def encode_visual(self, pixel_value, index, image_sizes=None, media_types=None, clip_ids=None, clip_mask=None, output_visual_attention=False):
        media_type = 'video' if media_types is None else media_types[index]
        image_size = None if image_sizes is None else image_sizes[index]
        clip_ids = None if clip_ids is None else clip_ids[index].unsqueeze(0)
        clip_mask = None if clip_mask is None else clip_mask[index].unsqueeze(0)
        if media_type=='image':
            image_features = self.encode_image(pixel_value, image_size, clip_ids, clip_mask)
            return image_features
        elif media_type=='video':
            image_features = self.encode_video(pixel_value, image_size, clip_ids, clip_mask, output_visual_attention=output_visual_attention)
            return image_features
        elif media_type=='multi-image':
            num_images = pixel_value.size(0)
            image_features = self.encode_video(pixel_value, image_size, clip_ids, clip_mask, multi_image=True)
            image_features = einops.rearrange(image_features, 'B (T L) D -> (B T) L D', T=num_images)
            return image_features
        
    @add_start_docstrings_to_model_forward(LLAVA_NEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=LlavaNextCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        clip_ids: Optional[torch.Tensor] = None,
        clip_mask: Optional[torch.Tensor] = None,
        dpo_forward: Optional[bool] = False,
        media_types: Optional[List[str]] = None,
        output_visual_attention: Optional[bool] = False,
    ) -> Union[Tuple, LlavaNextCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        >>> model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        >>> prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot (...)"
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        self.vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        if labels is not None:
            labels = labels[:,:input_ids.size(1)]
        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if output_visual_attention:
                return self.encode_visual(pixel_values[0], 0, image_sizes, media_types, clip_ids, clip_mask, output_visual_attention=True)
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_features = []
                for i, pixel_value in enumerate(pixel_values):
                    image_features.append(self.encode_visual(pixel_value, i, image_sizes, media_types, clip_ids, clip_mask))
      
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(f'cuda:{torch.long}' if isinstance(torch.long, int) else torch.long)

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=f'cuda:{attention_mask.device}' if isinstance(attention_mask.device, int) else attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]
        if dpo_forward:
            return logits, labels
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(f'cuda:{logits.device}' if isinstance(logits.device, int) else logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(f'cuda:{labels.device}' if isinstance(labels.device, int) else labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(f'cuda:{shift_logits.device}' if isinstance(shift_logits.device, int) else shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        clip_ids = kwargs.get("clip_ids", None)
        clip_mask = kwargs.get("clip_mask", None)
        media_types = kwargs.get("media_types", None)
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_sizes": image_sizes,
                "clip_ids": clip_ids,
                "clip_mask": clip_mask,
                "media_types": media_types,
            }
        )
        return model_inputs
        
    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration._reorder_cache
    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)

    def load_pretrained_weight(self, ckpt_path):
        if ckpt_path is None:
            return 0

        if os.path.isdir(ckpt_path):
            ckpt = self.get_state_dict(ckpt_path)
            msg = self.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = self.load_state_dict(ckpt, strict=False)

    @classmethod
    def get_state_dict(self, path, prefix='(model|non_lora_trainable)'):
        pattern = re.compile(r'^(model|non_lora_trainable).*?(\.safetensors|\.bin)$')
        matching_files = [filename for filename in os.listdir(path) if pattern.match(filename)]

        model_state_dict = {}
        for model_path in matching_files:
            if model_path.endswith('safetensors'):
                with safe_open(os.path.join(path,model_path), framework="pt", device='cpu') as f:
                    for k in f.keys():
                        model_state_dict[k] = f.get_tensor(k)
            elif model_path.endswith('bin'):
                partial_state_dict = torch.load(os.path.join(path,model_path), map_location=torch.device('cpu'))
                model_state_dict.update(partial_state_dict)
        return model_state_dict

    @classmethod
    def from_config(cls, cfg):
        llama_model = cfg.get("llama_model")
        pretrained_state_dict = cls.get_state_dict(llama_model)

        qwen = ('qwen' in llama_model)
        config_cls = LlavaOnevisionVidConfig if qwen else LlavaNextVidConfig
        pretrained_cfg = config_cls.from_pretrained(llama_model)
        
        pretrained_cfg.qwen = qwen
        pretrained_cfg.pooling = cfg.get("pooling","avp") 
        pretrained_cfg.clip_weight = cfg.get("clip_weight", None)
        pretrained_cfg.clip_post_pretrain = cfg.get("clip_post_pretrain", None)
        pretrained_cfg.btadapter = cfg.get("btadapter",False)
        pretrained_cfg.btadapter_depth = cfg.get("btadapter_depth",4)
        pretrained_cfg.max_T = cfg.get("max_T",64)
        pretrained_cfg.extend_clip = cfg.get("extend_clip",False)
        pretrained_cfg.pooling_temp = cfg.get("pooling_temp",0.01)
        pretrained_cfg.valuew_plus = cfg.get("valuew_plus",False)
        
        if cfg.get("pllava_pooling_shape",None):
            pretrained_cfg.pllava_pooling_shape = eval(cfg.get("pllava_pooling_shape"))
        if cfg.get("pooling_kernel",None):
            pretrained_cfg.pooling_kernel = eval(cfg.get("pooling_kernel"))
        if cfg.get("pooling_stride",None):
            pretrained_cfg.pooling_stride = eval(cfg.get("pooling_stride"))
        if cfg.get("image_pooling_kernel",None):
            pretrained_cfg.image_pooling_kernel = eval(cfg.get("image_pooling_kernel"))
        if cfg.get("image_pooling_stride",None):
            pretrained_cfg.image_pooling_stride = eval(cfg.get("image_pooling_stride"))
        if cfg.get("frame_shape",None):
            pretrained_cfg.frame_shape = eval(cfg.get("frame_shape"))
        
        model = cls(pretrained_cfg).to(f'cuda:{torch.float16}' if isinstance(torch.float16, int) else torch.float16)
        msg = model.load_state_dict(pretrained_state_dict, strict=False)
        del pretrained_state_dict
        gc.collect()

        if cfg.get("btadapter",False):
            model.vision_tower.vision_model.init_weights()

        if cfg.get("use_lora", False):
            kwargs = {}
            
            kwargs.update({"target_modules": ["q_proj", "v_proj"]})
            peft_config = LoraConfig(
                task_type="CAUSAL_LM", inference_mode=False, 
                r=cfg.get("lora_r"), lora_alpha=cfg.get("lora_alpha"), lora_dropout=cfg.get("lora_dropout"),
                **kwargs
            )
            model.language_model = get_peft_model(model.language_model, peft_config)
            model.language_model.print_trainable_parameters()
    
        if cfg.get("freeze_LLM",True):
            for n,p in model.language_model.named_parameters():
                if 'lora' not in n:
                    p.requires_grad = False
        
        if cfg.get("freeze_vision_tower",True):
            for n,p in model.vision_tower.named_parameters():
                if 'btadapter' not in n:
                    p.requires_grad = False   
        
        if cfg.get("onlyLLM", False):
            for n,p in model.named_parameters():
                if 'language_model' not in n:
                    p.requires_grad = False
            
        if cfg.get("gradient_checkpointing",False):
            model.gradient_checkpointing_enable()
        
        ckpt_path = cfg.get("ckpt", "") 
        if ckpt_path:
            #if os.path.isdir(ckpt_path) and not cfg.get("use_lora", False):
            #    model = model.from_pretrained(ckpt_path)
            #elif cfg.get("use_lora", False):
            if os.path.isdir(ckpt_path):
                ckpt = cls.get_state_dict(ckpt_path)
                msg = model.load_state_dict(ckpt, strict=False)
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                msg = model.load_state_dict(ckpt, strict=False)
        #[n for n,p in model.named_parameters() if p.requires_grad==True]
        return model

        