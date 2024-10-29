from abc import ABC, abstractmethod
import torch

import einops

from transformers.models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image

class LlavaInterleaveMetaForConditionalGeneration(ABC):
    def encode_image(self, pixel_value, image_size, clip_ids=None, clip_mask=None):
        batch_size, num_patches, num_channels, height, width = pixel_value.shape
        reshaped_pixel_values = pixel_value.view(batch_size * num_patches, num_channels, height, width)
        image_features = self.vision_tower(reshaped_pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        
        if self.pooling=='ppllava':
            image_feature = self.multi_modal_projector(selected_image_feature,
                                                    clip_ids, clip_mask, selected_image_feature.size(0), video=False)
            image_feature = einops.rearrange(image_feature, 'B (T L) D -> (B T) L D', T=selected_image_feature.size(0))
        else:
            image_feature = self.multi_modal_projector(selected_image_feature)

        height = width = int(image_feature.shape[1]**0.5)
        
        if image_feature.shape[0] > 1:
            base_image_feature = image_feature[0]
            image_feature = image_feature[1:]

            num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                image_size,
                self.config.image_grid_pinpoints,
                self.config.vision_config.image_size,
            )
            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
            image_feature = unpad_image(image_feature, image_size)
            image_feature = torch.cat(
                (
                    image_feature,
                    self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1),
                ),
                dim=-1,
            )
            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
        else:
            image_feature = image_feature[0]
            image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)
        
        return image_feature.unsqueeze(0)
        
    def encode_video(self, pixel_value, image_size, clip_ids=None, clip_mask=None, multi_image=False, output_visual_attention=False): 
        num_frames, num_patches, num_channels, height, width = pixel_value.shape
        if self.config.btadapter:
            reshaped_pixel_values = pixel_value.view(num_patches, num_frames , num_channels, height, width)
        else:
            reshaped_pixel_values = pixel_value.view(num_frames * num_patches, num_channels, height, width)
        image_features = self.vision_tower(reshaped_pixel_values, output_hidden_states=True)
        selected_image_feature = image_features.hidden_states[self.vision_feature_layer]
        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        
        if self.pooling == 'avp':
            image_features = self.multi_modal_projector(selected_image_feature) 
            image_features = einops.rearrange(image_features, '(B T) L D -> B T L D', T=num_frames)
            image_features = image_features.mean(1)
        elif self.pooling == 'none':
            image_features = self.multi_modal_projector(selected_image_feature) 
            image_features = einops.rearrange(image_features, '(B T) L D -> B (T L) D', T=num_frames)
        elif self.pooling == 'pllava' or self.pooling == 'pllava_npu':
            image_features = self.multi_modal_projector(selected_image_feature,
                                                    batch_size=1, num_frames=num_frames)
        elif self.pooling == 'ppllava':
            image_features = self.multi_modal_projector(selected_image_feature,
                                                    clip_ids, clip_mask, num_frames, video=(not multi_image),output_visual_attention=output_visual_attention)
        return image_features

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        batch_size, sequence_length = input_ids.shape
        embed_dim = image_features[0].size(-1)
        num_images = len(image_features)
        num_image_patches = torch.tensor([feature.size(0) if len(feature.size())==2 else feature.size(1) for feature in image_features]).to(inputs_embeds.device)
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens * (num_image_patches - 1)).max() + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches[:,None] - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=f'cuda:{inputs_embeds.device}' if isinstance(inputs_embeds.device, int) else inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=f'cuda:{inputs_embeds.device}' if isinstance(inputs_embeds.device, int) else inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=f'cuda:{input_ids.device}' if isinstance(input_ids.device, int) else input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device),
            non_image_indices.to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device),
            text_to_overwrite.to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device),
        )
        attention_mask = attention_mask.to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        if left_padding:
            image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device)
        else:
            image_to_overwrite &= (torch.arange(max_embed_dim).expand(batch_size, -1).to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device) <= new_token_positions[:, -1].unsqueeze(1).expand(-1, max_embed_dim))

        num_image_features = sum([feature.shape[:-1].numel() for feature in image_features])
        if image_to_overwrite.sum() != num_image_features:
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        for i, image_feature in enumerate(image_features):
            final_embedding[i, image_to_overwrite[i]] = image_feature.contiguous().reshape(-1, embed_dim).to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device)

        #final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(f'cuda:{target_device}' if isinstance(target_device, int) else target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    