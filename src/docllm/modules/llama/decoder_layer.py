import warnings
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

from docllm.modules.llama.attention import DocLLMAttention
from docllm.modules.llama.config import DocLLMLlamaConfig

LLAMA_ATTENTION_CLASSES = {
    "eager": DocLLMAttention,
    # "flash_attention_2": DocLLMFlashAttention2,
    # "sdpa": DocLLMSdpaAttention,
}


class DocLLMLlamaDecoderLayer(nn.Module):
    def __init__(self, config: DocLLMLlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def set_freeze_llama_layers(self, freeze: bool):
        self.self_attn.set_freeze_llama_layers(freeze)
        self.mlp.requires_grad_(not freeze)
        self.input_layernorm.requires_grad_(not freeze)
        self.post_attention_layernorm.requires_grad_(not freeze)

    @torch.no_grad()
    def init_additional_weights(self, init_func: Callable[[torch.Tensor], None] = torch.nn.init.xavier_normal_):
        self.self_attn.init_additional_weights(init_func)

    def forward(
        self,
        hidden_states: torch.Tensor,
        bounding_box_embeddings: Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        spatial_past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Cache], Optional[Cache]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            bounding_box_embeddings (`torch.FloatTensor`): bounding box embeddings of shape `(batch, seq_len, embed_dim)`
            position_ids (`torch.LongTensor`): position ids of shape `(batch, seq_len)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            past_key_value (`Cache`, *optional*): cached past key and value projection states
            spatial_past_key_value (`Cache`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        (
            hidden_states,
            self_attn_weights,
            present_key_value,
            spatial_present_key_value,
        ) = self.self_attn(
            hidden_states=hidden_states,
            bounding_box_embeddings=bounding_box_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            spatial_past_key_value=spatial_past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if not output_attentions:
            self_attn_weights = None

        return hidden_states, self_attn_weights, present_key_value, spatial_present_key_value
