import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.utils import logging

from docllm.modules.llama.config import DocLLMLlamaConfig, PositionalEmbeddingMode

logger = logging.get_logger(__name__)


class DocLLMAttention(nn.Module):
    def __init__(self, config: DocLLMLlamaConfig, layer_idx: Optional[int] = None):
        if config.pretraining_tp > 1:
            raise NotImplementedError()
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.layer_idx = layer_idx
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.spatial_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.spatial_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self._lambda_ts = config.lambda_ts
        self._lambda_st = config.lambda_st
        self._lambda_ss = config.lambda_ss
        self._positional_embedding_mode = config.positional_embedding_mode
        if config._attn_implementation == "sdpa":
            self._attn_fct = self._sdpa_attention
        elif config._attn_implementation == "eager":
            self._attn_fct = self._eager_attention
        else:
            raise ValueError(f"Unknown attention implementation: {config._attn_implementation}")

    def set_freeze_llama_layers(self, freeze: bool):
        self.q_proj.weight.requires_grad_(not freeze)
        self.k_proj.weight.requires_grad_(not freeze)
        self.v_proj.weight.requires_grad_(not freeze)
        self.o_proj.weight.requires_grad_(not freeze)

    @torch.no_grad()
    def init_additional_weights(self, init_func: Callable[[torch.Tensor], None] = torch.nn.init.xavier_normal_):
        init_func(self.spatial_q_proj.weight)
        init_func(self.spatial_k_proj.weight)

    def forward(
        self,
        hidden_states: Tensor,
        bounding_box_embeddings: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Cache | None = None,
        spatial_past_key_value: Cache | None = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[Cache]]:
        query_states, key_states, value_states = self._compute_q_k_v(
            hidden_states,
            bounding_box_embeddings,
            past_key_value,
            spatial_past_key_value,
            cache_position,
            position_embeddings,
        )

        attn_weights, attn_output = self._attn_fct(
            query_states,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
        )
        attn_output = attn_output.transpose(1, 2)  # .contiguous()
        bsz, q_len, _ = hidden_states.size()
        if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value, spatial_past_key_value

    def _compute_q_k_v(
        self,
        hidden_states: torch.Tensor,
        bounding_box_embeddings: torch.Tensor,
        past_key_value: Optional[Cache],
        spatial_past_key_value: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states, key_states, value_states = self._compute_q_k_v_for_projections(
            hidden_states=hidden_states,
            q_proj=self.q_proj,
            k_proj=self.k_proj,
            v_proj=self.v_proj,
            past_key_value=past_key_value,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )

        spatial_query_states, spatial_key_states, _ = self._compute_q_k_v_for_projections(
            hidden_states=bounding_box_embeddings,
            q_proj=self.spatial_q_proj,
            k_proj=self.spatial_k_proj,
            v_proj=None,
            past_key_value=spatial_past_key_value,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )

        query_states = query_states + spatial_query_states
        key_states = key_states + spatial_key_states

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return query_states, key_states, value_states

    def _compute_q_k_v_for_projections(
        self,
        hidden_states: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: Optional[nn.Linear],
        past_key_value: Optional[Cache],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        if v_proj is not None:
            value_states = v_proj(hidden_states)
        else:
            value_states = None

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if v_proj is not None:
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if v_proj is not None or self._positional_embedding_mode != PositionalEmbeddingMode.NONE:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            cos = sin = None

        if past_key_value is not None:
            cache_kwargs = (
                {"cache_position": cache_position}
                if cos is not None and sin is not None
                else {"sin": sin, "cos": cos, "cache_position": cache_position}
            )  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        return query_states, key_states, value_states

    def _eager_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if not output_attentions:
            attn_weights = None

        return attn_weights, attn_output

    def _sdpa_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if output_attentions:
            logger.warning_once(
                "Setting `output_attentions=True` is not supported for `sdpa` attention. Ignoring the argument."
            )
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query_states.contiguous(),
            key=key_states.contiguous(),
            value=value_states.contiguous(),
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout,
            scale=self.scaling,
            is_causal=True,
        )
        return None, attn_output
