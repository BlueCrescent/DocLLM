import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

from docllm.modules.llama.config import DocLLMLlamaConfig, PositionalEmbeddingMode


class DocLLMAttention(LlamaAttention):
    def __init__(self, config: DocLLMLlamaConfig, layer_idx: Optional[int] = None):
        if config.pretraining_tp > 1:
            raise NotImplementedError()
        super().__init__(config, layer_idx)
        self.spatial_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.spatial_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self._lambda_ts = config.lambda_ts
        self._lambda_st = config.lambda_st
        self._lambda_ss = config.lambda_ss
        self._positional_embedding_mode = config.positional_embedding_mode

    def set_freeze_llama_layers(self, freeze: bool):
        self.q_proj.weight.requires_grad_(not freeze)
        self.k_proj.weight.requires_grad_(not freeze)
        self.v_proj.weight.requires_grad_(not freeze)
        self.o_proj.weight.requires_grad_(not freeze)
        self.rotary_emb.requires_grad_(not freeze)

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
        bsz, q_len, _ = hidden_states.size()

        (query_states, key_states, value_states, past_key_value) = self._compute_q_k_v(
            hidden_states, past_key_value, self.q_proj, self.k_proj, self.v_proj, position_embeddings, cache_position
        )

        (spatial_query_states, spatial_key_states, _, spatial_past_key_value) = self._compute_q_k_v(
            bounding_box_embeddings,
            spatial_past_key_value,
            self.spatial_q_proj,
            self.spatial_k_proj,
            None,
            position_embeddings,
            cache_position,
        )

        attn_weights = (
            torch.matmul(
                query_states + spatial_query_states, key_states.transpose(2, 3) + spatial_key_states.transpose(2, 3)
            )
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if not output_attentions:
            attn_weights = None

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value, spatial_past_key_value

    def _compute_q_k_v(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Cache],
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: Optional[nn.Linear],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
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

        if v_proj is not None and self._positional_embedding_mode != PositionalEmbeddingMode.NONE:
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

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        if v_proj is not None:
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        return query_states, key_states, value_states, past_key_value
