import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

from docllm.llama.config import DocLLMLlamaConfig


class DocLLMAttention(LlamaAttention):
    def __init__(self, config: DocLLMLlamaConfig, layer_idx: Optional[int] = None):
        if config.pretraining_tp > 1:
            raise NotImplementedError()
        super().__init__(config, layer_idx)
        self.spatial_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.spatial_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # self.spatial_v_proj = nn.Linear(
        #     self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        # )
        self._lambda_ts = config.lambda_ts
        self._lambda_st = config.lambda_st
        self._lambda_ss = config.lambda_ss

    def set_freeze_llama_layers(self, freeze: bool):
        self.q_proj.weight.requires_grad_(not freeze)
        self.k_proj.weight.requires_grad_(not freeze)
        self.v_proj.weight.requires_grad_(not freeze)
        self.o_proj.weight.requires_grad_(not freeze)

    def forward(
        self,
        hidden_states: Tensor,
        bounding_box_embeddings: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: LongTensor | None = None,
        past_key_value: Cache | None = None,
        spatial_past_key_value: Cache | None = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size()

        (
            query_states,
            key_states,
            value_states,
            past_key_value,
            kv_seq_len,
        ) = self._compute_q_k_v(
            hidden_states,
            position_ids,
            past_key_value,
            self.q_proj,
            self.k_proj,
            self.v_proj,
        )

        (
            spatial_query_states,
            spatial_key_states,
            _,
            spatial_past_key_value,
            kv_seq_len,
        ) = self._compute_q_k_v(
            bounding_box_embeddings,
            position_ids,
            spatial_past_key_value,
            self.spatial_q_proj,
            self.spatial_k_proj,
            None,
        )

        attn_weights_tt = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        attn_weights_ts = torch.matmul(query_states, spatial_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        attn_weights_st = torch.matmul(spatial_query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        attn_weights_ss = torch.matmul(spatial_query_states, spatial_key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if (
            attn_weights_tt.size() != (bsz, self.num_heads, q_len, kv_seq_len)
            or attn_weights_ts.size() != (bsz, self.num_heads, q_len, kv_seq_len)
            or attn_weights_st.size() != (bsz, self.num_heads, q_len, kv_seq_len)
            or attn_weights_ss.size() != (bsz, self.num_heads, q_len, kv_seq_len)
        ):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights_tt.size()=} {attn_weights_ts.size()=} {attn_weights_st.size()=} {attn_weights_ss.size()=}"
            )

        attn_weights = (
            attn_weights_tt
            + self._lambda_ts * attn_weights_ts
            + self._lambda_st * attn_weights_st
            + self._lambda_ss * attn_weights_ss
        )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, spatial_past_key_value

    def _compute_q_k_v(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Optional[Cache],
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: Optional[nn.Linear],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Cache], int]:
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

        kv_seq_len = key_states.shape[-2]
        past_seen_tokens = 0
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            past_seen_tokens = past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            kv_seq_len += past_seen_tokens

        new_cache_positions = torch.arange(past_seen_tokens, past_seen_tokens + q_len, device=key_states.device)
        position_ids = new_cache_positions.unsqueeze(0) if position_ids is None else position_ids

        if v_proj is not None:  # FIXME do we need pos embeddings for spatial modality?
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = (
                {} if v_proj is None else {"sin": sin, "cos": cos, "position_ids": new_cache_positions}
            )  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        if v_proj is not None:
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        return query_states, key_states, value_states, past_key_value, kv_seq_len
