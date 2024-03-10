from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.utils import logging

from docllm.modules.llama.config import DocLLMLlamaConfig
from docllm.modules.llama.decoder_layer import DocLLMLlamaDecoderLayer
from docllm.modules.llama.pretrained_model import DocLLMLlamaPreTrainedModel
from docllm.modules.part_freezable_embedding import PartFreezableEmbedding

logger = logging.get_logger(__name__)


@dataclass
class DocLLMModelOutputWithPast(ModelOutput):
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`transformers.Cache)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        spatial_past_key_value (`transformers.Cache)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    spatial_past_key_value: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class LlamaDocLLM(DocLLMLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DocLLMLlamaDecoderLayer`]

    Args:
        config: DocLLMLlamaConfig
    """

    def __init__(self, config: DocLLMLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = PartFreezableEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=self.padding_idx,
            num_additional_tokens=config.additional_training_vocab_size,
        )
        self.embed_spatial = nn.Linear(4, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [DocLLMLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Register a causal mask to separate causal and padding mask creation. Merging happens in the attention class.
        # NOTE: This is not friendly with TorchScript, ONNX, ExportedProgram serialization for very large `max_position_embeddings`.
        causal_mask = torch.full(
            (config.max_position_embeddings, config.max_position_embeddings), fill_value=True, dtype=torch.bool
        )
        self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def set_freeze_llama_layers(self, freeze: bool):
        self.embed_tokens.set_freeze_original_embeddings(freeze)
        for layer in self.layers:
            layer.set_freeze_llama_layers(freeze)
        self.norm.requires_grad_(not freeze)

    @torch.no_grad()
    def fuse_additional_embeddings(self):
        self.embed_tokens.fuse_additional_embeddings()
        self.config.vocab_size = self.embed_tokens.num_embeddings
        self.config.additional_training_vocab_size = 0

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_coordinates: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        spatial_past_key_value: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        assert input_ids.device == input_coordinates.device

        if use_cache and not isinstance(past_key_values, Cache):
            raise ValueError("Legacy cache is not supported. past_key_values must be an instance of transformers.Cache")

        if use_cache and not isinstance(spatial_past_key_value, Cache):
            raise ValueError(
                "Legacy cache is not supported. spatial_past_key_value must be an instance of transformers.Cache"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        coordinate_embeds = self.embed_spatial(input_coordinates)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        next_spacial_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    coordinate_embeds,
                    position_ids,
                    causal_mask,
                    past_key_values,
                    spatial_past_key_value,
                    output_attentions,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    coordinate_embeds,
                    position_ids=position_ids,
                    attention_mask=causal_mask,
                    past_key_value=past_key_values,
                    spatial_past_key_value=spatial_past_key_value,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2]
                next_spacial_cache = layer_outputs[3]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not use_cache:
            next_decoder_cache = None
            next_spacial_cache = None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_decoder_cache, next_spacial_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return DocLLMModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            spatial_past_key_value=next_spacial_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype
        device = input_tensor.device

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = torch.full((2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]), fill_value=1)
            self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)

        # We use the current dtype to avoid any overflows
        min_dtype = torch.finfo(dtype).min
        causal_mask = self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) * min_dtype

        causal_mask = causal_mask.to(dtype=dtype, device=device)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if self.config._attn_implementation == "sdpa" and attention_mask is not None:
            # TODO: For dynamo, rather use a check on fullgraph=True once this is possible (https://github.com/pytorch/pytorch/pull/120400).
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                # Attend to all tokens in masked rows from the causal_mask, for example the relevant first rows when
                # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
                # Details: https://github.com/pytorch/pytorch/issues/110213
                causal_mask = causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True)).to(dtype)

        return causal_mask
