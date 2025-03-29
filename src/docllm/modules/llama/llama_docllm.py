from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.utils import logging

from docllm.modules.llama.config import DocLLMLlamaConfig
from docllm.modules.llama.decoder_layer import DocLLMLlamaDecoderLayer
from docllm.modules.llama.pretrained_model import DocLLMLlamaPreTrainedModel
from docllm.modules.part_freezable_embedding import PartFreezableEmbedding
from docllm.modules.spatial_embedder import SpatialEmbedder

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
        self.embed_spatial = SpatialEmbedder(
            hidden_size=config.hidden_size,
            embedding_type=config.embedding_type,
            include_width_height=config.embed_include_width_height,
            max_coord=config.embed_max_coord,
        )
        self.layers = nn.ModuleList(
            [DocLLMLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

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
    def init_additional_weights(self, init_func: Callable[[torch.Tensor], None] = torch.nn.init.xavier_normal_):
        self.embed_tokens.init_additional_embeddings(init_func)
        for layer in self.layers:
            layer.init_additional_weights(init_func)

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

        if not use_cache:
            past_key_values = None
            spatial_past_key_value = None

        if use_cache and past_key_values.get_seq_length() != spatial_past_key_value.get_seq_length():
            raise ValueError(
                "Lengths of cached sequences in past_key_values and spatial_past_key_value do not match. "
                f"{past_key_values.get_seq_length()} != {spatial_past_key_value.get_seq_length()}"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        coordinate_embeds = self.embed_spatial(input_coordinates)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

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
                    causal_mask,
                    past_key_values,
                    spatial_past_key_value,
                    output_attentions,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    coordinate_embeds,
                    attention_mask=causal_mask,
                    past_key_value=past_key_values,
                    spatial_past_key_value=spatial_past_key_value,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
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

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        is_attention_mask_3d = attention_mask is not None and len(attention_mask.shape) == 3
        if (
            self.config._attn_implementation == "sdpa"
            and not using_static_cache
            and not output_attentions
            and not is_attention_mask_3d
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device

        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                if len(attention_mask.shape) == 3:
                    # If the attention mask is 3D, i.e. has a shape of (batch_size, sequence_length, sequence_length),
                    # we directly expand it to 4D, i.e. (batch_size, 1, sequence_length, sequence_length).
                    mask_length2 = attention_mask.shape[-2]
                    padding_mask = causal_mask[:, :, :mask_length2, :mask_length] + attention_mask[:, None, :, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :mask_length2, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )
                else:
                    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                    padding_mask = padding_mask == 0
                    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )

        return causal_mask
