from test.modules.llama.helpers import InputSizes, ModelInputs
from typing import Tuple

import torch
from transformers.cache_utils import Cache

from docllm.modules.llama import DocLLMLlamaConfig, DocLLMLlamaDecoderLayer


def test_simple_forward_has_same_output_shape_as_input_shape(
    config: DocLLMLlamaConfig, model_inputs: ModelInputs, position_embeddings: Tuple[torch.Tensor, torch.Tensor]
):
    decoder_layer = DocLLMLlamaDecoderLayer(config, layer_idx=0)
    hidden_states, _, _, _ = decoder_layer.forward(
        model_inputs.input_embeddings, model_inputs.spatial_embeddings, position_embeddings=position_embeddings
    )
    assert hidden_states.shape == model_inputs.input_embeddings.shape


def test_output_attentions_have_correct_shape(
    config: DocLLMLlamaConfig,
    input_sizes: InputSizes,
    model_inputs: ModelInputs,
    attention_mask: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
):
    decoder_layer = DocLLMLlamaDecoderLayer(config, layer_idx=0)
    _, self_attn_weights, _, _ = decoder_layer.forward(
        model_inputs.input_embeddings,
        model_inputs.spatial_embeddings,
        attention_mask=attention_mask,
        output_attentions=True,
        position_embeddings=position_embeddings,
    )
    expected_attention_weights_shape = (
        input_sizes.batch_size,
        config.num_attention_heads,
        input_sizes.sequence_length,
        input_sizes.sequence_length,
    )
    assert self_attn_weights.shape == expected_attention_weights_shape


def test_key_value_caches_are_none_by_default(
    config: DocLLMLlamaConfig, model_inputs: ModelInputs, position_embeddings: Tuple[torch.Tensor, torch.Tensor]
):
    decoder_layer = DocLLMLlamaDecoderLayer(config, layer_idx=0)
    _, _, present_key_value, spatial_present_key_value = decoder_layer.forward(
        model_inputs.input_embeddings, model_inputs.spatial_embeddings, position_embeddings=position_embeddings
    )
    assert present_key_value is None
    assert spatial_present_key_value is None


def test_key_value_caches_are_returned(
    config: DocLLMLlamaConfig,
    model_inputs: ModelInputs,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    past_key_value: Cache,
    spatial_past_key_value: Cache,
):
    decoder_layer = DocLLMLlamaDecoderLayer(config, layer_idx=0)
    _, _, present_key_value, spatial_present_key_value = decoder_layer.forward(
        model_inputs.input_embeddings,
        model_inputs.spatial_embeddings,
        past_key_value=past_key_value,
        spatial_past_key_value=spatial_past_key_value,
        position_embeddings=position_embeddings,
    )
    assert present_key_value is not None
    assert spatial_present_key_value is not None


def test_after_freezing_llama_weights_spatial_layers_are_not_frozen(config: DocLLMLlamaConfig):
    decoder_layer = DocLLMLlamaDecoderLayer(config, layer_idx=0)
    decoder_layer.set_freeze_llama_layers(True)
    for name, param in decoder_layer.named_parameters(recurse=True):
        assert not param.requires_grad or "spatial_" in name


def test_after_unfreezing_llama_weights_everything_is_not_frozen(config: DocLLMLlamaConfig):
    decoder_layer = DocLLMLlamaDecoderLayer(config, layer_idx=0)
    decoder_layer.set_freeze_llama_layers(True)
    decoder_layer.set_freeze_llama_layers(False)
    for param in decoder_layer.parameters(recurse=True):
        assert param.requires_grad
