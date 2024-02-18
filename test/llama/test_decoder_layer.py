from test.llama.helpers import InputSizes, ModelInputs

import torch
from transformers.cache_utils import Cache

from docllm.llama import DocLLMLlamaConfig, DocLLMLlamaDecoderLayer


def test_simple_forward_has_same_output_shape_as_input_shape(
    config: DocLLMLlamaConfig, model_inputs: ModelInputs, attention_mask: torch.Tensor, position_ids: torch.LongTensor
):
    decoder_layer = DocLLMLlamaDecoderLayer(config, layer_idx=0)
    hidden_states, _, _, _ = decoder_layer.forward(model_inputs.input_embeddings, model_inputs.spatial_embeddings)
    assert hidden_states.shape == model_inputs.input_embeddings.shape


def test_output_attentions_have_correct_shape(
    config: DocLLMLlamaConfig,
    input_sizes: InputSizes,
    model_inputs: ModelInputs,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
):
    decoder_layer = DocLLMLlamaDecoderLayer(config, layer_idx=0)
    _, self_attn_weights, _, _ = decoder_layer.forward(
        model_inputs.input_embeddings,
        model_inputs.spatial_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=True,
    )
    expected_attention_weights_shape = (
        input_sizes.batch_size,
        config.num_attention_heads,
        input_sizes.sequence_length,
        input_sizes.sequence_length,
    )
    assert self_attn_weights.shape == expected_attention_weights_shape


def test_key_value_caches_are_none_by_default(config: DocLLMLlamaConfig, model_inputs: ModelInputs):
    decoder_layer = DocLLMLlamaDecoderLayer(config, layer_idx=0)
    _, _, present_key_value, spatial_present_key_value = decoder_layer.forward(
        model_inputs.input_embeddings, model_inputs.spatial_embeddings
    )
    assert present_key_value is None
    assert spatial_present_key_value is None


def test_key_value_caches_are_returned(
    config: DocLLMLlamaConfig,
    model_inputs: ModelInputs,
    past_key_value: Cache,
    spatial_past_key_value: Cache,
):
    decoder_layer = DocLLMLlamaDecoderLayer(config, layer_idx=0)
    _, _, present_key_value, spatial_present_key_value = decoder_layer.forward(
        model_inputs.input_embeddings,
        model_inputs.spatial_embeddings,
        past_key_value=past_key_value,
        spatial_past_key_value=spatial_past_key_value,
    )
    assert present_key_value is not None
    assert spatial_present_key_value is not None
