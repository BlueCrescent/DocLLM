from test.modules.llama.helpers import InputSizes, ModelInputs

import torch
import torch.nn as nn
from transformers.cache_utils import Cache

from docllm.modules.llama import DocLLMAttention, DocLLMLlamaConfig
from docllm.modules.llama.config import PositionalEmbeddingMode


def test_output_shape_is_input_shape(
    config: DocLLMLlamaConfig, model_inputs: ModelInputs, position_ids: torch.LongTensor
):
    attention = DocLLMAttention(config)
    output, _, _, _ = attention(model_inputs.input_embeddings, model_inputs.spatial_embeddings, position_ids)
    assert output.shape == model_inputs.input_embeddings.shape


def test_with_attention_mask(
    config: DocLLMLlamaConfig, model_inputs: ModelInputs, position_ids: torch.LongTensor, attention_mask: torch.Tensor
):
    attention = DocLLMAttention(config)
    output, _, _, _ = attention(
        model_inputs.input_embeddings, model_inputs.spatial_embeddings, position_ids, attention_mask=attention_mask
    )
    assert output.shape == model_inputs.input_embeddings.shape


def test_with_position_ids(config: DocLLMLlamaConfig, model_inputs: ModelInputs, position_ids: torch.LongTensor):
    attention = DocLLMAttention(config)
    output, _, _, _ = attention(
        model_inputs.input_embeddings, model_inputs.spatial_embeddings, position_ids=position_ids
    )
    assert output.shape == model_inputs.input_embeddings.shape


def test_with_cache(
    config: DocLLMLlamaConfig,
    model_inputs: ModelInputs,
    position_ids: torch.LongTensor,
    past_key_value: Cache,
    spatial_past_key_value: Cache,
):
    attention = DocLLMAttention(config, 0)
    output, _, _, _ = attention(
        model_inputs.input_embeddings,
        model_inputs.spatial_embeddings,
        position_ids=position_ids,
        past_key_value=past_key_value,
        spatial_past_key_value=spatial_past_key_value,
    )
    assert output.shape == model_inputs.input_embeddings.shape


def test_output_shape_is_input_shape_with_all_parameters(
    config: DocLLMLlamaConfig,
    model_inputs: ModelInputs,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    past_key_value: Cache,
    spatial_past_key_value: Cache,
):
    attention = DocLLMAttention(config, layer_idx=0)
    output, _, _, _ = attention(
        model_inputs.input_embeddings,
        model_inputs.spatial_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        spatial_past_key_value=spatial_past_key_value,
    )
    assert output.shape == model_inputs.input_embeddings.shape


def test_computed_q_k_v_have_expected_shape(
    config: DocLLMLlamaConfig,
    input_sizes: InputSizes,
    model_inputs: ModelInputs,
    position_ids: torch.LongTensor,
    past_key_value: Cache,
):
    head_dim = config.hidden_size // config.num_attention_heads
    attention = DocLLMAttention(config, layer_idx=0)
    q_proj = k_proj = v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * head_dim)
    q, k, v, _, _, _ = attention._compute_q_k_v(
        model_inputs.input_embeddings, position_ids, past_key_value, q_proj, k_proj, v_proj
    )
    expected_shape = (input_sizes.batch_size, config.num_attention_heads, input_sizes.sequence_length, head_dim)
    assert q.shape == expected_shape
    assert k.shape == expected_shape
    assert v.shape == expected_shape


def test_after_freezing_llama_weights_spatial_layers_are_not_frozen(config: DocLLMLlamaConfig):
    attention = DocLLMAttention(config)
    attention.set_freeze_llama_layers(True)
    for name, param in attention.named_parameters(recurse=True):
        assert not param.requires_grad or name.startswith("spatial_")


def test_after_unfreezing_llama_weights_everything_is_not_frozen(config: DocLLMLlamaConfig):
    attention = DocLLMAttention(config)
    attention.set_freeze_llama_layers(True)
    attention.set_freeze_llama_layers(False)
    for param in attention.parameters(recurse=True):
        assert param.requires_grad


def test_computed_q_k_v_returns_no_cos_sin_if_pos_mode_is_none(
    config: DocLLMLlamaConfig,
    model_inputs: ModelInputs,
    position_ids: torch.LongTensor,
    past_key_value: Cache,
):
    config.positional_embedding_mode = PositionalEmbeddingMode.NONE
    head_dim = config.hidden_size // config.num_attention_heads
    attention = DocLLMAttention(config, layer_idx=0)
    q_proj = k_proj = v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * head_dim)
    _, _, _, _, cos, sin = attention._compute_q_k_v(
        model_inputs.input_embeddings, position_ids, past_key_value, q_proj, k_proj, v_proj
    )
    assert cos is None and sin is None
