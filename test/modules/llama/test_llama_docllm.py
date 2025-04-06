import os
from tempfile import TemporaryDirectory
from typing import Tuple

import pytest
import torch
from transformers import LlamaModel

from docllm.data.precomputed_epoch_loader import compute_packing_attention_mask_and_pos_ids
from docllm.modules.llama import DocLLMLlamaConfig
from docllm.modules.llama.llama_docllm import DocLLMModelOutputWithPast, LlamaDocLLM


def test_forward(small_model: LlamaDocLLM, small_config: DocLLMLlamaConfig):
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    output = small_model.forward(input_ids, coordinates)
    assert isinstance(output, DocLLMModelOutputWithPast)
    assert output.last_hidden_state.shape == (input_ids.shape[0], input_ids.shape[1], small_config.hidden_size)


def test_get_input_embeddings(small_model: LlamaDocLLM):
    embeddings = small_model.get_input_embeddings()
    assert isinstance(embeddings, torch.nn.Embedding)


def test_set_input_embeddings(small_model: LlamaDocLLM):
    embeddings = torch.nn.Embedding(10, 20)
    small_model.set_input_embeddings(embeddings)
    assert small_model.get_input_embeddings() == embeddings


def test_after_freezing_llama_weights_spatial_layers_are_not_frozen(small_model: LlamaDocLLM):
    small_model.set_freeze_llama_layers(True)
    for name, param in small_model.named_parameters(recurse=True):
        assert not param.requires_grad or "spatial" in name or "additional" in name


def test_after_unfreezing_llama_weights_everything_is_not_frozen(small_model: LlamaDocLLM):
    small_model.set_freeze_llama_layers(True)
    small_model.set_freeze_llama_layers(False)
    for param in small_model.parameters(recurse=True):
        assert param.requires_grad


def test_loading_llama_weights_initiates_non_spatial_weights(small_config: DocLLMLlamaConfig):
    llama = LlamaModel(small_config)
    for param in llama.parameters(recurse=True):
        torch.nn.init.constant_(param, 1.0)
    with TemporaryDirectory() as dir:
        model_path = os.path.join(dir, "llama")
        llama.save_pretrained(model_path)
        model = LlamaDocLLM.from_pretrained(model_path)
        for name, param in model.named_parameters(recurse=True):
            assert (param == 1.0).all().item() ^ ("spatial" in name or "additional" in name)


def test_loading_llama_weights_does_not_touch_non_spatial_weights(small_config: DocLLMLlamaConfig):
    llama = LlamaModel(small_config)
    for param in llama.parameters(recurse=True):
        torch.nn.init.constant_(param, 1.0)
    with TemporaryDirectory() as dir:
        model_path = os.path.join(dir, "llama")
        llama.save_pretrained(model_path)
        model = LlamaDocLLM.from_pretrained(model_path)
        for name, param in model.named_parameters(recurse=True):
            assert (param != 1.0).all().item() ^ ("spatial" not in name and "additional" not in name)


def test_fuse_additional_embeddings_removes_additional_layers(small_model: LlamaDocLLM):
    small_model.fuse_additional_embeddings()
    for name, _ in small_model.named_parameters(recurse=True):
        assert "additional" not in name


def test_fuse_additional_embeddings_changes_config(small_model: LlamaDocLLM, small_config: DocLLMLlamaConfig):
    expected_vocab_size_after_fuse = small_config.vocab_size + small_config.additional_training_vocab_size
    small_model.fuse_additional_embeddings()
    assert small_model.config.vocab_size == expected_vocab_size_after_fuse
    assert small_model.config.additional_training_vocab_size == 0


def test_after_fusing_additional_embeddings_results_for_original_tokens_are_same(
    small_model: LlamaDocLLM, small_config: DocLLMLlamaConfig
):
    input = torch.randint(0, small_config.vocab_size, (2, 3))
    spatial_input = torch.rand((2, 3, 4))
    output = small_model(input, spatial_input)
    small_model.fuse_additional_embeddings()
    fused_output = small_model(input, spatial_input)
    assert torch.allclose(output.last_hidden_state, fused_output.last_hidden_state)


def test_after_fusing_additional_embeddings_results_for_new_tokens_are_same(
    small_model: LlamaDocLLM, small_config: DocLLMLlamaConfig
):
    input = torch.randint(
        small_config.vocab_size, small_config.vocab_size + small_config.additional_training_vocab_size, (2, 3)
    )
    spatial_input = torch.rand((2, 3, 4))
    output = small_model(input, spatial_input)
    small_model.fuse_additional_embeddings()
    fused_output = small_model(input, spatial_input)
    assert torch.allclose(output.last_hidden_state, fused_output.last_hidden_state)


def test_forward_with_3d_attention_mask(small_model: LlamaDocLLM, small_config: DocLLMLlamaConfig):
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    attention_mask = torch.randint(0, 2, (input_ids.size(0), input_ids.size(1), input_ids.size(1)), dtype=torch.bool)
    output = small_model.forward(input_ids, coordinates, attention_mask=attention_mask)
    assert isinstance(output, DocLLMModelOutputWithPast)
    assert output.last_hidden_state.shape == (input_ids.shape[0], input_ids.shape[1], small_config.hidden_size)


def test_forward_with_3d_attention_causal_mask_does_not_change_results(small_model: LlamaDocLLM):
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    attention_mask = torch.tril(
        torch.ones((input_ids.size(0), input_ids.size(1), input_ids.size(1) + 1), dtype=torch.bool)
    )
    output = small_model.forward(input_ids, coordinates)
    output_masked = small_model.forward(input_ids, coordinates, attention_mask=attention_mask)
    assert torch.allclose(output.last_hidden_state, output_masked.last_hidden_state)


def test_forward_with_3d_attention_attent_everything_mask_does_not_change_results(small_model: LlamaDocLLM):
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    attention_mask = torch.ones((input_ids.size(0), input_ids.size(1), input_ids.size(1)), dtype=torch.bool)
    output = small_model.forward(input_ids, coordinates)
    output_masked = small_model.forward(input_ids, coordinates, attention_mask=attention_mask)
    assert torch.allclose(output.last_hidden_state, output_masked.last_hidden_state)


def test_forward_with_3d_attention_mask_different_than_without_mask(small_model: LlamaDocLLM):
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    attention_mask = torch.tril(torch.ones((input_ids.size(0), input_ids.size(1), input_ids.size(1)), dtype=torch.bool))
    attention_mask[0, input_ids.size(1) - 1, 0] = 0
    output = small_model.forward(input_ids, coordinates)
    output_masked = small_model.forward(input_ids, coordinates, attention_mask=attention_mask)
    assert not torch.allclose(output.last_hidden_state, output_masked.last_hidden_state)


def test_update_causal_mask_produces_expected_shape(updated_mask: torch.Tensor, seq_len: int):
    # The "+ 1" is caused by the following:
    # An additional column is required by for producing the the same shape as the one produced by LlamaModel
    # when not setting the attention_mask. This column will be ignored.
    assert updated_mask.shape == (2, 1, seq_len, seq_len + 1)


def test_update_causal_mask_produces_expected_dtype(updated_mask: torch.Tensor):
    assert updated_mask.dtype == torch.float32


def test_update_causal_mask_with_3d_lower_triangle_is_same_as_no_mask(
    inputs_embeds: torch.Tensor, small_model: LlamaDocLLM, cache_position: torch.Tensor
):
    attention_mask = torch.tril(
        torch.ones((inputs_embeds.size(0), inputs_embeds.size(1), inputs_embeds.size(1) + 1), dtype=torch.bool)
    )
    updated_mask = small_model._update_causal_mask(
        attention_mask, inputs_embeds, cache_position=cache_position, past_key_values=None, output_attentions=False
    )
    updated_mask_no_mask = small_model._update_causal_mask(
        None, inputs_embeds, cache_position=cache_position, past_key_values=None, output_attentions=False
    )
    assert updated_mask.shape == updated_mask_no_mask.shape
    assert torch.allclose(updated_mask, updated_mask_no_mask)


@pytest.fixture
def small_model(small_config: DocLLMLlamaConfig) -> LlamaDocLLM:
    return LlamaDocLLM(small_config)


@pytest.fixture
def updated_mask(
    small_model: LlamaDocLLM,
    cache_position: torch.Tensor,
    sizes: Tuple[torch.LongTensor, torch.LongTensor],
    inputs_embeds: torch.Tensor,
) -> torch.Tensor:
    sizes1, sizes2 = sizes
    attention_mask1, _ = compute_packing_attention_mask_and_pos_ids(sizes1)
    attention_mask2, _ = compute_packing_attention_mask_and_pos_ids(sizes2)
    attention_mask2 = torch.cat([attention_mask2, torch.zeros(attention_mask2.size(0), 1, dtype=torch.bool)], dim=1)
    attention_mask2 = torch.cat([attention_mask2, torch.zeros(1, attention_mask2.size(1), dtype=torch.bool)], dim=0)
    attention_mask = torch.stack([attention_mask1, attention_mask2])
    # An additional column is required by for producing the the same shape as the one produced by LlamaModel
    # when not setting the attention_mask. This column will be ignored.
    attention_mask = torch.cat([attention_mask, torch.zeros(2, attention_mask.size(1), 1, dtype=torch.bool)], dim=2)
    return small_model._update_causal_mask(
        attention_mask, inputs_embeds, cache_position=cache_position, past_key_values=None, output_attentions=False
    )


@pytest.fixture
def cache_position(inputs_embeds: torch.Tensor) -> torch.Tensor:
    return torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)


@pytest.fixture
def inputs_embeds(small_config: DocLLMLlamaConfig, seq_len: int) -> torch.Tensor:
    return torch.rand((2, seq_len, small_config.hidden_size))


@pytest.fixture
def seq_len(sizes: Tuple[torch.LongTensor, torch.LongTensor]) -> int:
    return max(size.sum().item() for size in sizes)


@pytest.fixture
def sizes() -> Tuple[torch.LongTensor, torch.LongTensor]:
    sizes1 = torch.tensor([3, 2, 1])
    sizes2 = torch.tensor([2, 3])
    return sizes1, sizes2
