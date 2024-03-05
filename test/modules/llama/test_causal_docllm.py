import os
from tempfile import TemporaryDirectory

import pytest
import torch
from transformers import LlamaForCausalLM

from docllm.modules.llama.causal_docllm import CausalDocLLMOutputWithPast, CausalLlamaDocLLM
from docllm.modules.llama.config import DocLLMLlamaConfig


@pytest.fixture
def small_model(small_config: DocLLMLlamaConfig) -> CausalLlamaDocLLM:
    return CausalLlamaDocLLM(small_config)


def test_initialization(small_model: CausalLlamaDocLLM):
    assert isinstance(small_model, CausalLlamaDocLLM)


def test_forward_result_has_expected_type(small_model: CausalLlamaDocLLM):
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    output = small_model.forward(input_ids, coordinates)
    assert isinstance(output, CausalDocLLMOutputWithPast)


def test_forward_result_has_expected_logits_shape(small_model: CausalLlamaDocLLM, small_config: DocLLMLlamaConfig):
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    output = small_model.forward(input_ids, coordinates)
    assert output.logits.shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        small_config.vocab_size + small_config.additional_training_vocab_size,
    )


def test_loss_result_has_expected_shape(small_model: CausalLlamaDocLLM):
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    labels = torch.tensor([[2, 3, 4]])
    loss_mask = torch.tensor([[0, 1, 1]])
    output = small_model.forward(input_ids, coordinates, labels=labels, loss_mask=loss_mask)
    assert output.loss.shape == ()


def test_loss_is_zero_with_loss_mask_zero(small_model: CausalLlamaDocLLM):
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    labels = torch.tensor([[2, 3, 4]])
    loss_mask = torch.tensor([[0, 0, 0]])
    output = small_model.forward(input_ids, coordinates, labels=labels, loss_mask=loss_mask)
    assert output.loss == 0.0


def test_loss_is_not_zero_with_loss_mask_one(small_model: CausalLlamaDocLLM):
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    labels = torch.tensor([[2, 3, 4]])
    loss_mask = torch.tensor([[1, 1, 1]])
    output = small_model.forward(input_ids, coordinates, labels=labels, loss_mask=loss_mask)
    assert output.loss != 0.0


def test_after_freezing_llama_weights_spatial_layers_are_not_frozen(small_model: CausalLlamaDocLLM):
    small_model.set_freeze_llama_layers(True)
    for name, param in small_model.named_parameters(recurse=True):
        assert not param.requires_grad or "spatial" in name or "additional" in name


def test_after_unfreezing_llama_weights_everything_is_not_frozen(small_model: CausalLlamaDocLLM):
    small_model.set_freeze_llama_layers(True)
    small_model.set_freeze_llama_layers(False)
    for param in small_model.parameters(recurse=True):
        assert param.requires_grad


def test_loading_llama_weights_initiates_non_spatial_weights(small_config: DocLLMLlamaConfig):
    llama = LlamaForCausalLM(small_config)
    for param in llama.parameters(recurse=True):
        torch.nn.init.constant_(param, 1.0)
    with TemporaryDirectory() as dir:
        model_path = os.path.join(dir, "llama")
        llama.save_pretrained(model_path)
        model = CausalLlamaDocLLM.from_pretrained(model_path)
        for name, param in model.named_parameters(recurse=True):
            assert (param == 1.0).all().item() ^ ("spatial" in name or "additional" in name)


def test_loading_llama_weights_does_not_touch_non_spatial_weights(small_config: DocLLMLlamaConfig):
    llama = LlamaForCausalLM(small_config)
    for param in llama.parameters(recurse=True):
        torch.nn.init.constant_(param, 1.0)
    with TemporaryDirectory() as dir:
        model_path = os.path.join(dir, "llama")
        llama.save_pretrained(model_path)
        model = CausalLlamaDocLLM.from_pretrained(model_path)
        for name, param in model.named_parameters(recurse=True):
            assert (param != 1.0).all().item() ^ ("spatial" not in name and "additional" not in name)


def test_fuse_additional_embeddings_removes_additional_layers(small_model: CausalLlamaDocLLM):
    small_model.fuse_additional_embeddings()
    for name, _ in small_model.named_parameters(recurse=True):
        assert "additional" not in name


def test_fuse_additional_embeddings_changes_config(small_model: CausalLlamaDocLLM, small_config: DocLLMLlamaConfig):
    expected_vocab_size_after_fuse = small_config.vocab_size + small_config.additional_training_vocab_size
    small_model.fuse_additional_embeddings()
    assert small_model.config.vocab_size == expected_vocab_size_after_fuse
    assert small_model.config.additional_training_vocab_size == 0


def test_after_fusing_additional_embeddings_results_for_original_tokens_are_same(
    small_model: CausalLlamaDocLLM, small_config: DocLLMLlamaConfig
):
    input = torch.randint(0, small_config.vocab_size, (2, 3))
    spatial_input = torch.rand((2, 3, 4))
    output = small_model(input, spatial_input)
    small_model.fuse_additional_embeddings()
    fused_output = small_model(input, spatial_input)
    assert torch.allclose(output.logits, fused_output.logits)


def test_after_fusing_additional_embeddings_results_for_new_tokens_are_same(
    small_model: CausalLlamaDocLLM, small_config: DocLLMLlamaConfig
):
    input = torch.randint(
        small_config.vocab_size, small_config.vocab_size + small_config.additional_training_vocab_size, (2, 3)
    )
    spatial_input = torch.rand((2, 3, 4))
    output = small_model(input, spatial_input)
    small_model.fuse_additional_embeddings()
    fused_output = small_model(input, spatial_input)
    assert torch.allclose(output.logits, fused_output.logits)
