import os
from tempfile import TemporaryDirectory

import pytest
import torch
from transformers import LlamaModel

from docllm.modules.llama import DocLLMLlamaConfig
from docllm.modules.llama.llama_docllm import DocLLMModelOutputWithPast, LlamaDocLLM


@pytest.fixture
def small_model(small_config: DocLLMLlamaConfig) -> LlamaDocLLM:
    return LlamaDocLLM(small_config)


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
