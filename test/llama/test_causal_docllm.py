import torch

from docllm.llama.causal_docllm import CausalDocLLMOutputWithPast, CausalLlamaDocLLM
from docllm.llama.config import DocLLMLlamaConfig


def test_initialization(small_config: DocLLMLlamaConfig):
    model = CausalLlamaDocLLM(small_config)
    assert isinstance(model, CausalLlamaDocLLM)


def test_forward_result_has_expected_type(small_config: DocLLMLlamaConfig):
    model = CausalLlamaDocLLM(small_config)
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    output = model.forward(input_ids, coordinates)
    assert isinstance(output, CausalDocLLMOutputWithPast)


def test_forward_result_has_expected_logits_shape(small_config: DocLLMLlamaConfig):
    model = CausalLlamaDocLLM(small_config)
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    output = model.forward(input_ids, coordinates)
    assert output.logits.shape == (input_ids.shape[0], input_ids.shape[1], small_config.vocab_size)


def test_loss_result_has_expected_shape(small_config: DocLLMLlamaConfig):
    model = CausalLlamaDocLLM(small_config)
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    labels = torch.tensor([[2, 3, 4]])
    loss_mask = torch.tensor([[0, 1, 1]])
    output = model.forward(input_ids, coordinates, labels=labels, loss_mask=loss_mask)
    assert output.loss.shape == ()


def test_loss_is_zero_with_loss_mask_zero(small_config: DocLLMLlamaConfig):
    model = CausalLlamaDocLLM(small_config)
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    labels = torch.tensor([[2, 3, 4]])
    loss_mask = torch.tensor([[0, 0, 0]])
    output = model.forward(input_ids, coordinates, labels=labels, loss_mask=loss_mask)
    assert output.loss == 0.0


def test_loss_is_not_zero_with_loss_mask_one(small_config: DocLLMLlamaConfig):
    model = CausalLlamaDocLLM(small_config)
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    labels = torch.tensor([[2, 3, 4]])
    loss_mask = torch.tensor([[1, 1, 1]])
    output = model.forward(input_ids, coordinates, labels=labels, loss_mask=loss_mask)
    assert output.loss != 0.0


def test_after_freezing_llama_weights_spatial_layers_are_not_frozen(small_config: DocLLMLlamaConfig):
    model = CausalLlamaDocLLM(small_config)
    model.set_freeze_llama_layers(True)
    for name, param in model.named_parameters(recurse=True):
        assert not param.requires_grad or "spatial" in name


def test_after_unfreezing_llama_weights_everything_is_not_frozen(small_config: DocLLMLlamaConfig):
    model = CausalLlamaDocLLM(small_config)
    model.set_freeze_llama_layers(True)
    model.set_freeze_llama_layers(False)
    for param in model.parameters(recurse=True):
        assert param.requires_grad
