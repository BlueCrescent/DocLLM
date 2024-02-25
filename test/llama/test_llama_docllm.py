import torch

from docllm.llama import DocLLMLlamaConfig
from docllm.llama.llama_docllm import DocLLMModelOutputWithPast, LlamaDocLLM


def test_forward(small_config: DocLLMLlamaConfig):
    model = LlamaDocLLM(small_config)
    input_ids = torch.tensor([[1, 2, 3]])
    coordinates = torch.tensor([[[0.1, 0.0, 0.4, 1.0], [0.2, 0.1, 0.3, 0.2], [0.3, 0.2, 0.2, 0.3]]])
    output = model.forward(input_ids, coordinates)
    assert isinstance(output, DocLLMModelOutputWithPast)
    assert output.last_hidden_state.shape == (input_ids.shape[0], input_ids.shape[1], small_config.hidden_size)


def test_get_input_embeddings(small_config: DocLLMLlamaConfig):
    model = LlamaDocLLM(small_config)
    embeddings = model.get_input_embeddings()
    assert isinstance(embeddings, torch.nn.Embedding)


def test_set_input_embeddings(small_config: DocLLMLlamaConfig):
    model = LlamaDocLLM(small_config)
    embeddings = torch.nn.Embedding(10, 20)
    model.set_input_embeddings(embeddings)
    assert model.get_input_embeddings() == embeddings


def test_after_freezing_llama_weights_spatial_layers_are_not_frozen(small_config: DocLLMLlamaConfig):
    model = LlamaDocLLM(small_config)
    model.set_freeze_llama_layers(True)
    for name, param in model.named_parameters(recurse=True):
        assert not param.requires_grad or "spatial" in name


def test_after_unfreezing_llama_weights_everything_is_not_frozen(small_config: DocLLMLlamaConfig):
    model = LlamaDocLLM(small_config)
    model.set_freeze_llama_layers(True)
    model.set_freeze_llama_layers(False)
    for param in model.parameters(recurse=True):
        assert param.requires_grad
