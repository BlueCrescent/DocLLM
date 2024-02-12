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
