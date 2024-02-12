import pytest
import torch
from pydantic import BaseModel
from transformers.cache_utils import Cache, DynamicCache

from docllm.llama.config import DocLLMLlamaConfig


class InputSizes(BaseModel):
    batch_size: int
    sequence_length: int


class ModelInputs(BaseModel):
    input_embeddings: torch.Tensor
    spatial_embeddings: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


@pytest.fixture
def config() -> DocLLMLlamaConfig:
    return DocLLMLlamaConfig()


@pytest.fixture
def small_config() -> DocLLMLlamaConfig:
    return DocLLMLlamaConfig(
        hidden_size=256, num_attention_heads=4, num_hidden_layers=2, intermediate_size=1024, vocab_size=320
    )


@pytest.fixture
def input_sizes() -> InputSizes:
    return InputSizes(batch_size=2, sequence_length=10)


@pytest.fixture
def model_inputs(input_sizes, config) -> ModelInputs:
    return ModelInputs(
        input_embeddings=torch.randn(input_sizes.batch_size, input_sizes.sequence_length, config.hidden_size),
        spatial_embeddings=torch.randn(input_sizes.batch_size, input_sizes.sequence_length, config.hidden_size),
    )


@pytest.fixture
def attention_mask(input_sizes: InputSizes) -> torch.Tensor:
    return torch.ones(input_sizes.batch_size, 1, input_sizes.sequence_length, input_sizes.sequence_length).bool()


@pytest.fixture
def position_ids(input_sizes: InputSizes) -> torch.LongTensor:
    return torch.arange(input_sizes.sequence_length).unsqueeze(0).expand(input_sizes.batch_size, -1)


@pytest.fixture
def past_key_value() -> Cache:
    return DynamicCache()


@pytest.fixture
def spatial_past_key_value() -> Cache:
    return DynamicCache()
